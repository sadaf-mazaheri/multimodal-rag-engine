-- Metadata store for the multimodal RAG benchmark.
--
-- Design note: Postgres is the single source of truth for *what* exists in the
-- corpus and where it came from (document -> page -> element -> chunk). Vector
-- stores hold only ids + embeddings, so provenance is never duplicated and a
-- retrieval hit can always be resolved back to a page and a bounding box.
--
-- Metadata is stored as typed columns wherever it will be filtered or ranked
-- on, and in JSONB only for genuinely open-ended extras. A field that lives in
-- JSONB cannot be indexed or constrained cheaply, so anything the retrieval
-- layer needs to condition on gets a real column.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- Documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    source            TEXT,              -- publisher / issuing body
    source_url        TEXT,
    authors           TEXT[] NOT NULL DEFAULT '{}',
    organization      TEXT,
    publication_date  DATE,
    version           TEXT,
    doc_type          TEXT NOT NULL DEFAULT 'other',
    domain            TEXT,              -- finance / science / policy / technical / ...
    language          TEXT NOT NULL DEFAULT 'en',
    license           TEXT,

    file_name         TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    file_size_bytes   BIGINT,

    n_pages           INTEGER NOT NULL CHECK (n_pages >= 0),
    n_pages_ingested  INTEGER NOT NULL CHECK (n_pages_ingested >= 0),

    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    parser_version    TEXT,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT documents_ingested_pages_fit CHECK (n_pages_ingested <= n_pages)
);

CREATE INDEX IF NOT EXISTS documents_domain_idx    ON documents (domain);
CREATE INDEX IF NOT EXISTS documents_type_idx      ON documents (doc_type);
CREATE INDEX IF NOT EXISTS documents_pubdate_idx   ON documents (publication_date);
CREATE INDEX IF NOT EXISTS documents_language_idx  ON documents (language);

-- ---------------------------------------------------------------------------
-- Pages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pages (
    page_id       TEXT PRIMARY KEY,                  -- "{doc_id}#p{page_number}"
    doc_id        TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL CHECK (page_number >= 1),  -- as printed in a PDF reader

    width         REAL NOT NULL CHECK (width > 0),   -- PDF points
    height        REAL NOT NULL CHECK (height > 0),
    rotation      INTEGER NOT NULL DEFAULT 0 CHECK (rotation IN (0, 90, 180, 270)),

    image_path    TEXT,                              -- rendered page PNG, used by Method 3
    image_width   INTEGER,
    image_height  INTEGER,
    image_dpi     INTEGER,

    -- Running section at this point in the document, carried forward from the
    -- last heading seen. Lets a mid-chapter page still be filtered by chapter.
    section       TEXT,
    subsection    TEXT,
    header_text   TEXT,
    footer_text   TEXT,

    raw_text      TEXT,                              -- full page text, for page-level fallback
    n_elements    INTEGER NOT NULL DEFAULT 0,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,

    UNIQUE (doc_id, page_number)
);

CREATE INDEX IF NOT EXISTS pages_doc_idx     ON pages (doc_id, page_number);
CREATE INDEX IF NOT EXISTS pages_section_idx ON pages (doc_id, section);

-- ---------------------------------------------------------------------------
-- Elements: the atomic units of the corpus, one row per text block / table /
-- figure. All three methods index *these same rows* differently, which is what
-- makes the comparison fair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS elements (
    element_id      TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_id         TEXT NOT NULL REFERENCES pages (page_id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL CHECK (page_number >= 1),
    element_type    TEXT NOT NULL,

    -- Parent *element* (caption -> figure), not the page. The page link is
    -- page_id, which every element always has. Self-referential, so a deleted
    -- figure orphans rather than cascades away its caption's text.
    parent_id       TEXT REFERENCES elements (element_id) ON DELETE SET NULL,
    reading_order   INTEGER NOT NULL DEFAULT 0 CHECK (reading_order >= 0),

    -- Normalised bounding box in [0, 1] relative to page width/height, so it
    -- stays valid regardless of the DPI a page image was rendered at.
    bbox_x0         REAL CHECK (bbox_x0 BETWEEN 0 AND 1),
    bbox_y0         REAL CHECK (bbox_y0 BETWEEN 0 AND 1),
    bbox_x1         REAL CHECK (bbox_x1 BETWEEN 0 AND 1),
    bbox_y1         REAL CHECK (bbox_y1 BETWEEN 0 AND 1),

    section         TEXT,
    subsection      TEXT,

    extraction_method     TEXT NOT NULL DEFAULT 'pymupdf_text',
    extraction_confidence REAL NOT NULL DEFAULT 1.0
                          CHECK (extraction_confidence BETWEEN 0 AND 1),

    text            TEXT,                   -- literal extracted text
    caption         TEXT,                   -- caption, for both tables and figures

    -- Structured modality payloads. JSONB rather than columns because their
    -- shape differs per modality; the fields that get filtered on are lifted
    -- into the generated columns below.
    table_data      JSONB,
    figure_data     JSONB,

    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT elements_bbox_ordered CHECK (
        (bbox_x0 IS NULL AND bbox_y0 IS NULL AND bbox_x1 IS NULL AND bbox_y1 IS NULL)
        OR (bbox_x1 >= bbox_x0 AND bbox_y1 >= bbox_y0)
    ),
    CONSTRAINT elements_no_self_parent CHECK (parent_id IS NULL OR parent_id <> element_id),
    -- A table payload only ever belongs on a table, and a figure payload only
    -- on a visual element. Enforced here as well as in Pydantic, because the
    -- database outlives any one version of the code.
    CONSTRAINT elements_payload_matches_type CHECK (
        (table_data IS NULL OR element_type = 'table')
        AND (figure_data IS NULL OR element_type IN ('figure', 'chart', 'diagram'))
    )
);

CREATE INDEX IF NOT EXISTS elements_doc_page_idx ON elements (doc_id, page_number);
CREATE INDEX IF NOT EXISTS elements_type_idx     ON elements (element_type);
CREATE INDEX IF NOT EXISTS elements_page_idx     ON elements (page_id);
CREATE INDEX IF NOT EXISTS elements_parent_idx   ON elements (parent_id);
CREATE INDEX IF NOT EXISTS elements_method_idx   ON elements (extraction_method);
CREATE INDEX IF NOT EXISTS elements_section_idx  ON elements (doc_id, section);
-- Reading order within a page: the access pattern for context expansion
-- ("give me the two elements either side of this hit").
CREATE INDEX IF NOT EXISTS elements_reading_idx  ON elements (page_id, reading_order);

-- Filterable projections of the structured payloads.
CREATE INDEX IF NOT EXISTS elements_table_type_idx
    ON elements ((table_data ->> 'table_type')) WHERE table_data IS NOT NULL;
CREATE INDEX IF NOT EXISTS elements_figure_type_idx
    ON elements ((figure_data ->> 'figure_type')) WHERE figure_data IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Chunks: retrieval units (built in Step 3). A chunk points back at the
-- element(s) it was built from, which is how an answer citation resolves to a
-- bounding box on a page.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL CHECK (page_number >= 1),
    chunk_type    TEXT NOT NULL,
    text          TEXT NOT NULL,           -- what actually gets embedded / BM25'd
    element_ids   TEXT[] NOT NULL DEFAULT '{}',

    bbox_x0       REAL, bbox_y0 REAL, bbox_x1 REAL, bbox_y1 REAL,
    section       TEXT,
    subsection    TEXT,
    token_count   INTEGER,

    -- Which indexing strategy produced this chunk, so Method 1 and Method 2
    -- chunk sets can live side by side without colliding.
    variant       TEXT NOT NULL DEFAULT 'default',
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS chunks_variant_idx ON chunks (variant);
CREATE INDEX IF NOT EXISTS chunks_doc_idx     ON chunks (doc_id, page_number);
CREATE INDEX IF NOT EXISTS chunks_type_idx    ON chunks (chunk_type);
CREATE INDEX IF NOT EXISTS chunks_elements_idx ON chunks USING GIN (element_ids);

-- Postgres-native full-text search, used as a sanity cross-check against the
-- in-process BM25 index and as the metadata-filtered retrieval path in Method 2.
CREATE INDEX IF NOT EXISTS chunks_fts_idx
    ON chunks USING GIN (to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS chunks_trgm_idx
    ON chunks USING GIN (text gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- Convenience view: the full provenance chain in one place, for debugging and
-- error analysis ("show me every low-confidence figure on a landscape page").
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW element_provenance AS
SELECT
    e.element_id,
    e.element_type,
    e.parent_id,
    e.reading_order,
    e.extraction_method,
    e.extraction_confidence,
    COALESCE(e.section, p.section)       AS section,
    COALESCE(e.subsection, p.subsection) AS subsection,
    e.bbox_x0, e.bbox_y0, e.bbox_x1, e.bbox_y1,
    p.page_id,
    p.page_number,
    p.image_path                         AS page_image_path,
    p.width                              AS page_width,
    p.height                             AS page_height,
    p.rotation                           AS page_rotation,
    d.doc_id,
    d.title                              AS doc_title,
    d.source_url,
    d.doc_type,
    d.domain,
    d.publication_date
FROM elements e
JOIN pages     p ON p.page_id = e.page_id
JOIN documents d ON d.doc_id  = e.doc_id;
