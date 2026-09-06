-- Metadata store for the multimodal RAG benchmark.
--
-- Design note: Postgres is the single source of truth for *what* exists in the
-- corpus and where it came from (document -> page -> element -> chunk). Vector
-- stores hold only ids + embeddings, so provenance is never duplicated and a
-- retrieval hit can always be resolved back to a page and bounding box.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- Documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source_url      TEXT,
    publisher       TEXT,
    category        TEXT,            -- e.g. finance / science / policy / technical
    license         TEXT,
    sha256          TEXT NOT NULL,
    n_pages         INTEGER NOT NULL,
    file_path       TEXT NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS documents_category_idx ON documents (category);

-- ---------------------------------------------------------------------------
-- Pages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pages (
    page_id         TEXT PRIMARY KEY,          -- "{doc_id}#p{page_number}"
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,          -- 1-indexed, as printed in a PDF reader
    width           REAL NOT NULL,             -- PDF points
    height          REAL NOT NULL,
    image_path      TEXT,                      -- rendered page PNG, used by Method 3
    raw_text        TEXT,                      -- full page text, for page-level fallbacks
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (doc_id, page_number)
);

CREATE INDEX IF NOT EXISTS pages_doc_idx ON pages (doc_id, page_number);

-- ---------------------------------------------------------------------------
-- Elements: the atomic units of the corpus, one row per text block / table /
-- figure. All three methods index *these same rows* differently, which is what
-- makes the comparison fair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS elements (
    element_id      TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_id         TEXT NOT NULL REFERENCES pages (page_id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    element_type    TEXT NOT NULL,             -- text | title | table | figure | chart | diagram
    reading_order   INTEGER NOT NULL DEFAULT 0,

    -- Normalised bounding box in [0, 1] relative to page width/height, so it
    -- stays valid regardless of the DPI a page image was rendered at.
    bbox_x0         REAL, bbox_y0 REAL, bbox_x1 REAL, bbox_y1 REAL,

    text            TEXT,                      -- literal extracted text
    table_markdown  TEXT,                      -- tables: Markdown serialisation
    table_json      JSONB,                     -- tables: structured rows/headers
    image_path      TEXT,                      -- figures: cropped image on disk
    caption         TEXT,                      -- figure caption found in the layout
    ocr_text        TEXT,                      -- text recovered from the image itself
    description     TEXT,                      -- VLM-generated description

    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS elements_doc_page_idx ON elements (doc_id, page_number);
CREATE INDEX IF NOT EXISTS elements_type_idx     ON elements (element_type);
CREATE INDEX IF NOT EXISTS elements_page_idx     ON elements (page_id);

-- ---------------------------------------------------------------------------
-- Chunks: retrieval units. A chunk points back at the element(s) it was built
-- from, which is how an answer citation resolves to a bounding box on a page.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    chunk_type      TEXT NOT NULL,             -- text | table | figure | page
    text            TEXT NOT NULL,             -- what actually gets embedded / BM25'd
    element_ids     TEXT[] NOT NULL DEFAULT '{}',
    token_count     INTEGER,
    -- Which indexing strategy produced this chunk, so Method 1 and Method 2
    -- chunk sets can live side by side without colliding.
    variant         TEXT NOT NULL DEFAULT 'default',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS chunks_variant_idx  ON chunks (variant);
CREATE INDEX IF NOT EXISTS chunks_doc_idx      ON chunks (doc_id, page_number);
CREATE INDEX IF NOT EXISTS chunks_type_idx     ON chunks (chunk_type);

-- Postgres-native full-text search, used as a sanity cross-check against the
-- in-process BM25 index and as the metadata-filtered retrieval path in Method 2.
CREATE INDEX IF NOT EXISTS chunks_fts_idx
    ON chunks USING GIN (to_tsvector('english', text));

CREATE INDEX IF NOT EXISTS chunks_trgm_idx
    ON chunks USING GIN (text gin_trgm_ops);
