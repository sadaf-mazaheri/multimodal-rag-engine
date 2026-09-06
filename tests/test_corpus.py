"""Tests for the corpus manifest, lockfile, and downloader.

None of these touch the network. The downloader is exercised against a fake
``httpx`` transport so the failure modes that actually matter -- an HTML error
page served with a 200, a truncated stream, a changed upstream document -- are
tested deterministically rather than hoped about.
"""

from __future__ import annotations

import httpx
import pytest

from mmrag.corpus.download import _looks_like_pdf, download_entry, lock_manifest, sha256_file
from mmrag.corpus.manifest import (
    CorpusEntry,
    CorpusLock,
    CorpusLockEntry,
    CorpusManifest,
)

PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n%%EOF\n"
HTML_BYTES = b"<!doctype html><html><body>403 Forbidden</body></html>"


def _entry(**kw) -> CorpusEntry:
    defaults = dict(doc_id="testdoc", title="Test", url="https://example.invalid/x.pdf")
    return CorpusEntry(**{**defaults, **kw})


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------


class TestCorpusEntry:
    def test_doc_id_pattern_rejects_unsafe_names(self):
        """doc_id becomes a filename and an id prefix, so keep it boring."""
        for bad in ("Has Caps", "has/slash", "-leading-dash", "has.dot", ""):
            with pytest.raises(ValueError):
                _entry(doc_id=bad)

    def test_filename_and_local_path(self, tmp_path):
        e = _entry(doc_id="abc")
        assert e.filename == "abc.pdf"
        assert e.local_path(tmp_path) == tmp_path / "abc.pdf"

    def test_sha256_must_be_hex(self):
        with pytest.raises(ValueError, match="64 hex"):
            _entry(sha256="nope")
        with pytest.raises(ValueError, match="64 hex"):
            _entry(sha256="z" * 64)

    def test_sha256_is_normalised_to_lowercase(self):
        assert _entry(sha256="A" * 64).sha256 == "a" * 64

    def test_effective_pages_applies_page_limit(self):
        assert _entry(n_pages=642, page_limit=120).effective_pages == 120
        assert _entry(n_pages=50, page_limit=120).effective_pages == 50
        assert _entry(n_pages=50).effective_pages == 50
        assert _entry().effective_pages is None


class TestCorpusManifest:
    def test_duplicate_doc_ids_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate doc_id"):
            CorpusManifest(entries=[_entry(doc_id="a"), _entry(doc_id="a")])

    def test_active_excludes_disabled(self):
        m = CorpusManifest(entries=[_entry(doc_id="a"), _entry(doc_id="b", enabled=False)])
        assert [e.doc_id for e in m.active] == ["a"]

    def test_get_raises_a_helpful_error(self):
        m = CorpusManifest(entries=[_entry(doc_id="a")])
        with pytest.raises(KeyError, match="Known ids: a"):
            m.get("nope")

    def test_apply_lock_overlays_integrity_data(self):
        m = CorpusManifest(entries=[_entry(doc_id="a")])
        m.apply_lock(
            CorpusLock(entries={"a": CorpusLockEntry(sha256="a" * 64, size_bytes=10, n_pages=3)})
        )
        assert (m.get("a").sha256, m.get("a").size_bytes, m.get("a").n_pages) == ("a" * 64, 10, 3)

    def test_apply_lock_ignores_unknown_ids(self):
        m = CorpusManifest(entries=[_entry(doc_id="a")])
        m.apply_lock(CorpusLock(entries={"other": CorpusLockEntry(sha256="b" * 64, size_bytes=1)}))
        assert m.get("a").sha256 is None


class TestCommittedManifest:
    """The real configs/corpus.yaml and its lockfile."""

    def test_loads(self):
        assert len(CorpusManifest.load().active) > 0

    def test_every_entry_is_pinned(self):
        """An unpinned entry silently disables integrity checking for it."""
        unpinned = [e.doc_id for e in CorpusManifest.load().active if e.sha256 is None]
        assert unpinned == [], f"run 'mmrag corpus lock' for: {unpinned}"

    def test_every_entry_declares_a_licence_and_modalities(self):
        for e in CorpusManifest.load().active:
            assert e.license, f"{e.doc_id} has no licence recorded"
            assert e.modality_profile, f"{e.doc_id} declares no modality profile"

    def test_urls_are_https(self):
        for e in CorpusManifest.load().entries:
            assert e.url.startswith("https://"), f"{e.doc_id} is not fetched over https"

    def test_corpus_covers_every_modality_the_system_claims_to_handle(self):
        seen = {t for e in CorpusManifest.load().active for t in e.modality_profile}
        for required in ("tables", "charts", "diagrams", "dense_text", "multi_column"):
            assert required in seen, f"no corpus document stresses {required}"


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


class TestLockfile:
    def test_round_trips(self, tmp_path):
        p = tmp_path / "corpus.lock.yaml"
        CorpusLock(entries={"a": CorpusLockEntry(sha256="c" * 64, size_bytes=5, n_pages=2)}).save(p)
        loaded = CorpusLock.load(p)
        assert loaded.entries["a"].sha256 == "c" * 64
        assert loaded.generated_at  # stamped on save

    def test_saved_file_carries_a_do_not_edit_header(self, tmp_path):
        p = tmp_path / "l.yaml"
        CorpusLock().save(p)
        assert "do not edit by hand" in p.read_text(encoding="utf-8")

    def test_locking_preserves_pins_for_absent_documents(self, tmp_path):
        """Locking one re-downloaded file must not unpin the rest of the corpus."""
        raw, lock_path = tmp_path / "raw", tmp_path / "corpus.lock.yaml"
        raw.mkdir()
        (raw / "present.pdf").write_bytes(PDF_BYTES)

        CorpusLock(
            entries={
                "present": CorpusLockEntry(sha256="0" * 64, size_bytes=1, n_pages=1),
                "absent": CorpusLockEntry(sha256="f" * 64, size_bytes=99, n_pages=7),
            }
        ).save(lock_path)

        manifest = CorpusManifest(entries=[_entry(doc_id="present"), _entry(doc_id="absent")])
        lock, updated = lock_manifest(manifest, raw_dir=raw, lock_path=lock_path)

        assert updated == ["present"]
        assert lock.entries["present"].sha256 == sha256_file(raw / "present.pdf")
        assert lock.entries["absent"].sha256 == "f" * 64  # untouched


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class TestDownloader:
    def test_pdf_sniffing(self, tmp_path):
        pdf, html = tmp_path / "a", tmp_path / "b"
        pdf.write_bytes(PDF_BYTES)
        html.write_bytes(HTML_BYTES)
        assert _looks_like_pdf(pdf)
        assert not _looks_like_pdf(html)

    def test_sniffing_tolerates_a_leading_bom(self, tmp_path):
        p = tmp_path / "a"
        p.write_bytes(b"\xef\xbb\xbf\n" + PDF_BYTES)
        assert _looks_like_pdf(p)

    def test_successful_download(self, tmp_path):
        with _client(lambda _: httpx.Response(200, content=PDF_BYTES)) as c:
            r = download_entry(_entry(), raw_dir=tmp_path, client=c)
        assert r.status == "downloaded" and r.ok
        assert r.path is not None and r.path.read_bytes() == PDF_BYTES
        assert r.sha256 == sha256_file(r.path)

    def test_html_error_page_is_rejected_not_saved(self, tmp_path):
        """A 200 carrying an error page must never become a .pdf on disk."""
        with _client(lambda _: httpx.Response(200, content=HTML_BYTES)) as c:
            r = download_entry(_entry(), raw_dir=tmp_path, client=c, max_retries=1)
        assert r.status == "failed"
        assert "not a PDF" in (r.message or "")
        assert list(tmp_path.iterdir()) == []

    def test_truncated_stream_is_rejected(self, tmp_path):
        def handler(_):
            return httpx.Response(200, content=PDF_BYTES, headers={"content-length": "99999"})

        with _client(handler) as c:
            r = download_entry(_entry(), raw_dir=tmp_path, client=c, max_retries=1)
        assert r.status == "failed"
        assert list(tmp_path.iterdir()) == []

    def test_no_partial_file_survives_a_failure(self, tmp_path):
        with _client(lambda _: httpx.Response(500)) as c:
            r = download_entry(_entry(), raw_dir=tmp_path, client=c, max_retries=1)
        assert r.status == "failed"
        assert list(tmp_path.iterdir()) == []

    def test_hash_mismatch_is_reported_and_file_discarded(self, tmp_path):
        """The upstream document changed: fail loudly rather than index new bytes."""
        with _client(lambda _: httpx.Response(200, content=PDF_BYTES)) as c:
            r = download_entry(_entry(sha256="0" * 64), raw_dir=tmp_path, client=c, max_retries=1)
        assert r.status == "hash_mismatch" and not r.ok
        assert "corpus lock" in (r.message or "")
        assert list(tmp_path.iterdir()) == []

    def test_cached_file_is_not_refetched(self, tmp_path):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, content=PDF_BYTES)

        entry = _entry(sha256=__import__("hashlib").sha256(PDF_BYTES).hexdigest())
        (tmp_path / entry.filename).write_bytes(PDF_BYTES)
        with _client(handler) as c:
            r = download_entry(entry, raw_dir=tmp_path, client=c)
        assert r.status == "cached"
        assert calls == []

    def test_stale_cache_is_replaced(self, tmp_path):
        """A local file that no longer matches its pin must be re-fetched."""
        entry = _entry(sha256=__import__("hashlib").sha256(PDF_BYTES).hexdigest())
        (tmp_path / entry.filename).write_bytes(b"%PDF- stale content")
        with _client(lambda _: httpx.Response(200, content=PDF_BYTES)) as c:
            r = download_entry(entry, raw_dir=tmp_path, client=c)
        assert r.status == "downloaded"
        assert (tmp_path / entry.filename).read_bytes() == PDF_BYTES

    def test_retries_then_succeeds(self, tmp_path):
        attempts = {"n": 0}

        def handler(_):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return httpx.Response(503)
            return httpx.Response(200, content=PDF_BYTES)

        with _client(handler) as c:
            r = download_entry(_entry(), raw_dir=tmp_path, client=c, max_retries=3)
        assert r.status == "downloaded"
        assert attempts["n"] == 2
