"""Corpus acquisition: the pinned manifest and its verifying downloader."""

from mmrag.corpus.download import (
    DownloadResult,
    download_corpus,
    download_entry,
    lock_manifest,
    sha256_file,
)
from mmrag.corpus.manifest import CorpusEntry, CorpusLock, CorpusLockEntry, CorpusManifest

__all__ = [
    "CorpusEntry",
    "CorpusLock",
    "CorpusLockEntry",
    "CorpusManifest",
    "DownloadResult",
    "download_corpus",
    "download_entry",
    "lock_manifest",
    "sha256_file",
]
