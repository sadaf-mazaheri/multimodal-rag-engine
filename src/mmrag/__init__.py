"""Multimodal RAG.

One engine (``engine.RAGEngine``) routes, fuses, reranks and answers over
pluggable retrievers opened by index components (``indexing``). The benchmark
methods are configurations of it, over one shared ingestion layer:

* ``methods.method1_textified``  -- everything flattened to text, hybrid BM25 + dense (frozen).
* ``methods.method2_modality``   -- per-modality retrievers behind a query router.
* ``methods.method3_visual``     -- Method 2's retrievers plus visual page retrieval.
"""

__version__ = "0.1.0"
