"""Multimodal RAG benchmark.

Three retrieval architectures are implemented over one shared ingestion layer so
their results are directly comparable:

* ``methods.method1_textified``  -- everything flattened to text, hybrid BM25 + dense.
* ``methods.method2_modality``   -- per-modality retrievers behind a query router.
* ``methods.method3_visual``     -- adds late-interaction visual page retrieval.
"""

__version__ = "0.1.0"
