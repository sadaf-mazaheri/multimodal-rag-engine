"""One place that maps ``config.method`` to the class that implements it."""

from __future__ import annotations

from typing import Any

from mmrag.config import ExperimentConfig


def _classes() -> dict[str, Any]:
    # Resolved at call time, so a test that swaps a class on ``mmrag.methods``
    # is honoured here too.
    import mmrag.methods as methods

    return {
        "method1": methods.Method1Textified,
        "method2": methods.Method2ModalityAware,
        "method3": methods.Method3HybridVisual,
    }


METHODS = ("method1", "method2", "method3")


def build_method(config: ExperimentConfig, **kwargs: Any) -> Any:
    """The method a config selects, constructed over that config."""
    builder = _classes().get(config.method)
    if builder is None:
        raise ValueError(f"config selects {config.method!r}, which is not implemented")
    return builder(config, **kwargs)
