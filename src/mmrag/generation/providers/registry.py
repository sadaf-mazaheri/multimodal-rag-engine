"""Provider selection.

Which provider is reachable is a property of the machine, so it comes from
:class:`Settings` (the environment) rather than from the experiment config.
Which *model* to call is a property of the experiment, so that comes from the
YAML. Keeping the two apart is what lets the same committed config run against
OpenAI on one machine and a local server on another.
"""

from __future__ import annotations

from mmrag.config import Settings, get_settings
from mmrag.generation.providers.base import LLMProvider, ProviderError
from mmrag.generation.providers.echo import EchoProvider
from mmrag.logging_utils import get_logger

log = get_logger(__name__)


def get_provider(
    settings: Settings | None = None,
    *,
    name: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> LLMProvider:
    """Build the configured provider.

    ``name`` overrides the environment, which the CLI uses for ``--provider echo``
    so a pipeline can be exercised without spending anything. ``timeout`` and
    ``max_retries`` let a caller apply ``GenerationConfig``'s values; left unset,
    the provider's own defaults apply as before.
    """
    settings = settings or get_settings()
    provider = name or settings.generation_provider

    if provider == "echo":
        return EchoProvider()

    from mmrag.generation.providers.openai_provider import OpenAIProvider

    transport: dict[str, float | int] = {}
    if timeout is not None:
        transport["timeout"] = timeout
    if max_retries is not None:
        transport["max_retries"] = max_retries

    if provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if not key.strip() or key.startswith("sk-replace"):
            raise ProviderError(
                "OPENAI_API_KEY is not set in .env. Set it, or run with "
                "--provider echo to exercise the pipeline without a model."
            )
        return OpenAIProvider(base_url=settings.openai_base_url, api_key=key, **transport)

    if provider == "local":
        return OpenAIProvider(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key.get_secret_value(),
            **transport,
        )

    raise ProviderError(f"unknown generation provider: {provider!r}")
