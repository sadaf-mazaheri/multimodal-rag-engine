"""Configured token prices and per-query cost.

Prices live in ``configs/pricing.yaml`` and are the project's *configured
evaluation rates*, dated with ``as_of``; they are not a claim about any vendor's
current or permanent pricing. Nothing here hard-codes a price.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_PRICING = "configs/pricing.yaml"


class PricingError(ValueError):
    """No configured price for a model, or a malformed pricing file."""


class ModelPrice(BaseModel):
    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)
    aliases: list[str] = Field(default_factory=list)
    gateway: str | None = None


class PricingTable(BaseModel):
    version: int = 1
    kind: str = "configured_evaluation_rates"
    currency: str = "USD"
    unit: str = "per_1m_tokens"
    as_of: str
    source: str
    note: str | None = None
    models: dict[str, ModelPrice]

    @classmethod
    def load(cls, path: str | Path) -> PricingTable:
        p = Path(path)
        if not p.exists():
            raise PricingError(f"no pricing file at {p}")
        try:
            return cls.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
        except Exception as exc:
            raise PricingError(f"invalid pricing file {p}: {exc}") from exc

    def price_for(self, model: str) -> tuple[str, ModelPrice]:
        """The configured price for a model name or one of its aliases."""
        for name, price in self.models.items():
            if model == name or model in price.aliases:
                return name, price
        raise PricingError(
            f"no configured price for model {model!r}; add it (or an alias) to the pricing file"
        )

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        _, price = self.price_for(model)
        return (prompt_tokens * price.input_per_1m
                + completion_tokens * price.output_per_1m) / 1_000_000

    def describe(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
