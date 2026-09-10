"""Query router: deciding which modality retrievers to fire.

**Ownership: unique to Method 2.** Method 1 has no router -- it flattens
everything into one index, so there is nothing to route between.

The router is what makes Method 2 more than "Method 1 with extra indexes". It
reads the query for evidence about *what kind of thing* would answer it, and
fires only the retrievers that could plausibly hold it.

Two design commitments shape it:

**Text is never switched off.** It is the safety net. A router that confidently
routes "what were the 2023 revenues?" to tables alone, on a corpus where the
answer happens to be in a sentence, loses that answer outright -- and a false
negative here is unrecoverable, while a false positive only costs latency. So
text always fires, and the router's real job is deciding what to fire *besides*
text.

**The decision is recorded, not just made.** ``RoutingDecision`` carries the
matched signals per modality, so Step 6 can ask the question that actually
matters -- did routing help, or did it just add retrievers? -- instead of
treating the router as a black box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from mmrag.config import RouterConfig
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import MetadataFilter
from mmrag.schemas import Modality

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lexical signals
# ---------------------------------------------------------------------------

# Words that suggest the answer lives in a table. Weighted: "table" is near
# conclusive, "how many" only mildly suggestive since prose states counts too.
TABLE_SIGNALS: dict[str, float] = {
    r"\btables?\b": 1.0,
    r"\brows?\b": 0.6,
    r"\bcolumns?\b": 0.6,
    r"\bcells?\b": 0.5,
    r"\bhow (?:many|much)\b": 0.4,
    r"\btotals?\b": 0.4,
    r"\bcompared? (?:to|with)\b": 0.3,
    r"\bbreakdown\b": 0.5,
    r"\bper (?:year|quarter|segment|region|category)\b": 0.4,
    r"\blist (?:of|all)\b": 0.3,
    # A currency amount or a percentage is weak evidence on its own, but it is
    # the shape of question tables answer.
    r"[$€£]\s?\d": 0.3,
    r"\b\d+(?:\.\d+)?\s?%": 0.3,
}

# Words that suggest the answer is visual.
IMAGE_SIGNALS: dict[str, float] = {
    r"\bfigures?\b": 1.0,
    r"\bcharts?\b": 1.0,
    r"\bdiagrams?\b": 1.0,
    r"\bplots?\b": 0.8,
    r"\bgraphs?\b": 0.8,
    r"\bimages?\b": 0.8,
    r"\bphotos?\b": 0.8,
    r"\bschematic\b": 0.9,
    r"\billustrat\w*": 0.7,
    r"\bdepicts?\b": 0.7,
    r"\bshows?\b": 0.35,
    r"\bvisuali[sz]\w*": 0.7,
    r"\blooks? like\b": 0.6,
    r"\baxis|axes\b": 0.6,
    r"\blegend\b": 0.5,
    r"\bcurve\b": 0.5,
    r"\btrend\b": 0.35,
    r"\barchitecture\b": 0.4,
}

# An explicit page reference: "on page 12", "p. 7".
PAGE_RE = re.compile(r"\b(?:on\s+)?(?:page|pp?\.)\s*(\d{1,4})\b", re.IGNORECASE)

# An explicit figure/table number: "Figure 3", "Table 2".
FIGURE_NUMBER_RE = re.compile(r"\b(figure|fig\.?|table)\s*(\d{1,3})\b", re.IGNORECASE)


@dataclass
class RoutingDecision:
    """Which retrievers to fire, and the evidence behind that choice."""

    modalities: list[Modality]
    scores: dict[str, float] = field(default_factory=dict)
    signals: dict[str, list[str]] = field(default_factory=dict)
    filters: MetadataFilter = field(default_factory=MetadataFilter)
    strategy: str = "heuristic"
    fell_back: bool = False

    @property
    def confidence(self) -> float:
        """How strongly the strongest non-text modality was signalled."""
        non_text = [v for k, v in self.scores.items() if k != Modality.TEXT.value]
        return max(non_text) if non_text else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "modalities": [m.value for m in self.modalities],
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "signals": self.signals,
            "filters": self.filters.as_dict(),
            "strategy": self.strategy,
            "fell_back": self.fell_back,
            "confidence": round(self.confidence, 3),
        }


class HeuristicRouter:
    """Lexical query router.

    Deliberately rule-based rather than model-based. Three reasons: it is
    deterministic, so two runs of one config give identical routing; it costs
    nothing, so routing is not confounded with a generation provider; and its
    decisions are inspectable, so a bad route can be traced to the exact phrase
    that caused it. An LLM router is a later ablation, not the baseline.
    """

    def __init__(self, config: RouterConfig, *, available: list[Modality] | None = None):
        self.config = config
        # Visual page retrieval belongs to Method 3; Method 2 offers text,
        # tables and images.
        self.available = available or [Modality.TEXT, Modality.TABLE, Modality.IMAGE]

    def route(self, query: str, *, base_filters: MetadataFilter | None = None) -> RoutingDecision:
        scores, signals = self._score(query)
        filters = self._extract_filters(query)
        if base_filters is not None:
            # Caller-supplied constraints win: the router may narrow, never widen.
            filters = filters.merge(base_filters)

        if self.config.strategy == "all":
            return RoutingDecision(
                modalities=list(self.available),
                scores=scores,
                signals=signals,
                filters=filters,
                strategy="all",
            )

        chosen: list[Modality] = []
        # Text always fires: it is the safety net, and a missed answer cannot be
        # recovered later while an extra retriever only costs latency.
        if Modality.TEXT in self.available:
            chosen.append(Modality.TEXT)

        for modality in self.available:
            if modality is Modality.TEXT:
                continue
            if scores.get(modality.value, 0.0) >= self.config.min_confidence:
                chosen.append(modality)

        fell_back = False
        if len(chosen) <= 1 and self.config.fallback_to_all and not self._is_clearly_textual(query):
            # Nothing beyond text was signalled and the query gives no positive
            # reason to believe it is purely textual: query everything rather
            # than guess wrong.
            chosen = list(self.available)
            fell_back = True

        return RoutingDecision(
            modalities=chosen,
            scores=scores,
            signals=signals,
            filters=filters,
            strategy=self.config.strategy,
            fell_back=fell_back,
        )

    # -- scoring -------------------------------------------------------------

    def _score(self, query: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores: dict[str, float] = {Modality.TEXT.value: 1.0}
        signals: dict[str, list[str]] = {}

        for modality, table in (
            (Modality.TABLE, TABLE_SIGNALS),
            (Modality.IMAGE, IMAGE_SIGNALS),
        ):
            total = 0.0
            matched: list[str] = []
            for pattern, weight in table.items():
                found = re.search(pattern, query, re.IGNORECASE)
                if found:
                    total += weight
                    matched.append(found.group(0).strip())
            if matched:
                signals[modality.value] = matched
            # Saturating rather than linear: five weak hints should not
            # outweigh the single word "table".
            scores[modality.value] = min(1.0, total)

        # "Figure 3" / "Table 2" is an explicit pointer, not a hint.
        explicit = FIGURE_NUMBER_RE.search(query)
        if explicit:
            kind = explicit.group(1).lower()
            target = Modality.TABLE if kind.startswith("table") else Modality.IMAGE
            scores[target.value] = 1.0
            signals.setdefault(target.value, []).append(explicit.group(0))

        return scores, signals

    @staticmethod
    def _is_clearly_textual(query: str) -> bool:
        """Whether the query positively asks for prose.

        Only these phrasings suppress the fallback. Everything else fans out,
        because the cost of a missed modality is an unanswerable question and
        the cost of an extra one is a few hundred milliseconds.
        """
        patterns = (
            r"\b(?:summari[sz]e|explain|describe|what does .* mean|why|definition of)\b",
            r"\bwho (?:is|are|was|were)\b",
            r"\baccording to the (?:text|report|paper|document)\b",
        )
        return any(re.search(p, query, re.IGNORECASE) for p in patterns)

    # -- filters -------------------------------------------------------------

    @staticmethod
    def _extract_filters(query: str) -> MetadataFilter:
        """Pull structural constraints out of the query text.

        Only unambiguous ones. A page number written as "page 12" is a genuine
        constraint; a bare "12" is not, and treating it as one would silently
        discard every other page.
        """
        pages = [int(m) for m in PAGE_RE.findall(query)]
        return MetadataFilter(page_numbers=pages or None)
