from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from pipeline.llm_events.decision_schema import Usage


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PRICING_PATH = os.path.join(ROOT_DIR, "config", "llm_pricing.json")


@dataclass(frozen=True)
class TokenEstimate:
    low: int
    high: int


class PricingCatalog:
    def __init__(self, path: str = DEFAULT_PRICING_PATH):
        with open(path, "r", encoding="utf-8") as handle:
            self.data: dict[str, Any] = json.load(handle)

    @property
    def version(self) -> str:
        return str(self.data["version"])

    def rates(self, model: str, mode: str = "standard") -> dict[str, float]:
        models = self.data.get("models", {})
        resolved = model
        if resolved not in models:
            aliases = sorted((name for name in models if model.startswith(name)), key=len, reverse=True)
            if not aliases:
                raise KeyError(f"No pricing configured for model {model}")
            resolved = aliases[0]
        return {key: float(value) for key, value in models[resolved][mode].items()}

    def cost(self, model: str, usage: Usage, mode: str = "standard") -> float:
        rates = self.rates(model, mode)
        return (
            usage.uncached_input_tokens * rates["input"]
            + usage.cached_input_tokens * rates["cached_input"]
            + usage.output_tokens * rates["output"]
        ) / 1_000_000


def estimate_text_tokens(text: str) -> TokenEstimate:
    chars = len(text)
    return TokenEstimate(low=max(1, round(chars / 4.0)), high=max(1, round(chars / 3.5)))


def estimate_cost_range(
    catalog: PricingCatalog,
    model: str,
    input_tokens: TokenEstimate,
    output_tokens: TokenEstimate,
    mode: str = "standard",
) -> tuple[float, float]:
    low = catalog.cost(model, Usage(input_tokens=input_tokens.low, output_tokens=output_tokens.low), mode)
    high = catalog.cost(model, Usage(input_tokens=input_tokens.high, output_tokens=output_tokens.high), mode)
    return low, high
