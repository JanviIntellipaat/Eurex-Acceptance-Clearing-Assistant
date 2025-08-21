
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from .llm import LLMRouter, ChatMessage
from .vector_hnsw import Chunk
from .prompts import GHERKIN_SYSTEM, GHERKIN_USER_TEMPLATE

@dataclass
class GherkinMeta:
    feature: str
    scenarios: int
    constraints: List[str]

class GherkinGenerator:
    def __init__(self, router: LLMRouter):
        self.router = router

    def generate(self, feature_name: str, component: Optional[str], constraints: List[str], retrieved: List[Chunk], scenario_count: int = 5):
        ctx = ""
        if retrieved:
            ctx_lines = [f"[{i+1}] {c.source} p{c.page or '-'} — {c.text[:300]}..." for i,c in enumerate(retrieved)]
            ctx = "\n".join(ctx_lines)

        prompt = GHERKIN_USER_TEMPLATE.format(
            feature_name=feature_name,
            component=(f" in component {component}" if component else ""),
            scenarios=scenario_count,
            constraints=(", ".join(constraints) if constraints else "N/A"),
            context=(ctx if ctx else "(no context provided)")
        )
        result = self.router.complete(prompt, system=GHERKIN_SYSTEM)
        meta = GherkinMeta(feature=feature_name, scenarios=scenario_count, constraints=constraints)
        return result.strip(), meta.__dict__
