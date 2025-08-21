from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional
import os

@dataclass
class ChatMessage:
    role: str
    content: str

class LLMRouter:
    def __init__(self, settings):
        self.settings = settings
        self.provider = settings.provider.lower()
        self._openai = None
        self._ollama = None

    # --- Providers ---
    def _ensure_openai(self):
        if self._openai is None:
            from openai import OpenAI
            base_url = self.settings.base_url or os.environ.get("OPENAI_BASE_URL")
            api_key = self.settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            self._openai = OpenAI(base_url=base_url, api_key=api_key)

    def _ensure_ollama(self):
        if self._ollama is None:
            import ollama
            if self.settings.base_url:
                os.environ["OLLAMA_BASE_URL"] = self.settings.base_url
            self._ollama = ollama

    # --- Streaming ---
    def stream_chat(self, messages: List[ChatMessage]) -> Iterable[str]:
        if self.provider == "openai":
            self._ensure_openai()
            stream = self._openai.chat.completions.create(
                model=self.settings.model,
                temperature=self.settings.temperature,
                max_completion_tokens=self.settings.max_output_tokens,
                messages=[m.__dict__ for m in messages],
                stream=True,
            )
            for part in stream:
                delta = part.choices[0].delta.content or ""
                if delta:
                    yield delta
        else:
            self._ensure_ollama()
            resp = self._ollama.chat(
                model=self.settings.model,
                messages=[m.__dict__ for m in messages],
                options={"temperature": self.settings.temperature},
                stream=True
            )
            for chunk in resp:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=prompt))
        return "".join(self.stream_chat(msgs))
