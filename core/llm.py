
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

    def _get_openai(self):
        if self._openai is None:
            from openai import OpenAI
            base_url = self.settings.base_url
            if base_url:
                self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
            else:
                self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai

    def _get_ollama(self):
        if self._ollama is None:
            import ollama, os
            if self.settings.base_url:
                os.environ["OLLAMA_HOST"] = self.settings.base_url
            self._ollama = ollama
        return self._ollama

    def stream_chat(self, messages: List[ChatMessage]) -> Iterable[str]:
        if self.provider == "openai":
            client = self._get_openai()
            stream = client.chat.completions.create(
                model=self.settings.model,
                temperature=self.settings.temperature,
                messages=[m.__dict__ for m in messages],
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            client = self._get_ollama()
            stream = client.chat(
                model=self.settings.model,
                messages=[m.__dict__ for m in messages],
                options={"temperature": self.settings.temperature},
                stream=True
            )
            for part in stream:
                token = part.get("message", {}).get("content", "")
                if token:
                    yield token

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        return "".join(self.stream_chat(messages))
