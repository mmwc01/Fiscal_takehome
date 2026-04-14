from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


@dataclass
class LLMResponse:
    content: dict[str, Any]
    raw_text: str


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None

    def enabled(self) -> bool:
        return self._client is not None

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not self._client:
            raise RuntimeError("OPENAI_API_KEY is not set")

        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = (response.choices[0].message.content or "").strip()

        try:
            content = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM returned invalid JSON: {raw_text[:500]}") from e

        return LLMResponse(content=content, raw_text=raw_text)