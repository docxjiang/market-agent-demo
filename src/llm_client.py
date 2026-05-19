from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 60
    temperature: float = 0.2


class LLMClient:
    """Minimal OpenAI-compatible client placeholder for later stages."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        load_dotenv()
        self.config = config or LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("OPENAI_API_KEY or DEEPSEEK_API_KEY is not configured.")

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
