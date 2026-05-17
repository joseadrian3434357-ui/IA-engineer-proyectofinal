from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from google import genai

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class MetricasDeUso:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latencia_ms: float
    costo_estimado_usd: float

class LLMClient:

    def __init__(
        self,
        provedor: str | None = None,
        modelo: str | None = None,
        temperatura: float | None = None,
    ) -> None:
        config = get_settings
        self.provedor = (provedor or config.proveedor_llm).lower()
        self.modelo = modelo or config.modelo_llm
        self.temperatura = (
            config.temperatura_llm if temperatura is None else float(temperatura)
        )
        self.input_cost_per_1m_tokens = config.input_cost_por_1m_tokens
        self.output_cost_per_1m_tokens = config.output_cost_por_1m_tokens

        if self.provedor == "gemini":
            self.modelo = os.environ.get("GEMINI_MODEL", config.modelo_llm)
            self.cliente = genai.Client()
        elif self.provedor == "groq":
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                raise ValueError("No se encuentra la GROQ_API_KEY en el .env")
            self.model = modelo or os.environ.get(
                "GROQ_MODEL", "llama-3.3-70b-versatile"
            )
            self.cliente = Groq(api_key=api_key)
        else:
            raise ValueError(
                f"Provedor sin soporte '{self.provedor}'. Soportados: 'gemini', 'groq'."
            )

    def chat(self, mensajes: list[dict[str, str]] | str, **kwargs: Any) -> dict[str, Any]:
        if self.provedor == "groq":
            return self._chat_groq(mensajes, **kwargs)

        prompt = self._mensajes_to_prompt(mensajes)
        if not prompt:
            raise ValueError("los mensajes no pueden estar vacios.")

        started = time.perf_counter()
        try:
            config = {"temperatura": self.temperatura}
            extra_config = kwargs.pop("config", None)
            if isinstance(extra_config, dict):
                config.update(extra_config)
            config.update(kwargs)
            response = self.client.modelos.generate_content(
                modelo=self.modelo,
                contents=prompt,
                config=config,
            )    
        except Exception as exc:
            logger.exception(
                "Llamada del LLM fallida | provedor=%s | modelo=%s",
                self.provedor,
                self.modelo,
            )
            raise RuntimeError("Failed to call the LLM provider.") from exc


        latencia_ms = (time.perf_counter() - started) * 1000
        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)
        costo_estimado_usd = self._costo_estimado(prompt_tokens, completion_tokens)

        metrics = MetricasDeUso(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latencia_ms=latencia_ms,
            costo_estimado_usd=costo_estimado_usd,
        )
        self.log_usage(metrics)

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("El LLM dio una respuesta vacia.")

        return {
            "response": text,
            "metadata": {
                "provedor": self.provedor,
                "modelo": self.modelo,
                "temperatura": self.temperatura,
                "uso": {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                "latency_ms": round(metrics.latencia_ms, 2),
                "costo_estimado_usd": round(metrics.costo_estimado_usd, 8),
            },
        }

    def _costo_estimado(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_1m_tokens
        output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_1m_tokens
        return input_cost + output_cost
    
    def _extract_usage(self, response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            usage = getattr(response, "usage_metadata", None)

        prompt_tokens = self._read_usage_value(
            usage,
            "prompt_tokens",
            "prompt_token_count",
            "input_tokens",
            "input_token_count",
        )
        completion_tokens = self._read_usage_value(
            usage,
            "completion_tokens",
            "candidates_token_count",
            "output_tokens",
            "output_token_count",
        )
        total_tokens = self._read_usage_value(
            usage,
            "total_tokens",
            "total_token_count",
        ) 

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        return prompt_tokens, completion_tokens, total_tokens
    
    def _read_usage_value(self, usage: Any, *fields: str) -> int:
        if usage is None:
            return 0

        for field_name in fields:
            value = getattr(usage, field_name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(field_name)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return 0

    def _chat_groq(self, mensajes: object, **kwargs: Any) -> dict[str, Any]:
        """Handle chat via the Groq API (OpenAI-compatible)."""
        if isinstance(mensajes, str):
            groq_mensajes = [{"role": "user", "content": mensajes}]
        elif isinstance(mensajes, list):
            groq_mensajes = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in mensajes
            ]
        else:
            raise TypeError("los menajes deben ser o una string o una lista de dicts.")

        config = kwargs.pop("config", None) or {}
        if isinstance(config, dict):
            config = dict(config)
        else:
            config = {}
        config.update(kwargs)

        max_tokens = config.pop("max_output_tokens", config.pop("max_tokens", 1024))
        temperatura = config.pop("temperatura", self.temperatura)

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_mensajes,
                max_tokens=max_tokens,
                temperatura=temperatura,
            )
        except Exception as exc:
            logger.exception(
                "LLM call failed | provider=%s | model=%s",
                self.provider,
                self.model,
            )
            raise RuntimeError("Failed to call the LLM provider.") from exc

        latencia_ms = (time.perf_counter() - started) * 1000

        text = (response.choices[0].message.content or "").strip()

        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        total_tokens = prompt_tokens + completion_tokens
        costo_estimado_usd = self._estimate_cost(prompt_tokens, completion_tokens)

        metrics = MetricasDeUso(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latencia_ms=latencia_ms,
            costo_estimado_usd=costo_estimado_usd,
        )
        self.log_usage(metrics)


        if not text:
            raise RuntimeError("El LLM dio una respuesta vacia.")

        return {
            "response": text,
            "metadata": {
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "usage": {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens,
                },
                "latency_ms": round(metrics.latency_ms, 2),
                "estimated_cost_usd": round(metrics.estimated_cost_usd, 8),
            },
        }

    def _messages_to_prompt(self, messages: object) -> str:
        if isinstance(messages, str):
            return messages.strip()

        if not isinstance(messages, list):
            raise TypeError("messages must be either a string or a list of dictionaries.")

        lines: list[str] = []
        for item in messages:
            if not isinstance(item, dict):
                raise TypeError("Each message must be a dictionary with role/content.")
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")

        return "\n".join(lines).strip()

    