"""Configuración base del entorno"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    proveedor_llm: str  
    modelo_llm: str  
    temperatura_llm: str  
    nivel_log : str  
    gemini_api_key: str  
    output_cost_por_1m_tokens: float
    input_cost_por_1m_tokens: float

def leer_str(nombre: str, default: str) -> str:
    return os.getenv(nombre, default).strip()

def leer_float(nombre: str, default: float) -> float:  
    valor_bruto = leer_str(nombre, default)
    try:
        return float(valor_bruto)
    except ValueError as excepcion:
        raise ValueError(
            f"El valor de {nombre!r}: {valor_bruto!r}. Debe ser un float"
        ) from excepcion

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    proveedor_llm = leer_str("LLM_PROVIDER", "gemini").lower()
    modelo_llm = leer_str(
        "LLM_MODEL",
        leer_str("GEMINI_MODEL", "gemini-3-flash-preview"),
    )
    temperatura_llm = leer_float("LLM_TEMPERATURE", "0.3")
    nivel_log = leer_str("LOG_LEVEL", "INFO").upper()
    gemini_api_key = leer_str("GEMINI_API_KEY", "")
    output_cost_por_1m_tokens = leer_float("OUTPUT_COST_POR_1M_TOKENS", "0.0")
    input_cost_por_1m_tokens = leer_float("INPUT_COST_PER_1M_TOKENS", "0.0")

    if proveedor_llm == "gemini" and not gemini_api_key:
        raise ValueError("No se encuntra la GEMINI_API_KEY, ponla en tu documento .env")
   
   
    return Settings(
        proveedor_llm=proveedor_llm,
        modelo_llm=modelo_llm,
        temperatura_llm=temperatura_llm,
        nivel_log=nivel_log,
        gemini_api_key=gemini_api_key,
        output_cost_por_1m_tokens=output_cost_por_1m_tokens,
        input_cost_por_1m_tokens=input_cost_por_1m_tokens,
    )