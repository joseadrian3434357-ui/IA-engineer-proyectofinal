import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag.ingestion import Chunk
from rag.vectorstore import SearchResult, search as vector_search

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración LLM (Groq vía OpenAI-compatible API)
# ---------------------------------------------------------------------------

GROQ_MODEL = os.environ.get("GROQ_MODEL")


def _get_groq_client() -> OpenAI:

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY no está configurada, ponla en el .env. "
        )
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


_groq_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Singleton del cliente Groq."""
    global _groq_client
    if _groq_client is None:
        _groq_client = _get_groq_client()
    return _groq_client


# ---------------------------------------------------------------------------
# Tracker de uso de tokens (para métricas en compare_rag.py)
# ---------------------------------------------------------------------------

_usage_tracker = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def reset_usage_tracker():
    """Reinicia el acumulador de tokens."""
    global _usage_tracker
    _usage_tracker = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def get_usage():
    """Retorna copia del acumulador actual de tokens."""
    return dict(_usage_tracker)


# ---------------------------------------------------------------------------
# Helper: LLM call wrapper
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """Llama al LLM (GPT-OSS vía Groq) y retorna el texto de respuesta.

    Acumula métricas de uso en el tracker interno.
    """
    global _usage_tracker
    client = _get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )

    # Acumular métricas
    if response.usage:
        _usage_tracker["input_tokens"] += response.usage.prompt_tokens or 0
        _usage_tracker["output_tokens"] += response.usage.completion_tokens or 0
    _usage_tracker["calls"] += 1

    return response.choices[0].message.content



# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combina BM25 (keyword) y vector search (semántico) con score normalizado."""

    def __init__(
        self, collection, chunks: list[Chunk], alpha: float = 0.5
    ) -> None:
        """
        Args:
            collection: Colección de ChromaDB.
            chunks: Lista de Chunks indexados.
            alpha: Peso del vector search (1-alpha = peso BM25).
        """
        self.collection = collection
        self.chunks = chunks
        self.alpha = alpha
        self.bm25 = BM25Index(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Búsqueda híbrida con scores normalizados y combinados."""
        candidates = top_k * 2

        # --- BM25 ---
        bm25_results = self.bm25.search(query, top_k=candidates)
        bm25_scores: dict[str, float] = {}
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results)
            if max_bm25 > 0:
                for chunk, score in bm25_results:
                    bm25_scores[chunk.chunk_id] = score / max_bm25

        # --- Vector search ---
        vec_results = vector_search(self.collection, query, n_results=candidates)
        vec_scores: dict[str, float] = {}
        vec_content: dict[str, SearchResult] = {}
        if vec_results:
            max_vec = max(r.score for r in vec_results)
            if max_vec > 0:
                for r in vec_results:
                    vec_scores[r.chunk_id] = r.score / max_vec
                    vec_content[r.chunk_id] = r

        # --- Combinar scores ---
        all_ids = set(bm25_scores.keys()) | set(vec_scores.keys())
        combined: list[tuple[str, float]] = []
        for cid in all_ids:
            vec_norm = vec_scores.get(cid, 0.0)
            bm25_norm = bm25_scores.get(cid, 0.0)
            score = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)

        # --- Construir resultados ---
        results = []
        for cid, score in combined[:top_k]:
            if cid in vec_content:
                r = vec_content[cid]
                results.append(SearchResult(
                    content=r.content, metadata=r.metadata,
                    score=score, chunk_id=cid,
                ))
            elif cid in self._chunk_map:
                chunk = self._chunk_map[cid]
                results.append(SearchResult(
                    content=chunk.content, metadata=chunk.metadata,
                    score=score, chunk_id=cid,
                ))

        return results


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60
) -> list[tuple[str, float]]:
    """Fusiona múltiples listas de resultados usando RRF.

    Returns:
        Lista de (chunk_id, rrf_score) ordenada descendente.
    """
    rrf_scores: dict[str, float] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            rrf_scores[r.chunk_id] = (
                rrf_scores.get(r.chunk_id, 0.0) + 1 / (k + rank + 1)
            )
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

