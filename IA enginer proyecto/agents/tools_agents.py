import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

from rag.ingestion import Chunk, chunk_by_paragraphs, load_txt, load_document, load_pdf
from rag.retrieval import HybridRetriever
from rag.vectorstore import create_vectorstore, index_chunks

logger = logging.getLogger(__name__)


# ── Modelos Pydantic ──────────────────────────────────────────


class ToolCall(BaseModel):
    tool: str
    argument: str


class ToolResult(BaseModel):
    output: str
    success: bool
    source: str | None = None


# ── Estado interno ────────────────────────────────────────────

_last_search_context: str = ""
_retriever_cache: HybridRetriever | None = None
_retriever_profile_mtime: float | None = None

# ── ChromaDB setup ────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")


def _resolve_profile_doc(doc: str | Path | None) -> Path:
    """Ruta absoluta al PDF de LinkedIn de la persona representada."""
    candidate = PROJECT_ROOT / "data" / "linkedin_Ed_Donner.pdf"
    return candidate.resolve()


# Documento activo (export de LinkedIn de quien representa el agente)
_profile_doc_path: Path = _resolve_profile_doc(None)
_profile_source_name: str = _profile_doc_path.name


def _profile_collection_name() -> str:
    """Nombre de colección Chroma estable por archivo de perfil (RAG híbrido)."""
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", _profile_doc_path.stem)
    stem = stem.strip("_") or "profile"
    return f"agents_prof_{stem}_hybrid"[:63]


def _search_linkedin_text_fallback(query: str, max_snippets: int = 5) -> str | None:
    """Búsqueda por palabras clave en el TXT de perfil si el vector store no devuelve nada."""
    if not _profile_doc_path.is_file():
        return None
    try:
        text = load_document(str(_profile_doc_path)).content
    except Exception:
        text = _profile_doc_path.read_text(encoding="utf-8", errors="replace")
    q = query.strip().lower()
    if not q:
        return None
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text.strip()]
    terms = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+", q)
    if not terms:
        terms = [q]
    scored: list[tuple[int, str]] = []
    for p in paras:
        pl = p.lower()
        score = sum(1 for t in terms if t in pl)
        if score:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        for p in paras:
            if q in p.lower():
                scored.append((1, p))
    if not scored:
        return None
    lines_out: list[str] = []
    raw_parts: list[str] = []
    for i, (_, p) in enumerate(scored[:max_snippets], 1):
        snippet = (p[:800] + "…") if len(p) > 800 else p
        lines_out.append(f"[{i}] ({_profile_source_name}): {snippet}")
        raw_parts.append(p)
    global _last_search_context
    _last_search_context = "\n\n".join(raw_parts)
    return "\n".join(lines_out)


def _build_profile_chunks() -> list[Chunk]:
    """Chunks del perfil LinkedIn con IDs estables (alineados con Chroma)."""
    if not _profile_doc_path.is_file():
        return []
    doc = load_document(str(_profile_doc_path))
    doc.metadata = {
        **doc.metadata,
        "source": _profile_source_name,
        "profile_path": str(_profile_doc_path),
    }
    raw = chunk_by_paragraphs(doc, max_chunk_size=800, separator="\n\n")
    stem = _profile_doc_path.stem
    return [
        Chunk(
            content=c.content,
            metadata=dict(c.metadata),
            chunk_id=f"{stem}_{i}",
        )
        for i, c in enumerate(raw)
    ]


def _get_hybrid_retriever() -> HybridRetriever | None:
    """Colección Chroma + ``HybridRetriever`` (BM25 + vector) para el perfil activo."""
    global _retriever_cache, _retriever_profile_mtime

    if not _profile_doc_path.is_file():
        _retriever_cache = None
        _retriever_profile_mtime = None
        return None

    try:
        current_mtime = _profile_doc_path.stat().st_mtime
    except OSError:
        current_mtime = None

    if (
        _retriever_cache is not None
        and current_mtime is not None
        and _retriever_profile_mtime == current_mtime
    ):
        return _retriever_cache

    chunks = _build_profile_chunks()
    if not chunks:
        logger.warning(
            "Knowledge file missing or vacío: %s",
            _profile_doc_path,
        )
        _retriever_cache = None
        _retriever_profile_mtime = None
        return None

    collection_name = _profile_collection_name()
    collection = create_vectorstore(collection_name, CHROMA_DIR)

    try:
        n_existing = collection.count()
    except Exception:
        n_existing = 0

    if n_existing != len(chunks):
        logger.info(
            "Indexando %d chunks (RAG híbrido) en '%s'",
            len(chunks),
            collection_name,
        )
        index_chunks(collection, chunks)

    _retriever_cache = HybridRetriever(collection, chunks, alpha=0.5)
    _retriever_profile_mtime = current_mtime
    logger.info("HybridRetriever listo para '%s'", collection_name)
    return _retriever_cache


# ── Herramientas ──────────────────────────────────────────────

def search_linkedin(query: str) -> str:
    """Busca en el perfil vía RAG híbrido (BM25 + vector, ``HybridRetriever``)."""
    global _last_search_context

    if not (query or "").strip():
        return "Indica una consulta de búsqueda no vacía."

    try:
        retriever = _get_hybrid_retriever()
        if retriever is None:
            raise RuntimeError("Sin índice RAG (archivo de perfil ausente o vacío).")

        results = retriever.search(query.strip(), top_k=5)
        if results:
            formatted: list[str] = []
            context_parts: list[str] = []
            for i, r in enumerate(results, 1):
                src = r.metadata.get("source", _profile_source_name)
                formatted.append(
                    f"[{i}] ({src}) [score={r.score:.3f}]: {r.content}"
                )
                context_parts.append(r.content)
            _last_search_context = "\n".join(context_parts)
            return "\n".join(formatted)

    except Exception as e:
        logger.warning("search_docs RAG (HybridRetriever) falló: %s", e)

    fallback = _search_linkedin_text_fallback(query)
    if fallback:
        return fallback

    return (
        f"No se encontraron fragmentos relevantes para: '{query}' en "
        f"{_profile_source_name}. Intenta reformular o usa lookup con un término concreto."
    )


def lookup(term: str) -> str:
    """Busca un término en el documento de perfil (mismo TXT que usa ``search_docs`` / RAG)."""
    if not (term or "").strip():
        return "Indica un término no vacío."
    if not _profile_doc_path.is_file():
        return (
            f"No existe el archivo de perfil {_profile_doc_path}. "
            "Configura REPRESENTED_LINKEDIN_DOC o linkedin_doc del agente."
        )

    try:
        text = load_document(str(_profile_doc_path)).content
    except Exception:
        text = _profile_doc_path.read_text(encoding="utf-8", errors="replace")
        
    needle = term.strip().lower()

    def _clip(s: str, max_len: int = 450) -> str:
        s = s.strip()
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    sentences = re.split(r"[.!?\n]+", text)
    matches = [s for s in sentences if needle in s.lower() and s.strip()]
    if matches:
        return " | ".join(_clip(m) for m in matches[:5])

    lines = [ln for ln in text.splitlines() if needle in ln.lower() and ln.strip()]
    if lines:
        return " | ".join(_clip(ln) for ln in lines[:5])

    return f"Término '{term}' no encontrado en {_profile_source_name}."


# ── Registro de herramientas ──────────────────────────────────

TOOLS_REGISTRY = {
    "search_linkedin": {
        "description": (
            "Busca unicamente en el perfil de linkedin_Ed_Donner.pdf vía RAG híbrido usando HybridRetrieval (BM25 + vector)."
        ),
        "function": search_linkedin,
    },
    "lookup": {
        "description": (
            "Busca un término específico dentro del último documento recuperado. "
            "Argumento: término a buscar."
        ),
        "function": lookup,
    },
    "Finish": {
        "description": (
            "Termina la ejecución con la respuesta final. "
            "Argumento: respuesta completa."
        ),
        "function": None,
    },
}


def _profile_doc_display() -> str:
    try:
        return str(_profile_doc_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(_profile_doc_path)


def _sync_tool_descriptions() -> None:
    rel = _profile_doc_display()
    TOOLS_REGISTRY["search_linkedin"]["description"] = (
        "Busca ÚNICAMENTE en el export de LinkedIn data/linkedin_Ed_Donner.pdf usando RAG híbrido "
        f"(BM25 + vector; {rel}). Argumento: consulta en lenguaje natural."
    )
    TOOLS_REGISTRY["lookup"]["description"] = (
        "Busca un término o frase corta en el mismo documento de perfil que usa search_linkedin "
        f"({rel}). No depende de búsquedas previas."
    )


def set_represented_profile(doc: str | Path | None = None) -> Path:
    """Define el TXT de LinkedIn a indexar y consultar. ``doc`` relativo al proyecto o absoluto."""
    global _profile_doc_path, _profile_source_name, _retriever_cache, _retriever_profile_mtime, _last_search_context
    new_path = _resolve_profile_doc(doc)
    if _profile_doc_path.resolve() != new_path.resolve():
        _retriever_cache = None
        _retriever_profile_mtime = None
        _last_search_context = ""
    _profile_doc_path = new_path
    _profile_source_name = new_path.name
    _sync_tool_descriptions()
    return new_path


_sync_tool_descriptions()


# ── Parsing y ejecución ──────────────────────────────────────


def parse_action(text: str) -> ToolCall:
    """Parsea texto de acción a un ToolCall."""
    pattern = r'(\w+)\s*[\[\(]\s*["\']?(.*?)["\']?\s*[\]\)]'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return ToolCall(tool=match.group(1), argument=match.group(2))

    return ToolCall(tool="error", argument=f"No se pudo parsear la acción: {text}")


def execute_tool(action: ToolCall) -> ToolResult:
    """Ejecuta una herramienta y retorna el resultado."""
    if action.tool == "error":
        return ToolResult(output=action.argument, success=False, source="parser")

    if action.tool not in TOOLS_REGISTRY:
        available = list(TOOLS_REGISTRY.keys())
        return ToolResult(
            output=f"Herramienta '{action.tool}' no encontrada. Disponibles: {available}",
            success=False,
            source=action.tool,
        )

    func = TOOLS_REGISTRY[action.tool]["function"]

    if func is None:
        return ToolResult(output=action.argument, success=True, source=action.tool)

    try:
        result = func(action.argument)
        return ToolResult(output=result, success=True, source=action.tool)
    except Exception as e:
        logger.error("Error executing tool '%s': %s", action.tool, e)
        return ToolResult(
            output=f"Error ejecutando {action.tool}: {e}",
            success=False,
            source=action.tool,
        )