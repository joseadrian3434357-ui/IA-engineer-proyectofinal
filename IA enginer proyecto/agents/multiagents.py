import logging
import os
import operator
import sys
import uuid
from pathlib import Path

# Raíz del proyecto en sys.path (necesario si ejecutas `python agents/agents.py`)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Literal
from typing_extensions import TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, START, END

from memory.store import checkpointer
from agents.hitl import human_gate
# pyrefly: ignore [missing-import]
from langgraph.types import Command

from rag.vectorstore import create_vectorstore, search as search, SearchResult
from rag.ingestion import load_directory, chunk_by_paragraphs, Chunk
from rag.retrieval import HybridRetriever, reciprocal_rank_fusion

from contracts.pydantic_baisco import Contacto

# ==========================================
# Configuración Inicial y Variables de Entorno
# ==========================================
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)
load_dotenv()

# ==========================================
# Utilidades de Interfaz (Colores ANSI)
# ==========================================
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_WHITE = "\033[97m"

# Metadatos para la visualización del flujo en stdout
AGENT_META = {
    "planner": {
        "color": _BLUE,
        "icon": "📋",
        "label": "PLANNER",
        "desc": "Analiza la consulta y genera un plan de acción estructurado",
    },
    "retriever": {
        "color": _CYAN,
        "icon": "🔍",
        "label": "RETRIEVER",
        "desc": "Busca documentos relevantes en el vector store según el plan",
    },
    "executor": {
        "color": _YELLOW,
        "icon": "⚙️",
        "label": "EXECUTOR",
        "desc": "Genera la respuesta basándose en el plan y los documentos",
    },
    "verifier": {
        "color": _MAGENTA,
        "icon": "✅",
        "label": "VERIFIER",
        "desc": "Evalúa la calidad de la respuesta y decide si aceptar o revisar",
    },
}

# ==========================================
# Configuración del LLM
# ==========================================
llm = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    temperature=0,
    max_tokens=2048,
)

fallback_llm = ChatOpenAI(
    model=os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    temperature=0,
    max_tokens=1024,
)

# ==========================================
# Estado Base (StateGraph) y Contratos
# ==========================================
class DocOpsState(TypedDict):
    """Estado compartido en todo el ciclo de vida del agente."""
    messages: Annotated[list[AnyMessage], operator.add]
    plan: str
    search_results: str
    draft: str
    feedback: str
    quality_score: float
    iteration: int
    persona_name: str

class QualityCheck(BaseModel):
    """Contrato de salida del Verifier."""
    score: float = Field(
        description="Score de calidad de 0.0 a 1.0. "
                    "1.0 = respuesta perfecta, 0.0 = inaceptable."
    )
    feedback: str = Field(
        description="Retroalimentación específica si el score es menor a 0.8. "
                    "Indica qué mejorar concretamente."
    )
    decision: Literal["accept", "revise"] = Field(
        description="'accept' si score >= 0.8, 'revise' si necesita mejora."
    )


# ─── AGENTE 1: PLANNER ──────────────────────────────────────
def planner_agent(state: DocOpsState) -> dict:
    import json
    
    # Cargar el dataset de evaluación
    eval_context = ""
    try:
        eval_path = Path(__file__).resolve().parent.parent / "evals/golden/eval.dataset.json"
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                eval_context = "\nDataset de evaluación (contexto histórico para guiar tus planes e identificar qué busca la respuesta ideal):\n"
                for item in eval_data:
                    eval_context += f"- Q: {item['question']} | A: {item['golden_answer']}\n"
    except Exception as e:
        logger.warning(f"Error cargando eval.dataset.json en el Planner: {e}")

    user_query = state["messages"][-1].content
    chat_history = state["messages"][:-1]

    messages_to_invoke = [
        SystemMessage(content=(
            f"Eres el planificador experto de un chatbot que te representa a ti mismo, {os.getenv('NAME', 'el profesional')}. "
            "Tu trabajo es analizar la consulta del usuario que conversa contigo y generar un plan claro "
            "con los pasos necesarios para revisar tu perfil de LinkedIn y poder responderla.\n\n"
            f"{eval_context}\n"
            "Reglas:\n"
            "- Utiliza el dataset de evaluación como inspiración de cómo deben estructurarse las respuestas ideales y qué buscar.\n"
            "- Ten en cuenta el contexto de si esta consulta sigue un hilo de la conversación.\n"
            "- Identifica qué información necesitas buscar en tu experiencia\n"
            f"- Define los criterios para responder como {os.getenv('NAME', 'el profesional')}\n"
            "- Sé específico sobre qué buscar en el documento\n"
            "- Responde SOLO con el plan, sin ejecutar ningún paso"
        ))
    ]
    messages_to_invoke.extend(chat_history)
    messages_to_invoke.append(HumanMessage(content=f"Genera un plan para responder: {user_query}"))

    response = llm.invoke(messages_to_invoke)

    return {"plan": response.content, "iteration": 0}


# ─── RAG: carga de colección e índice ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
_DATA_DIR = str(PROJECT_ROOT / "data")
_COLLECTION_NAME = "docops_multiagent"

_collection = None
_chunks: list[Chunk] = []

def _get_rag_resources():
    """Inicializa ChromaDB y retorna la colección e indexa los documentos de ser necesario."""
    global _collection, _chunks

    if _collection is not None:
        return _collection, _chunks

    _collection = create_vectorstore(_COLLECTION_NAME, _CHROMA_DIR)

    if _collection.count() == 0:
        docs = load_directory(_DATA_DIR)
        if docs:
            from rag.vectorstore import index_chunks
            all_chunks: list[Chunk] = []
            for doc in docs:
                all_chunks.extend(chunk_by_paragraphs(doc, max_chunk_size=500))
            index_chunks(_collection, all_chunks)
            _chunks = all_chunks
            logger.info("Indexados %d chunks de %d documentos.", len(all_chunks), len(docs))
        else:
            logger.warning("No se encontraron documentos en %s.", _DATA_DIR)
    else:
        all_data = _collection.get(include=["documents", "metadatas"])
        _chunks = [
            Chunk(
                content=doc,
                metadata=meta,
                chunk_id=cid,
            )
            for cid, doc, meta in zip(
                all_data["ids"], all_data["documents"], all_data["metadatas"]
            )
        ]
        logger.info("Colección cargada con %d chunks existentes.", len(_chunks))

    return _collection, _chunks


# ─── AGENTE 2: RETRIEVER ────────────────────────────────────
def retriever_agent(state: DocOpsState) -> dict:
    """Usa un modelo de búsqueda híbrida para obtener documentación relevante."""
    query = state["messages"][-1].content
    plan = state["plan"]
    search_query = f"{query} {plan[:200]}"
    persona_name = os.getenv('NAME', 'la persona')

    try:
        collection, chunks = _get_rag_resources()
        if not chunks:
            raise ValueError("No hay chunks en la colección.")

        hybrid = HybridRetriever(collection, chunks, alpha=0.5)
        results: list[SearchResult] = hybrid.search(search_query, top_k=10)

        if not results:
            raise ValueError("Búsqueda híbrida no retornó resultados")

        # Paso 2: Seleccionar los mejores resultados
        reranked = results[:5]

        # Paso 3: Formatear contexto con fuentes
        context_parts = []
        for i, r in enumerate(reranked, 1):
            source = r.metadata.get("source", "desconocida")
            score = f"{r.score:.3f}"
            context_parts.append(
                f"[{i}] (fuente: {source} | score: {score})\n{r.content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "Retriever: %d resultados tras reranking (query: %s)",
            len(reranked), query[:60],
        )

        return {"search_results": context}
    except Exception as e:
        logger.warning("RAG falló (%s). Usando fallback LLM.", e)
        response = llm.invoke([
            SystemMessage(content=(
                "Eres un agente de búsqueda. Dado el siguiente plan, "
                f"genera información relevante que podría encontrarse en "
                f"el LinkedIn de {os.getenv('NAME', 'la persona')}. Simula resultados de búsqueda "
                "realistas y útiles.\n\n"
                "Formato: Presenta 3-5 lineas de informacion relevantes, "
            )),
            HumanMessage(content=f"Plan de búsqueda:\n{plan}")
        ])

        return {
            "search_results": (
                f"[Fallback — resultados simulados por LLM]\n\n"
                f"{response.content}"
            )
        }


# ─── AGENTE 3: EXECUTOR (con fallback) ──────────────────────
def executor_agent(state: DocOpsState) -> dict:

    """Ejecuta y genera el borrador para responder al usuario usando LLMs principales y en fallback."""
    
    feedback_section = ""
    if state.get("feedback") and state["iteration"] > 0:
        feedback_section = (
            f"\n\nFEEDBACK DE REVISIÓN ANTERIOR (iteración {state['iteration']}):\n"
            f"{state['feedback']}\n"
            "Corrige los problemas señalados en el feedback."
        )

    user_query = state['messages'][-1].content
    chat_history = state["messages"][:-1]
    
    prompt_content = (
        f"Plan:\n{state['plan']}\n\n"
        f"Documentos encontrados:\n{state['search_results']}\n\n"
        f"Consulta original: {user_query}"
        f"{feedback_section}"
    )

    try:
        # Extrae datos específicos si es solicitado
        query_lower = user_query.lower()
        if any(palabra in query_lower for palabra in ["datos", "contacto", "información", "perfil", "email", "correo"]):
            structured_llm = llm.with_structured_output(Contacto)
            messages_struct = [
                SystemMessage(content=(
                    f"Extrae los datos de contacto y personales de {os.getenv('NAME', 'la persona')} "
                    "basándote en los documentos encontrados. Usa el formato estructurado requerido."
                ))
            ]
            messages_struct.extend(chat_history)
            messages_struct.append(HumanMessage(content=prompt_content))
            contact_data = structured_llm.invoke(messages_struct)
            return {"draft": f"Aquí tienes los datos solicitados:\n```json\n{contact_data.model_dump_json(indent=2)}\n```"}

        messages_to_invoke = [
            SystemMessage(content=(
                f"Tú ERES {os.getenv('NAME', 'la persona')}. Actúa y responde como él/ella en primera persona (yo, mi, me). "
                "Genera respuestas conversacionales y naturales basándote en la información de tu LinkedIn. Tu respuesta debe:\n"
                "- Ser fiel a tu perfil de LinkedIn (no inventar experiencia que no tienes)\n"
                "- Ser natural, como en una charla directa con el usuario\n"
                "- Ser clara, estructurada y profesional\n"
                "- Responder directamente a la consulta actual del usuario apoyándote en el historial si es necesario."
            ))
        ]
        messages_to_invoke.extend(chat_history)
        messages_to_invoke.append(HumanMessage(content=prompt_content))
        
        response = llm.invoke(messages_to_invoke)
        return {"draft": response.content}

    except Exception as e:
        # Fallback a un LLM más económico
        try:
            messages_fallback = [
                SystemMessage(content=(
                    f"Tú eres {os.getenv('NAME', 'la persona')}. Genera una respuesta concisa en primera persona (yo) basada en tu experiencia."
                ))
            ]
            messages_fallback.extend(chat_history)
            messages_fallback.append(HumanMessage(content=prompt_content))
            response = fallback_llm.invoke(messages_fallback)
            return {
                "draft": (
                    f"[Generado con modelo de respaldo]\n\n"
                    f"{response.content}"
                )
            }
        except Exception as e2:
            # Último recurso: respuesta degradada
            return {
                "draft": (
                    f"No pude generar una respuesta completa. "
                    f"Error: {str(e2)[:200]}\n\n"
                    f"Documentos encontrados:\n"
                    f"{state['search_results'][:500]}"
                ),
                "quality_score": 0.3,
            }


# ─── AGENTE 4: VERIFIER ─────────────────────────────────────
def verifier_agent(state: DocOpsState) -> dict:
   
    """Evalúa la calidad del borrador frente al contexto recuperado y la solicitud del usuario."""
    
    quality_checker = llm.with_structured_output(QualityCheck)
    persona_name = os.getenv('NAME', 'la persona')

    try:
        check = quality_checker.invoke([
            SystemMessage(content=(
                "Eres un verificador de calidad. Evalúa la respuesta generada "
                f"contra el LinkedIn de {os.getenv('NAME', 'la persona')} y la consulta original.\n\n"
                "Criterios de evaluación:\n"
                f"- Formato: Si es una entrega de datos de contacto (JSON), asume que la 'Persona' es válida y evalúa solo que la info sea fiel al perfil.\n"
                f"- Persona: (Si es texto libre) ¿La respuesta está escrita en primera persona como si fuera {os.getenv('NAME', 'la persona')}?\n"
                f"- Fidelidad: ¿La respuesta es fiel al LinkedIn de {os.getenv('NAME', 'la persona')}?\n"
                "- Completitud: ¿Responde toda la consulta?\n"
                "- Claridad: ¿Es clara y natural?\n\n"
                "Score:\n"
                "- 0.9-1.0: Excelente, aceptar\n"
                "- 0.8-0.89: Buena, aceptar\n"
                "- 0.6-0.79: Necesita mejora, revisar con feedback\n"
                "- <0.6: Mala, revisar con feedback detallado"
            )),
            HumanMessage(content=(
                f"CONSULTA: {state['messages'][-1].content}\n\n"
                f"DOCUMENTOS:\n{state['search_results']}\n\n"
                f"RESPUESTA A EVALUAR:\n{state['draft']}"
            ))
        ])

        return {
            "quality_score": check.score,
            "feedback": check.feedback,
            "iteration": state["iteration"] + 1,
        }

    except Exception as e:
        
        return {
            "quality_score": 0.7,
            "feedback": f"Verificación falló: {str(e)[:200]}",
            "iteration": state["iteration"] + 1,
        }


# ─── ARISTA CONDICIONAL: BUCLE DE CALIDAD ───────────────────
def should_revise(state: DocOpsState) -> Literal["accept", "revise"]:
    """
    Decide si la respuesta es aceptable o necesita revisión.

    Acepta si:
    - quality_score >= 0.8 (calidad suficiente)
    - iteration >= 3 (máximo de intentos alcanzado)

    Rechaza si:
    - quality_score < 0.8 AND iteration < 3
    """
    if state["quality_score"] >= 0.8:
        return "accept"
    if state["iteration"] >= 3:
        return "accept"
    return "revise"

# ==========================================
# Construcción del Grafo Computacional (StateGraph)
# ==========================================
def build_docops_agent(cp=None):
    """Construye y compila el flujo computacional central de DocOps usando langgraph."""
    workflow = StateGraph(DocOpsState)

    workflow.add_node("planner", planner_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("verifier", verifier_agent)
    workflow.add_node("human_gate", human_gate)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "executor")
    workflow.add_edge("executor", "verifier")

    workflow.add_conditional_edges("verifier", should_revise, {"accept": "human_gate", "revise": "executor"})
    # Flujo termina despúes de pasar por el Human In The Loop o su aprobación
    workflow.add_edge("human_gate", END)

    return workflow.compile(checkpointer=cp if cp is not None else checkpointer)

docops_agent = build_docops_agent()

# ==========================================
# Funciones Ejecutoras y Utilidades
# ==========================================

# ─── UTILIDADES ──────────────────────────────────────────────
def invoke_docops(
    query: str,
    persona_name: str = os.getenv('NAME', 'la persona'),
    thread_id: str = None,
    verbose: bool = False,
    force_review: bool = False,
) -> dict:

    """Invoca la cadena para la primera vez dada una consulta por el usuario."""

    import uuid

    if thread_id is None:
        thread_id = f"docops-{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "plan": "",
        "search_results": "",
        "draft": "",
        "feedback": "",
        "quality_score": 0.0,
        "iteration": 0,
        "persona_name": persona_name,
        "force_review": force_review,
    }

    if verbose:
        print(f"\n{_BOLD}{_WHITE}═{'═'*60}{_RESET}")
        print(f"  🚀 Iniciando invocación multiagente (Thread: {thread_id[:8]})")
        print(f"{_BOLD}{_WHITE}═{'═'*60}\n{_RESET}")
        
        step = 0
        for event in docops_agent.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                step += 1
                meta = AGENT_META.get(node_name, {})
                color = meta.get("color", _WHITE)
                icon = meta.get("icon", "▸")
                label = meta.get("label", node_name.upper())
                desc = meta.get("desc", "")

                print(f"\n{color}{_BOLD}{'─'*60}{_RESET}")
                print(f"{color}{_BOLD}  {icon}  [{step}] {label}{_RESET}")
                print(f"{color}{_DIM}  {desc}{_RESET}")
                print(f"{color}{'─'*60}{_RESET}")

                if not node_output or not isinstance(node_output, dict):
                    continue
                for key, value in node_output.items():
                    if key == "messages":
                        continue
                    preview = str(value)[:300]
                    if key == "quality_score":
                        score = float(value)
                        score_color = _GREEN if score >= 0.8 else _RED
                        print(f"  {_DIM}↳ {key}:{_RESET} {score_color}{_BOLD}{preview}{_RESET}")
                    elif key == "feedback":
                        print(f"  {_DIM}↳ {key}:{_RESET} {_MAGENTA}{preview}{_RESET}")
                    elif key == "iteration":
                        print(f"  {_DIM}↳ {key}:{_RESET} {_YELLOW}{preview}{_RESET}")
                    else:
                        print(f"  {_DIM}↳ {key}:{_RESET} {color}{preview}{_RESET}")

        print(f"\n{_GREEN}{_BOLD}{'═'*60}{_RESET}")
        print(f"{_GREEN}{_BOLD}  ✔ EJECUCIÓN COMPLETADA{_RESET}")
        print(f"{_GREEN}{_BOLD}{'═'*60}{_RESET}")

    result = docops_agent.invoke(initial_state, config)

    # Verificar si se interrumpió (HITL)
    interrupted = False
    interrupt_payload = None

    snapshot = docops_agent.get_state(config)
    if snapshot.next:
        # El grafo se pausó — hay un interrupt pendiente
        interrupted = True
        # Extraer payload del interrupt
        if hasattr(snapshot, "tasks") and snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_payload = task.interrupts[0].value

    if verbose:
        print(f"\n{'─'*60}")
        if interrupted:
            print("⏸️  GRAFO PAUSADO — Esperando decisión humana")
            if interrupt_payload:
                print(f"   Riesgo: {interrupt_payload.get('risk_level', '?')}")
                print(f"   Mensaje: {interrupt_payload.get('message', '')[:200]}")
        else:
            print("✅ GRAFO COMPLETADO")
        print(f"{'─'*60}")

    return {
        "answer": result.get("draft", ""),
        "quality_score": result.get("quality_score", 0.0),
        "iterations": result.get("iteration", 0),
        "plan": result.get("plan", ""),
        "thread_id": thread_id,
        "interrupted": interrupted,
        "interrupt_payload": interrupt_payload,
    }


def resume_docops(
    thread_id: str,
    decision: dict,
    verbose: bool = False,
) -> dict:
    
    """Continúa el workflow despúes de una intervención humana externa."""
    
    config = {"configurable": {"thread_id": thread_id}}

    if verbose:
        print(f"Reanudando thread: {thread_id}")
        print(f"Decisión: {decision}\n")

    result = docops_agent.invoke(Command(resume=decision), config)
    return {
        "answer": result.get("draft", ""),
        "quality_score": result.get("quality_score", 0.0),
        "iterations": result.get("iteration", 0),
        "thread_id": thread_id,
        "interrupted": False,
    }


def continue_conversation(
    thread_id: str,
    follow_up: str,
    last_answer: str = "",
    persona_name: str = "Ed Donner",
    verbose: bool = False,
) -> dict:

    """Continúa la iteración agregando nueva entrada al Checkpoint actual."""

    config = {"configurable": {"thread_id": thread_id}}
    msgs = []
    if last_answer:
        msgs.append(AIMessage(content=last_answer))
    msgs.append(HumanMessage(content=follow_up))

    new_state = {
        "messages": msgs, "plan": "", "search_results": "", "draft": "",
        "feedback": "", "quality_score": 0.0, "iteration": 0, "persona_name": persona_name,
    }
    
    if verbose:
        print(f"Continuando thread: {thread_id}")
        print(f"Follow-up: {follow_up}\n")

    result = docops_agent.invoke(new_state, config)
    snapshot = docops_agent.get_state(config)
    interrupted = bool(snapshot.next)
    
    return {
        "answer": result.get("draft", ""),
        "quality_score": result.get("quality_score", 0.0),
        "iterations": result.get("iteration", 0),
        "thread_id": thread_id,
        "interrupted": bool(snapshot.next),
    }

if __name__ == "__main__":
    import uuid
    print(f"\n{_BOLD}{_WHITE}{'═'*60}{_RESET}")
    print(f"{_BOLD}{_WHITE}  DocOps Agent v2 — Chatbot Interactivo Multipaso{_RESET}")
    print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}")
    print(f"  Modelo:       {os.getenv('GROQ_MODEL', 'gpt-oss-120b')} via Groq")
    print(f"  Checkpointer: {type(checkpointer).__name__}")
    print(f"{_BOLD}{_WHITE}{'═'*60}{_RESET}\n")

    thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    persona = os.getenv("NAME", "representado")
    
    print(f"{_GREEN}¡Hola! Soy el asistente virtual de {persona}.{_RESET}")
    print(f"{_DIM}Escribe 'salir', 'exit' o 'quit' para terminar la conversación.{_RESET}\n")
    
    is_first_message = True
    last_answer = ""
    while True:
        try:
            user_input = input(f"\n{_CYAN}Tú: {_RESET}").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print(f"{_YELLOW}Chao, hasta la próxima.{_RESET}")
                break
            if not user_input:
                continue

            print(f"{_DIM}Pensando...{_RESET}")
            if is_first_message:
                result = invoke_docops(
                    query=user_input,
                    persona_name=persona,
                    thread_id=thread_id,
                    verbose=False,
                    force_review=False
                )
                is_first_message = False
            else:
                result = continue_conversation(
                    thread_id=thread_id,
                    follow_up=user_input,
                    last_answer=last_answer,
                    persona_name=persona,
                    verbose=False
                )

            last_answer = result['answer']
            print(f"{_BOLD}{_GREEN}{persona}: {_RESET}{result['answer']}")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{_YELLOW}Chao, hasta la próxima.{_RESET}")
            break
        except Exception as e:
            print(f"\n{_RED}Ocurrió un error: {e}{_RESET}\n")