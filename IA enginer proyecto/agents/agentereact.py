import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agents.tools_agents import TOOLS_REGISTRY, execute_tool, parse_action, set_represented_profile

load_dotenv()

logger = logging.getLogger(__name__)

AGENT_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _profile_for_prompt(path: Path) -> str:
    try:
        return str(path.relative_to(AGENT_PROJECT_ROOT))
    except ValueError:
        return str(path)

# ANSI terminal styles (used when verbose=True)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"

name = "Ed Donner"

REACT_SYSTEM_PROMPT = """\

Estas actuando como {name}. Respondes preguntas en la pagina web de {name}, 
particularmente preguntas acerca de la carrera de {name}, antecedentes, habilidades y experencia. 
Tu responsabulidad es representar a {name} para interacciones en su pagina web lo mas fielmente posible. 
Tienes un documento sobre {name} para reponder preguntas. 

Resuelve las tareas intercalando Thought, Action y Observation paso a paso.

IMPORTANTE: NO USES el feature nativo de tool calling en JSON. Solo responde con texto plano en estricto formato:
Thought N: [razonamiento]
Action N: tool_name["argumento"]

Herramientas disponibles:
- search_linkedin["query"]: Usa HybridRetrieval de rag/retrieval.py para buscar en tu perfil de LinkedIn ({profile_doc}). Usa consultas en lenguaje natural.
- lookup["term"]: Busca un término específico en tu documento de perfil ({profile_doc}).
- Finish["respuesta"]: Termina y responde al usuario. Usa esta herramienta para dar tu respuesta final como chatbot.

Reglas:
1. SIEMPRE genera un Thought antes de cada Action
2. Usa search_linkedin para encontrar información relevante
3. Usa lookup para acotar por palabra clave en el documento de perfil
4. Usa Finish SOLO cuando tengas evidencia suficiente de los documentos. Una vez que search_linkedin encuentre la respuesta, DEBES usar la herramienta Finish en el siguiente paso.
5. Si una búsqueda no retorna resultados útiles, reformula con otros términos
6. Máximo {max_steps} pasos. Si no encuentras respuesta, usa Finish con lo que tengas.
7. Responde en español.

Formato estricto por paso:
Thought N: [tu razonamiento sobre qué hacer]
Action N: tool_name["argumento"]

Después de cada Action, recibirás:
Observation N: [resultado de la herramienta]

Ejemplo:
Question: ¿A que te dedicas?
Thought 1: Necesito buscar información sobre a que se dedica {name}.
Action 1: search_linkedin["software engineer scientist career"]
Observation 1: [1] ({profile_file}): Im a software engineer and a scientist...
Thought 2: Encontré la información con search_linkedin, por lo que utilizaré Finish inmediatamente.
Action 2: Finish["Me dedico al software enginner y scientist."]

Ejemplo 2:
Question: ¿Que es lo mejor que sabes hacer?
Thought 1: Necesito buscar lo mejor que sabe hacer {name}.
Action 1: search_linkedin["skills experience strengths"]
Observation 1: [1] ({profile_file}): No se encontraron datos relevantes. Intenta reformular
Thought 2: La búsqueda no encontró resultados. Voy a intentar con lookup sobre habilidades.
Action 2: lookup["Top Skills"]
Observation 2: Top Skills: CTO, Large Language Models (LLM), PyTorch
Thought 3: Encontré las habilidades destacadas gracias a lookup. Usare Finish para responder.
Action 3: Finish["Lo que mejor se hacer es CTO, Large Language Models (LLM), PyTorch."]
"""

class ReactAgent:
    def __init__(
        self,
        tools: dict | None = None,
        model: str = "openai/gpt-oss-120b",
        max_steps: int = 100000,
        linkedin_doc: str | Path | None = None,
    ):
        self.profile_path = set_represented_profile(linkedin_doc)
        self.tools = tools or TOOLS_REGISTRY
        self.model = model
        self.max_steps = max_steps
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=30.0,
        )

    def run(self, query: str, history: list[dict] | None = None, verbose: bool = True) -> dict:
        steps: list[dict] = []
        
        chat_context = ""
        if history:
            chat_context = "Historial de la conversación:\n"
            for msg in history[-5:]:
                role = "User" if msg.get("role") == "user" else name
                chat_context += f"{role}: {msg.get('content', '')}\n"
            chat_context += "\n"

        trajectory = f"{chat_context}Question: {query}\n"
        system = REACT_SYSTEM_PROMPT.format(
            name=name,
            max_steps=self.max_steps,
            profile_doc=_profile_for_prompt(self.profile_path),
            profile_file=self.profile_path.name,
        )

        if verbose:
            print(f"\n{_BOLD}{'─' * 60}{_RESET}")
            print(f"{_BOLD}  AGENTE REACT (Thought → Action → Observation){_RESET}")
            print(f"{_DIM}  Query: {query}{_RESET}")
            print(f"{_BOLD}{'─' * 60}{_RESET}")

        for step_num in range(1, self.max_steps + 1):
            # Detección de loops
            if self._detect_loop(trajectory):
                loop_thought = (
                    "Estoy repitiendo acciones. "
                    "Debo reformular mi estrategia con términos diferentes."
                )
                trajectory += f"Thought {step_num}: {loop_thought}\n"
                if verbose:
                    print(f"  {_RED}[Loop detectado]{_RESET}")
                    print(f"  {_BLUE}Thought {step_num}:{_RESET} {_BLUE}{loop_thought}{_RESET}")
                logger.warning(
                    "Step %d - Loop detected, injecting reformulation thought",
                    step_num,
                )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": trajectory},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                raw = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("Step %d - LLM error: %s", step_num, e)
                if verbose:
                    print(f"  {_RED}Error: {e}{_RESET}")
                steps.append(
                    {
                        "step": step_num,
                        "thought": "Error en LLM",
                        "action": "error",
                        "observation": str(e),
                        "success": False,
                    }
                )
                break

            thought, action_line = self._parse_react_output(raw, step_num)
            logger.info("Step %d - Thought: %s", step_num, thought[:100])
            logger.info("Step %d - Action: %s", step_num, action_line)

            tool_call = parse_action(action_line)

            # Finish
            if tool_call.tool == "Finish":
                if verbose:
                    print(f"  {_BLUE}Thought {step_num}:{_RESET} {_BLUE}{thought}{_RESET}")
                    print(f"  {_YELLOW}Action {step_num}:{_RESET} {action_line[:70]}")
                    print(f"\n  {_GREEN}{_BOLD}Respuesta:{_RESET} {_GREEN}{tool_call.argument}{_RESET}")
                    print(f"  {_DIM}Total pasos: {step_num}{_RESET}\n")
                steps.append(
                    {
                        "step": step_num,
                        "thought": thought,
                        "action": action_line,
                        "observation": tool_call.argument,
                        "success": True,
                    }
                )
                trajectory += (
                    f"Thought {step_num}: {thought}\n"
                    f"Action {step_num}: {action_line}\n"
                )
                logger.info("Step %d - Finished: %s", step_num, tool_call.argument[:100])

                print(trajectory)
                return {
                    "answer": tool_call.argument,
                    "steps": steps,
                    "total_steps": step_num,
                    "trajectory": trajectory,
                }

            # Execute tool
            result = execute_tool(tool_call)
            observation = result.output

            if verbose:
                print(f"  {_BLUE}Thought {step_num}:{_RESET} {_BLUE}{thought}{_RESET}")
                print(f"  {_YELLOW}Action {step_num}:{_RESET} {action_line[:70]}")
                print(f"  {_MAGENTA}Observation {step_num}:{_RESET} {_MAGENTA}{observation[:120]}{_RESET}")
                print()

            steps.append(
                {
                    "step": step_num,
                    "thought": thought,
                    "action": action_line,
                    "observation": observation[:200],
                    "success": result.success,
                }
            )
            trajectory += (
                f"Thought {step_num}: {thought}\n"
                f"Action {step_num}: {action_line}\n"
                f"Observation {step_num}: {observation}\n"
            )
            logger.info("Step %d - Observation: %s", step_num, observation[:100])

        if verbose:
            print(f"  {_RED}Max pasos alcanzados sin respuesta final.{_RESET}\n")
        


        
        return {
            "answer": None,
            "steps": steps,
            "total_steps": len(steps),
            "trajectory": trajectory,
        }

    @staticmethod
    def _detect_loop(trajectory: str, window: int = 3) -> bool:
        """Detecta si las últimas `window` acciones son idénticas."""
        actions = re.findall(r"Action\s+\d+:\s*(.+)", trajectory)
        if len(actions) < window:
            return False
        return len(set(actions[-window:])) == 1

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Elimina marcadores de markdown (**, *, etc.) del texto."""
        return re.sub(r"\*{1,2}", "", text).strip()

    def _parse_react_output(self, raw: str, step_num: int) -> tuple[str, str]:
        """Extrae Thought y Action de la salida del LLM."""
        # Strip markdown bold markers
        cleaned = self._clean_markdown(raw)

        # Thought — try with step number, then any
        thought_match = re.search(
            rf"Thought\s*{step_num}?\s*:\s*(.+?)(?=Action|$)", cleaned, re.DOTALL
        )
        thought = (
            thought_match.group(1).strip()
            if thought_match
            else "Sin razonamiento explícito"
        )

        # Action — try exact step, then any step
        action_match = re.search(rf"Action\s*{step_num}\s*:\s*(.+)", cleaned)
        if not action_match:
            action_match = re.search(r"Action\s*\d*\s*:\s*(.+)", cleaned)

        if action_match:
            return thought, action_match.group(1).strip()

        # Fallback: look for tool call pattern
        for line in reversed(cleaned.split("\n")):
            if re.search(r"\w+\s*[\[\(]", line):
                return thought, line.strip()

        # Last resort: model gave a direct answer — wrap in Finish
        direct = cleaned.replace("\n", " ")[:300]
        return thought, f'Finish["{direct}"]'

