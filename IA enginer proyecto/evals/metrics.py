import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class EvaluationResult(BaseModel):
    score: float = Field(description="Score de 0.0 a 1.0 indicando la precisión y similitud semántica con la respuesta golden.")
    justification: str = Field(description="Justificación detallada del por qué se asignó dicho score.")

def evaluate_answer(question: str, generated_answer: str, golden_answer: str) -> EvaluationResult:
    """Evalúa una respuesta generada contra la golden answer usando un LLM como juez."""
    
    llm = ChatOpenAI(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0,
    )
    
    evaluator = llm.with_structured_output(EvaluationResult)
    
    system_prompt = (
        "Eres un juez experto evaluando la calidad de las respuestas de un chatbot.\n"
        "Compara la respuesta generada con la respuesta 'golden' (ideal) según la pregunta hecha.\n\n"
        "Criterios de evaluación (Score de 0.0 a 1.0):\n"
        "- 1.0: Contiene toda la información clave de la respuesta golden, incluso si es con otras palabras. No hay alucinaciones.\n"
        "- 0.8: Contiene la mayoría de la información de la respuesta golden. Pequeñas omisiones.\n"
        "- 0.5: Parcialmente correcta. Falta información importante.\n"
        "- 0.2: Información mayormente incorrecta o irrelevante.\n"
        "- 0.0: Totalmente incorrecta, alucinada, o no responde la pregunta."
    )
    
    user_prompt = (
        f"PREGUNTA: {question}\n\n"
        f"RESPUESTA GOLDEN (Esperada): {golden_answer}\n\n"
        f"RESPUESTA GENERADA (A evaluar): {generated_answer}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        result = evaluator.invoke(messages)
        return result
    except Exception as e:
        return EvaluationResult(
            score=0.0,
            justification=f"Error en la evaluación (LLM falló): {str(e)}"
        )
