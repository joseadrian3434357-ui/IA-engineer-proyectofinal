"""Chatbot response classifier prompt templates (v1-v4) for multiagents."""

from __future__ import annotations

from prompting.promptkit import PromptChain, PromptRegistry, PromptTemplate

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------
registry = PromptRegistry()

# ---------------------------------------------------------------------------
# v1 — Base (intentionally weak / ambiguous)
# ---------------------------------------------------------------------------
v1 = PromptTemplate(
    name="response_classifier_v1",
    template=(
        "Clasifica la siguiente respuesta dada por un chatbot de soporte/perfil profesional.\n"
        "Respuesta del chatbot: {respuesta}\n"
        "Responde con la categoría y la calidad de la respuesta."
    ),
    metadata={"version": "1.0", "description": "Base - prompt débil y ambiguo"},
)

# ---------------------------------------------------------------------------
# v2 — Few-shot with explicit format
# ---------------------------------------------------------------------------
v2 = PromptTemplate(
    name="response_classifier_v2",
    template=(
        "Clasifica la siguiente respuesta generada por un chatbot (agente virtual).\n"
        "\n"
        "Categorías válidas: informativa, contacto, saludo, disculpa, fuera_de_dominio\n"
        "Calidad válida: alta, media, baja\n"
        "\n"
        'Responde en formato JSON: {{"categoria": "...", "calidad": "..."}}\n'
        "\n"
        "Ejemplos:\n"
        "\n"
        'Input: "¡Hola! Soy el asistente virtual de Eduardo. ¿En qué te puedo ayudar hoy?"\n'
        'Output: {{"categoria": "saludo", "calidad": "alta"}}\n'
        "\n"
        'Input: "Mi correo es eduardo@example.com y mi teléfono es 123456789"\n'
        'Output: {{"categoria": "contacto", "calidad": "alta"}}\n'
        "\n"
        'Input: "Tengo experiencia de 5 años trabajando con Python y React en distintas empresas."\n'
        'Output: {{"categoria": "informativa", "calidad": "alta"}}\n'
        "\n"
        'Input: "No entiendo la pregunta."\n'
        'Output: {{"categoria": "disculpa", "calidad": "baja"}}\n'
        "\n"
        "Respuesta del chatbot: {respuesta}"
    ),
    metadata={"version": "2.0", "description": "Few-shot con formato JSON explícito para respuestas"},
)

# ---------------------------------------------------------------------------
# v3 — Restrictions on top of v2
# ---------------------------------------------------------------------------
v3 = PromptTemplate(
    name="response_classifier_v3",
    template=(
        "Clasifica la siguiente respuesta generada por un chatbot (agente virtual).\n"
        "\n"
        "Categorías válidas: informativa, contacto, saludo, disculpa, fuera_de_dominio\n"
        "Calidad válida: alta, media, baja\n"
        "\n"
        "Reglas:\n"
        "- Responde SOLO con JSON válido, sin texto adicional ni markdown\n"
        "- Usa EXACTAMENTE una de las categorías listadas\n"
        "- Si la respuesta es cortante, repetitiva o parece un error del sistema documental, asigna calidad 'baja'\n"
        "- Si la respuesta incluye información valiosa (experiencia, habilidades), clasifica como 'informativa' y calidad 'alta'\n"
        '- Si la respuesta responde a algo no relacionado al profesional (como pedir una receta), clasifica como "fuera_de_dominio"\n'
        "\n"
        'Formato de respuesta: {{"categoria": "...", "calidad": "..."}}\n'
        "\n"
        "Ejemplos:\n"
        "\n"
        'Input: "Mi correo es eduardo@example.com y mi teléfono es 123456789"\n'
        'Output: {{"categoria": "contacto", "calidad": "alta"}}\n'
        "\n"
        'Input: "Tengo experiencia de 5 años trabajando con Python y React en distintas empresas."\n'
        'Output: {{"categoria": "informativa", "calidad": "alta"}}\n'
        "\n"
        'Input: "¿Cómo hacer un pastel de chocolate? Mezcla harina..."\n'
        'Output: {{"categoria": "fuera_de_dominio", "calidad": "media"}}\n'
        "\n"
        "Respuesta del chatbot: {respuesta}"
    ),
    metadata={"version": "3.0", "description": "Restricciones estrictas para calificar al chatbot"},
)

# ---------------------------------------------------------------------------
# v4 — Two-step PromptChain
# ---------------------------------------------------------------------------
_v4_step1 = PromptTemplate(
    name="response_classifier_v4_extractor",
    template=(
        "Analiza la siguiente respuesta dada por un chatbot y extrae la siguiente información:\n"
        "- tema_principal: de qué habla la respuesta (ej. experiencia, contacto, saludo, error)\n"
        "- tono: profesional, amigable, cortante, error_de_sistema\n"
        "- contiene_datos_personales: sí o no\n"
        "- coherencia: alta, media o baja\n"
        "\n"
        "Responde en JSON.\n"
        "\n"
        "Respuesta del chatbot: {respuesta}"
    ),
    metadata={"version": "4.0", "description": "Chain paso 1 - extractor de características de respuesta"},
)

_v4_step2 = PromptTemplate(
    name="response_classifier_v4_clasificador",
    template=(
        "Usando la siguiente extracción de características de una respuesta de chatbot:\n"
        "{extraction_result}\n"
        "\n"
        "Clasifica la respuesta con una categoría y calidad.\n"
        "\n"
        "Categorías válidas: informativa, contacto, saludo, disculpa, fuera_de_dominio\n"
        "Calidad válida: alta, media, baja\n"
        "\n"
        "Reglas:\n"
        "- Responde SOLO con JSON válido, sin texto adicional ni markdown\n"
        '- Si "contiene_datos_personales" es "sí", la categoría probablemente es "contacto"\n'
        '- Si "coherencia" es baja o "tono" es error_de_sistema, la calidad DEBE ser "baja"\n'
        '- Formato: {{"categoria": "...", "calidad": "..."}}\n'
    ),
    metadata={"version": "4.0", "description": "Chain paso 2 - clasificador de respuesta final"},
)

v4_chain = PromptChain(templates=[_v4_step1, _v4_step2])

# ---------------------------------------------------------------------------
# Register all templates
# ---------------------------------------------------------------------------
registry.register(v1)
registry.register(v2)
registry.register(v3)
registry.register(_v4_step1)
registry.register(_v4_step2)
