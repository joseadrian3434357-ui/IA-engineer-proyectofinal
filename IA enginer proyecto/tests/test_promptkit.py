import pytest
from prompting.promptkit import PromptTemplate, PromptChain

def test_prompt_template_formatting():
    """Verifica que un template formatee correctamente cadenas con variables."""
    pt = PromptTemplate(
        name="test_prompt",
        template="Hola {nombre}. Tienes {n_mensajes} mensajes nuevos.",
        metadata={"version": "1.0"}
    )
    
    result = pt.render(nombre="Eduardo", n_mensajes=5)
    assert result == "Hola Eduardo. Tienes 5 mensajes nuevos."

def test_prompt_chain():
    """Verifica el flujo encadenado de la estructura de PromptChain."""
    pt_1 = PromptTemplate(name="step_1", template="Calcula paso 1: {input}")
    pt_2 = PromptTemplate(name="step_2", template="Calcula paso 2 sobre: {step_1}")
    
    chain = PromptChain(templates=[pt_1, pt_2])
    
    assert len(chain.templates) == 2
    
    # Comprobar si los inputs en un uso hipotetico de cadena son renderizables (mock behavior)
    assert pt_1.name == "step_1"
    assert pt_2.name == "step_2"
    
def test_prompt_template_missing_kwargs():
    """Lanzar KeyError si se formatea faltando un kwarg especificado."""
    pt = PromptTemplate(
        name="test_prompt",
        template="El valor es {valor}"
    )
    
    with pytest.raises(KeyError):
        pt.render(otro_valor=1)
