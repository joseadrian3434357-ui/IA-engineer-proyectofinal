"""Tests para agents.multiagents."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from agents.multiagents import (
    DocOpsState,
    should_revise,
    planner_agent,
    executor_agent,
)

class TestMultiagents:
    """Tests unitarios para los nodos principales de multiagents."""

    def test_should_revise_logic(self):
        """Verificar la lógica de reinicio/aceptación en la arista condicional del Verifier."""
        # Quality score alto, se acepta
        state_pass = {"quality_score": 0.85, "iteration": 1}
        assert should_revise(state_pass) == "accept"
        
        state_pass_perfect = {"quality_score": 1.0, "iteration": 0}
        assert should_revise(state_pass_perfect) == "accept"

        # Quality score bajo, se revisa
        state_fail = {"quality_score": 0.7, "iteration": 1}
        assert should_revise(state_fail) == "revise"
        
        # Supera límite de iteraciones, se acepta para evitar loop infinito
        state_max_iter = {"quality_score": 0.5, "iteration": 3}
        assert should_revise(state_max_iter) == "accept"

    @patch("agents.multiagents.llm")
    def test_planner_agent(self, mock_llm):
        """El planner_agent debe retornar un diccionario con el plan generado por el LLM."""
        mock_response = MagicMock()
        mock_response.content = "Plan estructurado de prueba."
        mock_llm.invoke.return_value = mock_response

        state = {
            "messages": [HumanMessage(content="¿Cuál es tu cargo?")],
        }

        result = planner_agent(state)
        
        assert "plan" in result
        assert result["plan"] == "Plan estructurado de prueba."
        assert result["iteration"] == 0
        mock_llm.invoke.assert_called_once()

    @patch("agents.multiagents.llm")
    @patch("agents.multiagents.os.getenv")
    def test_executor_agent_with_feedback(self, mock_getenv, mock_llm):
        """El executor_agent debe generar un borrador e incorporar el feedback pasado en el state."""
        mock_getenv.return_value = "Usuario Test"
        
        mock_response = MagicMock()
        mock_response.content = "Borrador inicial respondiendo al usuario."
        mock_llm.invoke.return_value = mock_response

        state = {
            "messages": [HumanMessage(content="Hola")],
            "plan": "Saludar al usuario.",
            "search_results": "Info del perfil.",
            "iteration": 1,
            "feedback": "Hazlo más amable.",
        }

        result = executor_agent(state)
        
        assert "draft" in result
        assert result["draft"] == "Borrador inicial respondiendo al usuario."
        
        # Verificar que el LLM fue invocado
        mock_llm.invoke.assert_called_once()
        
        # Comprobar que en la llamada al LLM se incluyó el feedback
        called_args = mock_llm.invoke.call_args[0][0]
        # called_args es la lista de mensajes (System, Human)
        human_msg = called_args[-1].content
        assert "FEEDBACK DE REVISIÓN ANTERIOR" in human_msg
        assert "Hazlo más amable." in human_msg
