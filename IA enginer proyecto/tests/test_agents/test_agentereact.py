"""Tests para agents.agentereact."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from agents.agentereact import ReactAgent

class TestReactAgent:
    """Tests unitarios para las funciones utilitarias y de control del ReactAgent."""

    def test_parse_react_output_valid(self):
        """Verifica que un bloque válido Action y Thought sea parseado correctamente."""
        agent = ReactAgent()
        
        raw_output = (
            "Thought 1: Necesito buscar en mi perfil mi experiencia de vida.\n"
            "Action 1: search_linkedin[\"experiencia de vida\"]"
        )
        
        thought, action = agent._parse_react_output(raw_output, step_num=1)
        assert thought == "Necesito buscar en mi perfil mi experiencia de vida."
        assert action == 'search_linkedin["experiencia de vida"]'

    def test_parse_react_output_with_markdown(self):
        """Verifica la limpieza de markdown antes de parsear."""
        agent = ReactAgent()
        
        raw_output = (
            "**Thought 2:** La búsqueda devolvió resultados relevantes.\n"
            "**Action 2:** Finish[\"Hola\"]"
        )
        
        thought, action = agent._parse_react_output(raw_output, step_num=2)
        assert thought == "La búsqueda devolvió resultados relevantes."
        assert action == 'Finish["Hola"]'

    def test_parse_react_output_direct_fallback(self):
        """Si el LLM olvida el formato y da una respuesta directa, hace fallback a Finish."""
        agent = ReactAgent()
        
        raw_output = "Soy un asistente de IA."
        
        # Al no matchear Action ni Thought explícito, asume fallback
        thought, action = agent._parse_react_output(raw_output, step_num=1)
        assert action == 'Finish["Soy un asistente de IA."]'

    def test_detect_loop(self):
        """El agente debería detectar si está enclavado llamando a la misma acción."""
        agent = ReactAgent()
        
        # Loop: 3 acciones seguidas idénticas
        trajectory_loop = (
            "Thought 1: a\nAction 1: lookup[\"habilidad\"]\n"
            "Thought 2: b\nAction 2: lookup[\"habilidad\"]\n"
            "Thought 3: c\nAction 3: lookup[\"habilidad\"]\n"
        )
        assert agent._detect_loop(trajectory_loop, window=3) is True
        
        # Sin loop
        trajectory_no_loop = (
            "Thought 1: a\nAction 1: lookup[\"habilidad\"]\n"
            "Thought 2: b\nAction 2: lookup[\"otra\"]\n"
            "Thought 3: c\nAction 3: lookup[\"habilidad\"]\n"
        )
        assert agent._detect_loop(trajectory_no_loop, window=3) is False

    @patch("agents.agentereact.execute_tool")
    def test_agent_run_finish_immediately(self, mock_execute_tool):
        """Verifica que el bucle de razonamiento se detenga cuando la Acción es Finish."""
        # Se mockea al cliente LLM del agent
        with patch("agents.agentereact.OpenAI") as mock_openai:
            mock_client = MagicMock()
            
            # Simulamos que el LLM directo decide Finish
            mock_completions = MagicMock()
            mock_completions.choices = [MagicMock()]
            mock_completions.choices[0].message.content = (
                "Thought 1: No necesito buscar.\n"
                "Action 1: Finish[\"La respuesta es correcta\"]"
            )
            mock_client.chat.completions.create.return_value = mock_completions
            mock_openai.return_value = mock_client
            
            agent = ReactAgent()
            # Inyectamos el mock ya instanciado
            agent.client = mock_client
            
            result = agent.run("Pregunta de prueba", verbose=False)
            
            assert result["total_steps"] == 1
            assert result["answer"] == "La respuesta es correcta"
            # execute_tool NO debió ser llamado para un Finish
            mock_execute_tool.assert_not_called()
