import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from agents.multiagents import (
    DocOpsState,
    build_docops_agent,
    invoke_docops
)

class TestChatbotFlowIntegration:
    """Pruebas de integración E2E sobre multiagents.py."""

    @patch("agents.multiagents.llm")
    def test_invoke_docops_end_to_end_success(self, mock_llm):
        """Mockeamos el LLM local para simular la generación de grafo completo (planner -> retriever -> executor -> verifier)."""
        
        # Configuramos MagicMock para modelar el .invoke() y .with_structured_output()
        mock_invoke_result = MagicMock()
        mock_invoke_result.content = "Contenido simulado perfecto"
        mock_llm.invoke.return_value = mock_invoke_result

        # Mock de salida estructurada para el Verifier
        mock_verifier_struct = MagicMock()
        mock_verifier_struct.score = 0.95  # Alta calidad
        mock_verifier_struct.feedback = "Excelente."
        mock_verifier_struct.decision = "accept"
        
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_verifier_struct

        # Interceptar el vectorstore / retriever para no depender de Chroma en disco durante test E2E.
        # Patch a retriever_agent (evitamos buscar realmente en Chroma y retornar un texto duro)
        with patch("agents.multiagents.retriever_agent") as mock_retriever:
            mock_retriever.return_value = {"search_results": "Resultados fake encontrados"}
            
            # Ejecutar invoke_docops con la pregunta fake
            result = invoke_docops(query="¿Cuál es tu cargo?", persona_name="Eduardo")
            
            # Aserciones: Debería llegar a completar un ciclo de QA.
            assert "answer" in result
            assert result["quality_score"] == 0.95
            assert result["iterations"] == 1
            assert result["interrupted"] is False
