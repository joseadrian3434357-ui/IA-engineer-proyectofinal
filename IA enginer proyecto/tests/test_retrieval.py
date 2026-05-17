import pytest
from unittest.mock import MagicMock

from rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from rag.ingestion import Chunk

@pytest.fixture
def sample_chunks():
    """Genera algunos chunks simulados."""
    return [
        Chunk(content="Experto en Python y React", chunk_id="1", metadata={"source": "cv"}),
        Chunk(content="Trabaja como CTO en Google", chunk_id="2", metadata={"source": "linkedin"}),
        Chunk(content="Receta de galletas de chocolate", chunk_id="3", metadata={"source": "blog"}),
    ]

@pytest.fixture
def mock_collection():
    """Mockea un Chroma Collection."""
    collection = MagicMock()
    # Mocking the semantic search query response
    collection.query.return_value = {
        "ids": [["1", "2", "3"]],
        "distances": [[0.1, 0.5, 0.9]],  # 0.1 is best here, inverted ranking usually? Assume distances returned ascending.
    }
    return collection

def test_hybrid_retriever_initialization(mock_collection, sample_chunks):
    """Verifica que el HybridRetriever se inicializa correctamente con BM25."""
    # Instanciamos retriever con alpha en 0.5 (Mitad Semántica, Mitad Palabra Clave)
    retriever = HybridRetriever(mock_collection, sample_chunks, alpha=0.5)
    
    # Comprobar que construyó el índice BM25
    assert retriever.bm25 is not None
    assert retriever.bm25.bm25 is not None
    assert retriever.bm25.bm25.corpus_size == 3

def test_reciprocal_rank_fusion():
    """Prueba el correcto ordenamiento de resultados (Fusión RRF)."""
    # rankings de dos métodos. clave=id, valor=score original (el RRF solo le importa la posición)
    results_a = [MagicMock(chunk_id="1"), MagicMock(chunk_id="2")]
    results_b = [MagicMock(chunk_id="2"), MagicMock(chunk_id="1")]
    
    results_a[0].chunk_id = "chunk_A"
    results_a[1].chunk_id = "chunk_B"
    
    results_b[0].chunk_id = "chunk_B"
    results_b[1].chunk_id = "chunk_A"

    # reciprocal_rank_fusion devuelve una list of tuple (chunk_id, score)
    score_list = reciprocal_rank_fusion([results_a, results_b], k=60)
    score_dict = dict(score_list)
    
    # Dado que A está primero en A y segundo en B: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01612 = 0.0325
    # B está segundo en A y primero en B: 1/(60+2) + 1/(60+1) = 0.0325
    # Deberian tener igual ranking
    assert abs(score_dict["chunk_A"] - score_dict["chunk_B"]) < 0.0001
