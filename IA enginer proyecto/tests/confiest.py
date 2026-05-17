import hashlib
import uuid

import chromadb
import numpy as np
import pytest


@pytest.fixture
def chroma_collection():
    client = chromadb.Client()
    collection_name = f"test_{uuid.uuid4().hex[:12]}"
    collection = client.create_collection(name=collection_name)
    yield collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


@pytest.fixture
def dummy_embed_fn():
   
    def _embed(text: str) -> list[float]:
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = hash_bytes * (384 // len(hash_bytes) + 1)
        raw = np.array(
            [b / 255.0 for b in repeated[:384]], dtype=np.float32
        )
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        return raw.tolist()

    return _embed


@pytest.fixture
def sample_docs():
    import json
    from pathlib import Path
    
    dataset_path = Path(__file__).resolve().parent.parent / "evals" / "golden" / "eval.dataset.json"
    docs = []
    
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            for i, item in enumerate(dataset):
                docs.append({
                    "id": f"golden_doc_{i+1:03d}",
                    "content": f"{item['question']} {item['golden_answer']}"
                })
    
    return docs[:5]