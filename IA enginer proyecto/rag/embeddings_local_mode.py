from openai import OpenAI
import numpy as np
import os
# Cargar modelo multilingüe (primera vez descarga ~500MB)
client = OpenAI(
    base_url="http://192.168.1.69:1234/v1/embeddings",
    api_key="lm-studio",
)

model = "text-embeddings-sentence-transformers_all-minilm-l12-v2"

def remote_embedding(text):
    # Point to the remote
    client = OpenAI(base_url=os.environ.get("URL_"), api_key="not_needed")
    # Get embeddings
    texts_a_subir = []
    response = client.embeddings.create(
        model="text-embeddings-sentence-transformers_all-minilm-l12-v2",
        input=text
    )
    return response.data[0].embedding

def remotoe_batch (texts):
    client = OpenAI(base_url=os.envuroment.get , api_key="not_needed")
    texts_a_subir = []
    for t in texts:
        response = client.embeddings.create(
        model="text-embeddings-sentence-transformers_all-minilm-l12-v2",
        input=t
        )
        texts_a_subir.append(response.data[0].embedding)
    return texts_a_subir

