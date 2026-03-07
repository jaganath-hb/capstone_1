from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model(name: str = "all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def embed_texts(texts):
    model = get_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb
