from typing import Tuple
import numpy as np
import faiss


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def build_ip_index(vectors: np.ndarray, normalize: bool = True) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    if normalize:
        vectors = l2_normalize(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index, vectors


def search_index(index: faiss.IndexFlatIP, queries: np.ndarray, top_k: int = 3, normalize: bool = True):
    if normalize:
        queries = l2_normalize(queries)
    scores, idx = index.search(queries.astype(np.float32), top_k)
    return scores, idx

