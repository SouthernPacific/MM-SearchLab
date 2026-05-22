"""FAISS vector index helpers for MM-SearchLab."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import faiss
import numpy as np


PathLike = Union[str, Path]


def ensure_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("vectors must be a 2D array, got shape {}".format(array.shape))
    if array.shape[0] == 0:
        raise ValueError("vectors must contain at least one row")
    return np.ascontiguousarray(array)


def build_flat_index(vectors: np.ndarray, metric: str = "inner_product") -> faiss.Index:
    matrix = ensure_float32_matrix(vectors)
    dim = matrix.shape[1]
    if metric == "inner_product":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("unsupported metric: {}".format(metric))
    index.add(matrix)
    return index


def save_index(index: faiss.Index, path: PathLike) -> None:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_index(path: PathLike) -> faiss.Index:
    index_path = Path(path)
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found: {}".format(index_path))
    return faiss.read_index(str(index_path))


def search_index(index: faiss.Index, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    queries = ensure_float32_matrix(query_vectors)
    scores, indices = index.search(queries, top_k)
    return scores, indices


def search_one(index: faiss.Index, query_vector: np.ndarray, item_ids: Sequence[str], top_k: int) -> List[dict]:
    scores, indices = search_index(index, query_vector.reshape(1, -1), top_k)
    results: List[dict] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        results.append({"rank": rank, "id": item_ids[int(idx)], "score": float(score)})
    return results
