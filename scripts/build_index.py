#!/usr/bin/env python3
"""Build the Day 3 FAISS image index from Day 2 image embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mm_search_lab.config import load_yaml  # noqa: E402
from mm_search_lab.vector_store import build_flat_index, save_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    paths = config["paths"]
    retrieval = config.get("retrieval", {})
    metric = retrieval.get("index_metric", "inner_product")

    embedding_dir = ROOT / paths["embedding_dir"]
    index_dir = ROOT / paths["index_dir"]
    image_embeddings_path = embedding_dir / "image_embeddings.npy"
    item_ids_path = embedding_dir / "item_ids.json"

    if not image_embeddings_path.exists():
        raise FileNotFoundError("missing image embeddings: {}".format(image_embeddings_path))
    if not item_ids_path.exists():
        raise FileNotFoundError("missing item ids: {}".format(item_ids_path))

    image_embeddings = np.load(image_embeddings_path)
    with item_ids_path.open("r", encoding="utf-8") as f:
        item_ids = json.load(f)

    if image_embeddings.shape[0] != len(item_ids):
        raise ValueError(
            "embedding rows ({}) != item ids ({})".format(image_embeddings.shape[0], len(item_ids))
        )

    index = build_flat_index(image_embeddings, metric=metric)
    index_path = index_dir / "image.faiss"
    save_index(index, index_path)

    metadata = {
        "index_type": "IndexFlatIP" if metric == "inner_product" else "IndexFlatL2",
        "metric": metric,
        "num_items": len(item_ids),
        "dim": int(image_embeddings.shape[1]),
        "source_embeddings": str(image_embeddings_path.relative_to(ROOT)),
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "image_index_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("loaded image_embeddings shape: {}".format(image_embeddings.shape))
    print("built image index: {}".format(index_path))
    print("index ntotal: {}".format(index.ntotal))


if __name__ == "__main__":
    main()
