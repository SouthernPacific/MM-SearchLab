#!/usr/bin/env python3
"""Build text and image embeddings for the baseline catalog."""

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
from mm_search_lab.data import image_paths, item_ids, iter_texts, load_catalog  # noqa: E402
from mm_search_lab.encoder import OpenClipEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)

    paths = config["paths"]
    model_config = config["model"]
    retrieval_config = config.get("retrieval", {})

    catalog_path = ROOT / paths["catalog"]
    embedding_dir = ROOT / paths["embedding_dir"]
    embedding_dir.mkdir(parents=True, exist_ok=True)

    items = load_catalog(catalog_path, project_dir=ROOT, check_images=True)
    texts = list(iter_texts(items))
    images = image_paths(items)

    encoder = OpenClipEncoder(
        model_name=model_config.get("name", "ViT-B-32"),
        pretrained=model_config.get("pretrained", "openai"),
        device=model_config.get("device", "cpu"),
        batch_size=int(model_config.get("batch_size", 8)),
        normalize=bool(retrieval_config.get("normalize_embeddings", True)),
    )

    text_embeddings = encoder.encode_texts(texts)
    image_embeddings = encoder.encode_images(images)

    np.save(embedding_dir / "text_embeddings.npy", text_embeddings)
    np.save(embedding_dir / "image_embeddings.npy", image_embeddings)
    with (embedding_dir / "item_ids.json").open("w", encoding="utf-8") as f:
        json.dump(item_ids(items), f, ensure_ascii=False, indent=2)

    print("loaded items: {}".format(len(items)))
    print("text_embeddings shape: {}".format(text_embeddings.shape))
    print("image_embeddings shape: {}".format(image_embeddings.shape))
    print("wrote embeddings to {}".format(embedding_dir))


if __name__ == "__main__":
    main()
