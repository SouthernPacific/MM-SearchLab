#!/usr/bin/env python3
"""Build a tiny local catalog for Day 1 smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw


ITEMS = [
    ("item_001", "red running shoes", "a pair of red running shoes on white background", "product", (220, 40, 50)),
    ("item_002", "blue hiking backpack", "a blue backpack for outdoor travel", "product", (40, 90, 210)),
    ("item_003", "green mountain lake", "a green mountain landscape beside a quiet lake", "travel", (40, 160, 90)),
    ("item_004", "yellow city taxi", "a yellow taxi driving through a city street", "travel", (230, 190, 40)),
    ("item_005", "black wireless headphones", "black headphones for music and gaming", "product", (35, 35, 35)),
    ("item_006", "white coffee mug", "a white mug filled with hot coffee", "food", (235, 235, 220)),
    ("item_007", "orange basketball", "an orange basketball on a court", "sport", (230, 120, 35)),
    ("item_008", "purple flower bouquet", "a bouquet of purple flowers in a vase", "lifestyle", (150, 80, 180)),
    ("item_009", "silver laptop computer", "a silver laptop open on a desk", "product", (170, 180, 190)),
    ("item_010", "pink dessert cake", "a pink strawberry dessert cake on a plate", "food", (235, 120, 155)),
    ("item_011", "brown leather wallet", "a brown leather wallet for daily use", "product", (130, 80, 45)),
    ("item_012", "cyan swimming pool", "a cyan swimming pool at a summer resort", "travel", (40, 190, 210)),
]


def build_image(path: Path, title: str, color: Tuple[int, int, int], image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_size, image_size), color=color)
    draw = ImageDraw.Draw(image)
    label = title[:22]
    draw.rectangle((12, image_size - 48, image_size - 12, image_size - 12), fill=(255, 255, 255))
    draw.text((18, image_size - 38), label, fill=(20, 20, 20))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/raw/catalog.jsonl")
    parser.add_argument("--image-dir", default="data/raw/images")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    image_dir = Path(args.image_dir)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item_id, title, caption, genre, color in ITEMS:
        image_path = image_dir / f"{item_id}.jpg"
        build_image(image_path, title, color, args.image_size)
        rows.append(
            {
                "id": item_id,
                "title": title,
                "caption": caption,
                "genre": genre,
                "image_path": str(image_path),
            }
        )

    with catalog_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} items to {catalog_path}")
    print(f"wrote images to {image_dir}")


if __name__ == "__main__":
    main()
