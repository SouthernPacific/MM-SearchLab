"""Catalog loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union


REQUIRED_FIELDS = {"id", "title", "caption", "genre", "image_path"}


@dataclass(frozen=True)
class CatalogItem:
    id: str
    title: str
    caption: str
    genre: str
    image_path: Path

    @property
    def text_for_embedding(self) -> str:
        return build_text_for_embedding(self.title, self.caption, self.genre)

    def to_metadata(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "caption": self.caption,
            "genre": self.genre,
            "image_path": str(self.image_path),
        }


def build_text_for_embedding(title: str, caption: str, genre: str) -> str:
    parts = [title.strip(), caption.strip(), "genre: {}".format(genre.strip())]
    return ". ".join(part for part in parts if part)


def _validate_row(row: dict, line_no: int) -> None:
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"catalog line {line_no} missing fields: {fields}")
    for field in REQUIRED_FIELDS:
        if row[field] in (None, ""):
            raise ValueError("catalog line {} has empty field: {}".format(line_no, field))


def _resolve_image_path(image_path: str, project_dir: Path) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _row_to_item(row: dict, line_no: int, project_dir: Path, check_images: bool) -> CatalogItem:
    _validate_row(row, line_no)
    image_path = _resolve_image_path(str(row["image_path"]), project_dir)
    if check_images and not image_path.exists():
        raise FileNotFoundError(
            "catalog line {} image_path does not exist: {}".format(line_no, image_path)
        )
    return CatalogItem(
        id=str(row["id"]),
        title=str(row["title"]),
        caption=str(row["caption"]),
        genre=str(row["genre"]),
        image_path=image_path,
    )


def load_catalog(
    path: Union[str, Path],
    project_dir: Optional[Union[str, Path]] = None,
    check_images: bool = True,
) -> List[CatalogItem]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"catalog file not found: {catalog_path}")

    root = Path(project_dir) if project_dir is not None else Path.cwd()
    root = root.resolve()

    items: List[CatalogItem] = []
    with catalog_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(_row_to_item(row, line_no, root, check_images))

    if not items:
        raise ValueError(f"catalog is empty: {catalog_path}")
    return items


def iter_texts(items: Iterable[CatalogItem]) -> Iterable[str]:
    for item in items:
        yield item.text_for_embedding


def image_paths(items: Iterable[CatalogItem]) -> List[Path]:
    return [item.image_path for item in items]


def item_ids(items: Iterable[CatalogItem]) -> List[str]:
    return [item.id for item in items]
