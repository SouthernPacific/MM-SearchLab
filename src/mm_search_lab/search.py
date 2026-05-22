"""Shared search service for CLI and FastAPI endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from mm_search_lab.config import load_yaml
from mm_search_lab.data import load_catalog
from mm_search_lab.encoder import OpenClipEncoder
from mm_search_lab.vector_store import load_index, search_one


PathLike = Union[str, Path]
LOGGER = logging.getLogger(__name__)


class SearchService:
    """Lazy-loading retrieval service over the Day 3 image FAISS index."""

    def __init__(self, config_path: PathLike, project_dir: PathLike) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.config_path = _resolve_path(config_path, self.project_dir)
        self.config = load_yaml(self.config_path)
        self.paths = self.config["paths"]
        self.model_config = self.config["model"]
        self.retrieval_config = self.config.get("retrieval", {})

        self.catalog = load_catalog(
            self.project_dir / self.paths["catalog"],
            project_dir=self.project_dir,
            check_images=True,
        )
        self.metadata_by_id: Dict[str, dict] = {item.id: item.to_metadata() for item in self.catalog}
        self.item_ids = self._load_item_ids()
        self.index = load_index(self.index_path)
        self._encoder: Optional[OpenClipEncoder] = None
        if self.index.ntotal != len(self.item_ids):
            raise ValueError(
                "index size ({}) does not match item ids ({})".format(self.index.ntotal, len(self.item_ids))
            )
        LOGGER.info("loaded search service: items=%s index=%s", len(self.item_ids), self.index_path)

    @property
    def embedding_dir(self) -> Path:
        return self.project_dir / self.paths["embedding_dir"]

    @property
    def index_dir(self) -> Path:
        return self.project_dir / self.paths["index_dir"]

    @property
    def index_path(self) -> Path:
        return self.index_dir / "image.faiss"

    @property
    def encoder(self) -> OpenClipEncoder:
        if self._encoder is None:
            LOGGER.info(
                "loading OpenCLIP encoder: model=%s pretrained=%s device=%s",
                self.model_config.get("name", "ViT-B-32"),
                self.model_config.get("pretrained", "openai"),
                self.model_config.get("device", "cpu"),
            )
            self._encoder = OpenClipEncoder(
                model_name=self.model_config.get("name", "ViT-B-32"),
                pretrained=self.model_config.get("pretrained", "openai"),
                device=self.model_config.get("device", "cpu"),
                batch_size=int(self.model_config.get("batch_size", 8)),
                normalize=bool(self.retrieval_config.get("normalize_embeddings", True)),
            )
        return self._encoder

    def health(self) -> dict:
        return {
            "status": "ok",
            "num_items": len(self.item_ids),
            "index_path": str(self.index_path),
            "index_ntotal": int(self.index.ntotal),
            "model": self.model_config.get("name", "ViT-B-32"),
            "pretrained": self.model_config.get("pretrained", "openai"),
            "device": self.model_config.get("device", "cpu"),
        }

    def search_text(self, query: str, top_k: int) -> List[dict]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        top_k = self._normalize_top_k(top_k)
        LOGGER.info("text search: query=%r top_k=%s", query, top_k)
        query_embedding = self.encoder.encode_texts([query])
        return self._search(query_embedding[0], top_k)

    def search_image(self, image_path: PathLike, top_k: int) -> List[dict]:
        top_k = self._normalize_top_k(top_k)
        path = _resolve_path(image_path, self.project_dir)
        if not path.exists():
            raise FileNotFoundError("image file not found: {}".format(path))
        if not path.is_file():
            raise ValueError("image_path must be a file: {}".format(path))
        LOGGER.info("image search: image_path=%s top_k=%s", path, top_k)
        query_embedding = self.encoder.encode_images([path])
        return self._search(query_embedding[0], top_k)

    def _search(self, query_vector, top_k: int) -> List[dict]:
        results = search_one(self.index, query_vector, self.item_ids, top_k=top_k)
        enriched: List[dict] = []
        for result in results:
            row = self.metadata_by_id.get(result["id"], {}).copy()
            row.update(result)
            enriched.append(row)
        return enriched

    def _normalize_top_k(self, top_k: int) -> int:
        try:
            value = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be an integer") from exc
        if value <= 0:
            raise ValueError("top_k must be positive")
        if value > len(self.item_ids):
            LOGGER.warning("top_k=%s exceeds catalog size=%s; capping", value, len(self.item_ids))
            value = len(self.item_ids)
        return value

    def _load_item_ids(self) -> List[str]:
        item_ids_path = self.embedding_dir / "item_ids.json"
        if not item_ids_path.exists():
            raise FileNotFoundError("missing item ids: {}".format(item_ids_path))
        with item_ids_path.open("r", encoding="utf-8") as f:
            item_ids = json.load(f)
        if len(item_ids) != len(self.catalog):
            raise ValueError("item_ids length does not match catalog length")
        return item_ids


def _resolve_path(path: PathLike, project_dir: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_dir / resolved
    return resolved.resolve()


def format_results(results: Sequence[dict]) -> str:
    if not results:
        return "No results."

    headers = ["rank", "id", "score", "genre", "title", "image_path"]
    rows = []
    for row in results:
        rows.append(
            [
                str(row.get("rank", "")),
                str(row.get("id", "")),
                "{:.4f}".format(float(row.get("score", 0.0))),
                str(row.get("genre", "")),
                str(row.get("title", "")),
                _display_path(row.get("image_path", "")),
            ]
        )

    widths = []
    for col_idx, header in enumerate(headers):
        widths.append(max(len(header), max(len(row[col_idx]) for row in rows)))

    def render(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def _display_path(value: object) -> str:
    text = str(value)
    marker = "/home/MM-SearchLab/"
    if text.startswith(marker):
        return text[len(marker) :]
    return text
