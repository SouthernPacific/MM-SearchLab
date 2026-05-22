"""FastAPI app for Day 4 multimodal retrieval."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from mm_search_lab.search import SearchService


LOGGER = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None, project_dir: Optional[str] = None) -> FastAPI:
    config = config_path or os.environ.get("MM_SEARCH_CONFIG", "configs/baseline.yaml")
    root = Path(project_dir or os.environ.get("MM_SEARCH_ROOT", Path.cwd())).resolve()
    app = FastAPI(title="MM-SearchLab", version="0.1.0")
    state = {"service": None}

    def get_service() -> SearchService:
        if state["service"] is None:
            state["service"] = SearchService(config, root)
        return state["service"]

    @app.get("/health")
    def health() -> dict:
        try:
            return get_service().health()
        except Exception as exc:  # pragma: no cover - exercised manually in smoke test.
            LOGGER.exception("health check failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/search/text")
    def search_text(
        query: str = Query(..., min_length=1),
        top_k: int = Query(5, ge=1, le=50),
    ) -> dict:
        try:
            results = get_service().search_text(query, top_k=top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("text search failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"query": query, "top_k": top_k, "results": results}

    @app.get("/search/image")
    def search_image(
        image_path: str = Query(..., min_length=1),
        top_k: int = Query(5, ge=1, le=50),
    ) -> dict:
        try:
            results = get_service().search_image(image_path, top_k=top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("image search failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"image_path": image_path, "top_k": top_k, "results": results}

    return app


app = create_app()
