from pathlib import Path

import numpy as np
import pytest

from mm_search_lab.config import load_yaml
from mm_search_lab.api import create_app
from mm_search_lab.data import CatalogItem, load_catalog
from mm_search_lab.encoder import OpenClipEncoder, l2_normalize
from mm_search_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, summarize_retrieval
from mm_search_lab.search import SearchService, format_results
from mm_search_lab.vector_store import build_flat_index, search_one


def test_baseline_config_loads():
    config = load_yaml("configs/baseline.yaml")
    assert config["project"]["name"] == "MM-SearchLab"
    assert config["paths"]["catalog"] == "data/raw/catalog.jsonl"


def test_catalog_item_text_for_embedding():
    item = CatalogItem(
        id="item_001",
        title="red running shoes",
        caption="a pair of red running shoes",
        genre="product",
        image_path=Path("data/raw/images/item_001.jpg"),
    )
    assert "red running shoes" in item.text_for_embedding
    assert "genre: product" in item.text_for_embedding


def test_project_skeleton_exists():
    assert Path("scripts/build_toy_data.py").exists()
    assert Path("scripts/build_embeddings.py").exists()
    assert Path("scripts/build_index.py").exists()
    assert Path("scripts/search_image.py").exists()
    assert Path("scripts/search_text.py").exists()
    assert Path("scripts/evaluate_baseline.py").exists()
    assert Path("src/mm_search_lab/api.py").exists()
    assert Path("src/mm_search_lab/logging_utils.py").exists()
    assert Path("src/mm_search_lab/metrics.py").exists()
    assert Path("src/mm_search_lab/search.py").exists()
    assert Path("docs/DAY6_DEMO.md").exists()
    assert Path("src/mm_search_lab").exists()


def test_load_catalog_checks_images():
    catalog = Path("data/raw/catalog.jsonl")
    if not catalog.exists():
        pytest.skip("toy catalog has not been generated yet")
    items = load_catalog(catalog)
    assert len(items) > 0
    assert items[0].image_path.exists()
    assert items[0].text_for_embedding


def test_encoder_import_without_model_download():
    assert OpenClipEncoder.__name__ == "OpenClipEncoder"
    embeddings = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normalized = l2_normalize(embeddings)
    assert normalized.shape == (2, 2)
    assert np.allclose(np.linalg.norm(normalized, axis=1), np.ones(2))


def test_faiss_vector_store_smoke():
    vectors = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    index = build_flat_index(vectors, metric="inner_product")
    results = search_one(index, vectors[0], ["a", "b", "c"], top_k=2)
    assert index.ntotal == 3
    assert results[0]["id"] == "a"
    assert results[0]["score"] == pytest.approx(1.0)


def test_day4_api_and_service_import_without_model_download():
    app = create_app()
    assert app.title == "MM-SearchLab"
    assert SearchService.__name__ == "SearchService"


def test_day5_metrics_smoke():
    results = [
        {"rank": 1, "id": "wrong", "score": 0.9},
        {"rank": 2, "id": "right", "score": 0.8},
    ]
    assert recall_at_k(results, ["right"], 1) == 0.0
    assert recall_at_k(results, ["right"], 2) == 1.0
    assert reciprocal_rank(results, ["right"]) == pytest.approx(0.5)
    assert ndcg_at_k(results, ["right"], 2) == pytest.approx(1.0 / np.log2(3))

    summary = summarize_retrieval(
        [
            {
                "results": results,
                "relevant_ids": ["right"],
                "latency_ms": 10.0,
            }
        ],
        ks=[1, 2],
    )
    assert summary["recall@1"] == 0.0
    assert summary["recall@2"] == 1.0
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["avg_ms"] == pytest.approx(10.0)


def test_day6_readable_result_format_contains_required_fields():
    text = format_results(
        [
            {
                "rank": 1,
                "id": "item_001",
                "score": 0.35433578,
                "genre": "product",
                "title": "red running shoes",
                "image_path": "/home/MM-SearchLab/data/raw/images/item_001.jpg",
            }
        ]
    )
    assert "rank" in text
    assert "id" in text
    assert "score" in text
    assert "genre" in text
    assert "title" in text
    assert "image_path" in text
    assert "item_001" in text
    assert "data/raw/images/item_001.jpg" in text
