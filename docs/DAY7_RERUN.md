# Day 7 Full Rerun

Day 7 verifies that the Week 1 baseline can be rebuilt from generated-output cleanup, then documents the final demo-ready state.

## Environment

- Path: `/home/MM-SearchLab`
- Runtime: Ubuntu VM
- Device: CPU
- Model: `ViT-B-32`
- Pretrained weights: `openai`
- Vector index: FAISS `IndexFlatIP`

## Rerun Commands

```bash
cd /home/MM-SearchLab
source .venv/bin/activate

rm -rf artifacts reports/baseline
mkdir -p artifacts/embeddings artifacts/index reports/baseline

python scripts/build_toy_data.py
python scripts/build_embeddings.py --config configs/baseline.yaml
python scripts/build_index.py --config configs/baseline.yaml
python scripts/search_text.py --query "red running shoes" --top-k 3
python scripts/search_image.py --image-path data/raw/images/item_001.jpg --top-k 3
python scripts/evaluate_baseline.py --config configs/baseline.yaml --ks 1,5
pytest -q
```

## Rerun Result

```text
wrote 12 items to data/raw/catalog.jsonl
text_embeddings shape: (12, 512)
image_embeddings shape: (12, 512)
index ntotal: 12
```

Text query:

```text
query: red running shoes
rank  id        score   genre    title
1     item_001  0.3543  product  red running shoes
```

Image query:

```text
image_path: data/raw/images/item_001.jpg
rank  id        score   genre    title
1     item_001  1.0000  product  red running shoes
```

Smoke tests:

```text
9 passed
```

## Baseline Metrics

The evaluation uses each catalog item as a query and treats its own `id` as the relevant target.

```text
text_to_image:
  recall@1 = 1.0
  recall@5 = 1.0
  mrr = 1.0
  avg_ms = 122.7463
  p95_ms = 135.2931

image_to_image:
  recall@1 = 1.0
  recall@5 = 1.0
  mrr = 1.0
  avg_ms = 200.3786
  p95_ms = 209.0978

overall:
  recall@1 = 1.0
  recall@5 = 1.0
  mrr = 1.0
  avg_ms = 161.5625
  p95_ms = 201.6476
```

`warmup_ms` is recorded separately because the first request includes OpenCLIP model loading.

## Artifacts

Generated locally and intentionally ignored by Git:

- `artifacts/embeddings/text_embeddings.npy`
- `artifacts/embeddings/image_embeddings.npy`
- `artifacts/embeddings/item_ids.json`
- `artifacts/index/image.faiss`
- `artifacts/index/image_index_meta.json`
- `reports/baseline/metrics.json`
- `reports/baseline/per_query_results.jsonl`
