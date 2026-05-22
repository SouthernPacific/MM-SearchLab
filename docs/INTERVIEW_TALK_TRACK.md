# Interview Talk Track

## 30-Second Intro

MM-SearchLab is a Week 1 baseline for multimodal supply understanding in a search/recommendation scenario. I used OpenCLIP to map image and text fields into the same vector space, built a FAISS recall index, exposed text-to-image and image-to-image retrieval through CLI and FastAPI, and added offline evaluation with Recall@K, MRR, nDCG, and latency.

## Architecture

```text
catalog.jsonl
  -> OpenCLIP text/image encoder
  -> L2 normalized embeddings
  -> FAISS IndexFlatIP
  -> SearchService
  -> CLI / FastAPI / evaluation
```

## Input And Output

Input item:

```json
{"id":"item_001","title":"red running shoes","caption":"a pair of red running shoes on white background","genre":"product","image_path":"data/raw/images/item_001.jpg"}
```

Output result:

```json
{"id":"item_001","title":"red running shoes","genre":"product","rank":1,"score":0.3543,"image_path":"data/raw/images/item_001.jpg"}
```

## Core Implementation Points

- `data.py`: validates catalog schema and image paths.
- `encoder.py`: loads OpenCLIP, encodes text and images, and L2-normalizes embeddings.
- `vector_store.py`: wraps FAISS index build, save, load, and TopK search.
- `search.py`: provides a shared `SearchService` for CLI and API.
- `metrics.py`: implements Recall@K, MRR, nDCG@K, and latency summaries.
- `evaluate_baseline.py`: evaluates text-to-image and image-to-image retrieval.

## Why This Matches The JD

- Multimodal understanding: image, title, caption, and genre are represented in a shared embedding space.
- Search/recommendation landing: the model output is not just a classification label; it powers recall.
- Evaluation system: the project reports Recall@K, MRR, nDCG, and latency.
- AI-native evolution path: later experiments can automatically try prompt templates, index types, and reranking weights.

## Current Baseline Result

On the 12-item toy catalog:

- Text-to-image `recall@1 = 1.0`, `mrr = 1.0`
- Image-to-image `recall@1 = 1.0`, `mrr = 1.0`
- Overall `p95_ms = 201.6476` after warmup

This is a toy dataset, so I would not oversell the metric as production quality. Its value is proving the complete retrieval and evaluation loop.

## Likely Interview Questions

**Q: Why use inner product in FAISS?**  
A: The embeddings are L2-normalized, so inner product is equivalent to cosine similarity. That makes `IndexFlatIP` a simple baseline for semantic retrieval.

**Q: Why separate warmup latency?**  
A: The first query loads OpenCLIP weights and preprocessors, which is cold-start overhead. I report warmup separately so query latency reflects steady-state retrieval.

**Q: What would you improve next?**  
A: I would add more realistic data, hard negatives, genre-grouped metrics, prompt template experiments, and a lightweight reranker that combines CLIP score with supply metadata.

**Q: What is the project risk?**  
A: The toy data is small and synthetic. The engineering loop is complete, but real quality claims require a real multimodal dataset and human-labeled relevance.
