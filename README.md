# MM-SearchLab

面向内容供给理解的多模态搜推 baseline MVP。

本项目使用预训练 CLIP 模型将图片、标题、caption 映射到统一向量空间，基于 FAISS 实现文本搜图、图片搜相似内容，并用 Recall@K、MRR 和延迟指标评估 baseline 效果。

## MVP Scope

- 读取 `catalog.jsonl` 内容供给表
- 抽取 image/text embeddings
- 构建 FAISS 向量索引
- 支持 text-to-image retrieval
- 支持 image-to-image retrieval
- 输出 baseline evaluation report

暂不包含模型训练、在线推荐服务、大规模分布式索引和 Agent 自进化。第一周目标是先跑通 baseline 闭环。

## Data Schema

每行一个 JSON：

```json
{"id":"item_001","title":"red running shoes","caption":"a pair of red running shoes on white background","genre":"product","image_path":"data/raw/images/item_001.jpg"}
```

字段说明：

- `id`: 内容供给唯一 ID
- `title`: 标题或短文本
- `caption`: 图片描述或长文本
- `genre`: 内容体裁，例如 `product`、`food`、`travel`
- `image_path`: 相对项目根目录的图片路径

## Environment

推荐在 Ubuntu 虚拟机中运行：

```bash
cd /home/MM-SearchLab
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Quick Start

```bash
source .venv/bin/activate

python scripts/build_toy_data.py
python scripts/build_embeddings.py --config configs/baseline.yaml
python scripts/build_index.py --config configs/baseline.yaml
python scripts/search_text.py --query "red running shoes" --top-k 5
python scripts/evaluate_baseline.py --config configs/baseline.yaml
```

API demo:

```bash
PYTHONPATH=src python -m uvicorn mm_search_lab.api:app --host 127.0.0.1 --port 8000
curl -s "http://127.0.0.1:8000/health"
curl -s "http://127.0.0.1:8000/search/text?query=red%20running%20shoes&top_k=3"
```

Smoke test:

```bash
python scripts/build_toy_data.py
pytest -q
```

## Baseline Outputs

- `data/raw/catalog.jsonl`
- `artifacts/embeddings/image_embeddings.npy`
- `artifacts/embeddings/text_embeddings.npy`
- `artifacts/index/image.faiss`
- `reports/baseline/metrics.json`

## Week 1 Goal

跑通从内容供给数据到多模态向量检索的最小闭环：

```text
catalog.jsonl -> CLIP embeddings -> FAISS index -> TopK retrieval -> baseline metrics
```

## Interview Focus

这个 baseline 对应搜推链路中的召回层：输入多模态内容供给，输出可检索的统一表征，并用离线指标评估召回质量。

## Week 1 Baseline Status

Day 7 full rerun passed on Ubuntu VM CPU:

- Toy catalog: 12 items
- Embedding shape: text `(12, 512)`, image `(12, 512)`
- FAISS index: `IndexFlatIP`, `ntotal=12`
- Text query `"red running shoes"`: Top1 `item_001`
- Image query `data/raw/images/item_001.jpg`: Top1 `item_001`, score `1.0000`
- Smoke tests: `9 passed`

Latest baseline metrics after warmup:

- `text_to_image recall@1 = 1.0`, `mrr = 1.0`, `avg_ms = 122.7463`
- `image_to_image recall@1 = 1.0`, `mrr = 1.0`, `avg_ms = 200.3786`
- `overall recall@1 = 1.0`, `mrr = 1.0`, `p95_ms = 201.6476`

See:

- `docs/DAY6_DEMO.md`
- `docs/DAY7_RERUN.md`
- `docs/INTERVIEW_TALK_TRACK.md`
