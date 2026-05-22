# Day 6 Demo Notes

Day 6 focuses on demo readiness: readable results, basic logs, and predictable error handling.

## Start From The VM

```bash
cd /home/MM-SearchLab
source .venv/bin/activate
```

## CLI Demo

Text-to-image search:

```bash
python scripts/search_text.py --config configs/baseline.yaml --query "red running shoes" --top-k 3
```

Expected shape of the output:

```text
query: red running shoes
top_k: 3
rank  id        score   genre    title              image_path
----  --------  ------  -------  -----------------  ----------------------------
1     item_001  0.3543  product  red running shoes  data/raw/images/item_001.jpg
```

Image-to-image search:

```bash
python scripts/search_image.py --config configs/baseline.yaml --image-path data/raw/images/item_001.jpg --top-k 3
```

Expected behavior: `item_001` appears at rank 1 with score close to `1.0000`.

Enable logs when debugging:

```bash
python scripts/search_text.py --query "red running shoes" --top-k 3 --verbose
```

Error example:

```bash
python scripts/search_image.py --image-path data/raw/images/missing.jpg --top-k 3
```

Expected behavior: the command exits with code `2` and prints a clear `error:` message.

## API Demo

Start the API:

```bash
PYTHONPATH=src python -m uvicorn mm_search_lab.api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl -s "http://127.0.0.1:8000/health"
```

Text search:

```bash
curl -s "http://127.0.0.1:8000/search/text?query=red%20running%20shoes&top_k=3"
```

Image search:

```bash
curl -s "http://127.0.0.1:8000/search/image?image_path=data/raw/images/item_001.jpg&top_k=3"
```

Each result contains:

- `id`
- `title`
- `genre`
- `score`
- `image_path`

## Day 6 Acceptance

```bash
pytest -q
python scripts/search_text.py --query "red running shoes" --top-k 3
python scripts/search_image.py --image-path data/raw/images/item_001.jpg --top-k 3
```

The text query should return `item_001` in the top results. The image query should return `item_001` at rank 1.
