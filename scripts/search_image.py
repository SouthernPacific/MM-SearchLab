#!/usr/bin/env python3
"""Image-to-image retrieval CLI for Day 4."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mm_search_lab.logging_utils import setup_logging  # noqa: E402
from mm_search_lab.search import SearchService, format_results  # noqa: E402


LOGGER = logging.getLogger("mm_search_lab.search_image")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print JSON lines instead of a table")
    parser.add_argument("--verbose", action="store_true", help="enable INFO logs")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    try:
        service = SearchService(args.config, ROOT)
        results = service.search_image(args.image_path, top_k=args.top_k)
    except Exception as exc:
        if args.verbose:
            LOGGER.exception("image search failed")
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)

    if args.json:
        for row in results:
            print(json.dumps(row, ensure_ascii=False))
        return

    print("image_path: {}".format(args.image_path))
    print("top_k: {}".format(args.top_k))
    print(format_results(results))


if __name__ == "__main__":
    main()
