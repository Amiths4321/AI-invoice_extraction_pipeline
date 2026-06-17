#!/usr/bin/env python3
"""
CLI for the invoice extraction pipeline.

Examples:
    # Single invoice, text file -> prints JSON to stdout
    python cli.py extract sample_invoices/sample_invoice_1.txt

    # Single invoice, write result to a file
    python cli.py extract sample_invoices/sample_invoice_1.txt -o result.json

    # Batch: every .txt file in a directory -> one .json per invoice
    python cli.py batch sample_invoices/ -o output_dir/
"""
import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config
from extractor import extract_invoice, ExtractionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("invoice_pipeline.cli")


def cmd_extract(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    raw_text = input_path.read_text(encoding="utf-8")

    try:
        result = extract_invoice(raw_text)
    except ExtractionError as e:
        logger.error("Extraction failed for %s: %s", input_path.name, e)
        return 1

    output_json = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        logger.info("Wrote result to %s", args.output)
    else:
        print(output_json)

    return 0


def _process_one(path: Path, output_dir: Path) -> tuple[str, bool, str]:
    try:
        raw_text = path.read_text(encoding="utf-8")
        result = extract_invoice(raw_text)
        out_path = output_dir / f"{path.stem}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return (path.name, True, str(out_path))
    except Exception as e:
        return (path.name, False, str(e))


def cmd_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        logger.warning("No .txt files found in %s", input_dir)
        return 0

    logger.info("Processing %d invoice(s) with up to %d workers...", len(files), config.BATCH_MAX_WORKERS)

    successes, failures = 0, 0
    with ThreadPoolExecutor(max_workers=config.BATCH_MAX_WORKERS) as pool:
        futures = {pool.submit(_process_one, f, output_dir): f for f in files}
        for future in as_completed(futures):
            name, ok, detail = future.result()
            if ok:
                successes += 1
                logger.info("OK   %-40s -> %s", name, detail)
            else:
                failures += 1
                logger.error("FAIL %-40s -> %s", name, detail)

    logger.info("Done. %d succeeded, %d failed.", successes, failures)
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Invoice extraction pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract a single invoice text file")
    p_extract.add_argument("input_file", help="Path to a .txt file containing raw invoice text")
    p_extract.add_argument("-o", "--output", help="Path to write the JSON result (default: stdout)")
    p_extract.set_defaults(func=cmd_extract)

    p_batch = sub.add_parser("batch", help="Extract every .txt file in a directory")
    p_batch.add_argument("input_dir", help="Directory containing .txt invoice files")
    p_batch.add_argument("-o", "--output", required=True, help="Directory to write .json results to")
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
