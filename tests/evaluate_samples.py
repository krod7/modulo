"""Score every annotated sample flowchart and print a per-stage report.

Usage (from the repository root):
    python tests/evaluate_samples.py              # all samples
    python tests/evaluate_samples.py sample7 sample8
    python tests/evaluate_samples.py --json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flowchart_scoring import evaluate_sample, load_ground_truth  # noqa: E402


def _run(job: tuple[str, dict, str | None]) -> dict:
    name, spec, image_dir = job
    started = time.time()
    score, _ = evaluate_sample(name, spec, image_dir=Path(image_dir) if image_dir else None)
    result = dict(score.__dict__)
    result.update(passed=score.passed, structure_ok=score.structure_ok,
                  summary=score.summary(), seconds=round(time.time() - started, 1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ground-truth", help="alternate ground-truth JSON (e.g. generated charts)")
    parser.add_argument("--images", help="directory holding the ground-truth images")
    args = parser.parse_args()
    truth = load_ground_truth(Path(args.ground_truth) if args.ground_truth else None)
    images = args.images or (str(Path(args.ground_truth).parent) if args.ground_truth else None)
    order = (lambda item: int(item.replace("sample", ""))) if not args.ground_truth else (lambda item: item)
    names = args.names or sorted(truth, key=order)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run, [(name, truth[name], images) for name in names]))
    for result in results:
        verdict = "PASS" if result["passed"] else "FAIL"
        print(f"{result['name']:9} {verdict}  [{result['pipeline_status']}, {result['seconds']}s] {result['summary']}")
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} samples structurally correct")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
