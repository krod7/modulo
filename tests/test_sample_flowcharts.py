"""End-to-end structural regression tests for the ten sample flowcharts.

Each sample is scored against hand-annotated ground truth
(tests/fixtures/samples_ground_truth.json): every node must be found with the
right shape type and readable text, every directed edge and branch label must
match, and there may be no extra nodes or edges. Charts written in Modulo's
executable notation must additionally reach pseudocode. Producing output
without errors is not enough to pass.

OCR-dependent and slow (~2 minutes), so gated like the other pipeline tests:
    FLOWCHART_RUN_PIPELINE=1 python -m pytest tests/test_sample_flowcharts.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from flowchart_scoring import evaluate_sample, load_ground_truth  # noqa: E402

RUN = os.getenv("FLOWCHART_RUN_PIPELINE") == "1"
TRUTH = load_ground_truth()


@pytest.mark.skipif(not RUN, reason="set FLOWCHART_RUN_PIPELINE=1 for OCR-dependent sample regression")
@pytest.mark.parametrize("name", sorted(TRUTH, key=lambda item: int(item.replace("sample", ""))))
def test_sample_structure_matches_ground_truth(name, record_property):
    score, _ = evaluate_sample(name, TRUTH[name])
    record_property("summary", score.summary())
    assert score.structure_ok, f"{name}: {score.summary()}"
    assert not score.text_errors, f"{name}: {score.summary()}"
    if score.notation_supported:
        assert "pseudocode" in score.stages, f"{name}: logic recovery did not complete ({score.errors})"


@pytest.mark.skipif(not RUN, reason="set FLOWCHART_RUN_PIPELINE=1 for OCR-dependent generalisation check")
def test_generated_unseen_charts_have_correct_graphs(tmp_path):
    """Charts in styles never used for tuning: every graph must be exact.

    Node text is reported but not asserted here; one font's "i = i + 1" is a
    known Tesseract misreading (see docs/RECOGNITION_FIXES.md).
    """
    from generate_unseen_flowcharts import main as generate

    generate(str(tmp_path))
    truth = load_ground_truth(tmp_path / "ground_truth.json")
    failures = []
    for name, spec in truth.items():
        score, _ = evaluate_sample(name, spec, image_dir=tmp_path)
        if not score.structure_ok:
            failures.append(f"{name}: {score.summary()}")
    assert not failures, "\n".join(failures)
