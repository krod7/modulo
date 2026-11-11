"""The "Try a Sample" rotation serves only verified flowcharts, in manifest order.

The fast tests check the manifest and the API. The end-to-end check runs every
rotation chart through analysis and code generation; it needs Tesseract, so it
is gated like the other pipeline tests:
    FLOWCHART_RUN_PIPELINE=1 python -m pytest tests/test_sample_rotation.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend import pipeline
from backend.app import ACCEPTED_DIR, SAMPLE_ROTATION, create_app
from backend.config import Settings
from backend.models import AnalysisResponse

RUN = os.getenv("FLOWCHART_RUN_PIPELINE") == "1"
ENTRIES = json.loads(SAMPLE_ROTATION.read_text(encoding="utf-8"))["samples"]


def test_rotation_lists_twelve_distinct_existing_charts():
    paths = [entry["path"] for entry in ENTRIES]
    assert len(paths) == 12 and len(set(paths)) == 12
    allowed = set(Settings().ALLOWED_EXTENSIONS)
    for relative in paths:
        path = ACCEPTED_DIR / relative
        assert path.is_file(), relative
        assert path.suffix.lower().lstrip(".") in allowed, relative


def test_api_serves_rotation_in_manifest_order():
    client = create_app().test_client()
    samples = client.get("/api/samples").get_json()["samples"]
    assert [Path(item["url"]).name for item in samples] == [Path(entry["path"]).name for entry in ENTRIES]
    for item in samples:
        assert item["under_limit"]
        response = client.get(item["url"])
        assert response.status_code == 200 and len(response.data) == item["bytes"]


def test_accepted_route_does_not_escape_its_folder():
    client = create_app().test_client()
    assert client.get("/accepted/../backend/app.py").status_code == 404


@pytest.mark.skipif(not RUN, reason="set FLOWCHART_RUN_PIPELINE=1 for OCR-dependent rotation check")
@pytest.mark.parametrize("relative", [entry["path"] for entry in ENTRIES])
def test_rotation_chart_generates_python(relative, record_property):
    path = ACCEPTED_DIR / relative
    context = pipeline.analyze(path.read_bytes(), path.name)
    response = AnalysisResponse(context).to_dict()
    record_property("recognition_score", response["recognition_score"]["overall"])
    errors = [item.code for item in context.diagnostics if item.severity == "error"]
    assert response["status"] == "ready", f"{relative}: {errors}"
    # Verified charts must not be graded below Moderate.
    assert response["recognition_score"]["overall"] >= 70, response["recognition_score"]
    context.approved = True
    context = pipeline.generate(context)
    assert not context.has_errors, [item.code for item in context.diagnostics if item.severity == "error"]
    code = "\n".join("    " * line.indent + line.text for line in context.code)
    compile(code, relative, "exec")
    assert "def main():" in code
