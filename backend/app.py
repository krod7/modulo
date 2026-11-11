"""Local Flask API and static frontend; recognition belongs to services."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
from threading import BoundedSemaphore
import os
import sys

# Support the README's `python backend/app.py` as well as module/pytest imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from backend.analysis_store import AnalysisStore
from backend.config import Settings
from backend.models import AnalysisResponse
from backend import pipeline
from backend.services.intake_service import UploadError, inspect_upload

ROOT = Path(__file__).resolve().parents[1]
# Verified charts and the ordered "Try a Sample" rotation (see accepted_flowcharts/README.md).
ACCEPTED_DIR = ROOT / "accepted_flowcharts"
SAMPLE_ROTATION = ACCEPTED_DIR / "rotation.json"


def sample_rotation(allowed: set[str]) -> list[tuple[str, Path]]:
    """Return (URL path, file) pairs for the sample rotation, in display order.

    The rotation manifest lists verified charts relative to ``accepted_flowcharts/``.
    Entries that are missing, unsupported, or outside that folder are skipped. If the
    manifest is absent or unreadable, every supported file in ``samples/`` is used,
    which was the original behaviour.
    """
    try:
        entries = json.loads(SAMPLE_ROTATION.read_text(encoding="utf-8"))["samples"]
    except (OSError, ValueError, KeyError, TypeError):
        entries = None
    rotation = []
    if isinstance(entries, list):
        base = ACCEPTED_DIR.resolve()
        for entry in entries:
            relative = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(relative, str):
                continue
            path = (ACCEPTED_DIR / relative).resolve()
            if (base in path.parents and path.is_file()
                    and path.suffix.lower().lstrip(".") in allowed):
                rotation.append((f"/accepted/{path.relative_to(base).as_posix()}", path))
    if rotation:
        return rotation
    legacy = sorted(
        (path for path in (ROOT / "samples").iterdir()
         if path.is_file() and path.suffix.lower().lstrip(".") in allowed),
        key=lambda path: (path.stem.rstrip('0123456789'), int(''.join(ch for ch in path.stem if ch.isdigit()) or 0), path.suffix.lower()),
    )
    return [(f"/samples/{path.name}", path) for path in legacy]



def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings()
    app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
    app.config.update(MAX_CONTENT_LENGTH=settings.max_upload_bytes + 1024 * 1024,
                      MAX_FORM_MEMORY_SIZE=settings.max_upload_bytes + 1024 * 1024,
                      MAX_FORM_PARTS=10)
    store = AnalysisStore(settings)
    app.extensions["analysis_store"] = store
    capacity = BoundedSemaphore(2)

    def error(code: str, message: str, status: int):
        return jsonify(error=code, message=message), status

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error("FILE_TOO_LARGE", f"Choose a file no larger than {settings.MAX_UPLOAD_MB} MB.", 413)

    @app.errorhandler(BadRequest)
    def malformed(_):
        return error("INVALID_REQUEST", "The request is malformed. Select the file again and retry.", 400)

    @app.get("/")
    def index():
        return send_from_directory(ROOT / "frontend", "index.html")

    @app.get("/manus-routes.json")
    def routes():
        return jsonify(routes=[{"path": "/", "title": "Modulo"}])

    @app.get("/samples/<path:filename>")
    def sample(filename):
        return send_from_directory(ROOT / "samples", filename)

    @app.get("/accepted/<path:filename>")
    def accepted_sample(filename):
        return send_from_directory(ACCEPTED_DIR, filename)

    @app.get("/api/samples")
    def samples():
        entries = sample_rotation(set(settings.ALLOWED_EXTENSIONS))
        return jsonify(samples=[{"name": path.name, "url": url,
                                 "bytes": path.stat().st_size,
                                 "under_limit": path.stat().st_size <= settings.max_upload_bytes}
                                for url, path in entries])

    @app.post("/api/analyze")
    def analyze():
        # Prevent cross-origin browser form posts from starting local OCR work.
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return error("ORIGIN_NOT_ALLOWED", "Analyze from the local Modulo page.", 403)
        files = request.files.getlist("file")
        if len(files) != 1 or not files[0].filename:
            return error("FILE_REQUIRED", "Choose exactly one flowchart file.", 400)
        upload = files[0]
        data = upload.stream.read(settings.max_upload_bytes + 1)
        try:
            inspect_upload(data, upload.filename, settings)
        except UploadError as problem:
            return error(problem.code, problem.message, problem.status)
        if not capacity.acquire(blocking=False):
            return error("SERVER_BUSY", "Two analyses are already running. Wait for one to finish and retry.", 503)
        try:
            context = pipeline.analyze(data, upload.filename, settings)
            if context.has_errors and context.completed_stages == ["intake"] and context.page_image is None:
                diagnostic = context.diagnostics[0]
                return error(diagnostic.code, diagnostic.message, 413 if diagnostic.code == "IMAGE_TOO_LARGE" else 400)
            # Retain only the page, geometry and diagnostics needed for review.
            context.uploaded_bytes = b""
            context.binary = context.grayscale = context.stroke_mask = None
            identifier = expires = None
            if context.page_image is not None:
                success, encoded = cv2.imencode(".png", context.page_image)
                if not success:
                    return error("IMAGE_ENCODING_FAILED", "The normalized image could not be prepared. Re-export and retry.", 500)
                try:
                    identifier, expires = store.put(context, encoded.tobytes())
                except ValueError:
                    return error("ANALYSIS_CAPACITY", "This diagram exceeds the temporary memory limit. Try a smaller export.", 413)
            return jsonify(AnalysisResponse(context, identifier, expires).to_dict())
        finally:
            capacity.release()

    @app.get("/api/analysis/<identifier>/image")
    def image(identifier):
        entry = store.get(identifier)
        if entry is None:
            return error("ANALYSIS_EXPIRED", "This analysis is unknown or expired. Upload the diagram again.", 404)
        return send_file(BytesIO(entry.image_png), mimetype="image/png", max_age=0)

    @app.delete("/api/analysis/<identifier>")
    def remove(identifier):
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            return error("ORIGIN_NOT_ALLOWED", "Use the local Modulo page to clear an analysis.", 403)
        store.delete(identifier)
        return "", 204

    @app.post("/api/generate")
    def generate():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("analysis_id"), str) or set(payload) != {"analysis_id"}:
            return error("INVALID_REQUEST", "Send only an analysis_id from a server-held analysis.", 400)
        entry = store.get(payload["analysis_id"])
        if entry is None:
            return error("ANALYSIS_EXPIRED", "This analysis is unknown or expired. Upload the diagram again.", 404)
        if not entry.context.approved:
            return error("APPROVAL_REQUIRED", "Pseudocode approval is required before Python generation.", 422)
        entry.context = pipeline.generate(entry.context)
        if entry.context.has_errors:
            return jsonify({"status": "failed", "diagnostics": [item.to_dict() for item in entry.context.diagnostics], "trace": entry.context.trace}), 422
        return jsonify({"status": "done", "analysis_id": payload["analysis_id"], "code": [line.__dict__ for line in entry.context.code], "trace": entry.context.trace, "diagnostics": [item.to_dict() for item in entry.context.diagnostics]})

    @app.post("/api/approve")
    def approve():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"analysis_id"} or not isinstance(payload.get("analysis_id"), str):
            return error("INVALID_REQUEST", "Send only an analysis_id from a server-held analysis.", 400)
        entry = store.get(payload["analysis_id"])
        if entry is None:
            return error("ANALYSIS_EXPIRED", "This analysis is unknown or expired. Upload the diagram again.", 404)
        if entry.context.has_errors or entry.context.ir is None or not entry.context.pseudocode:
            return error("ANALYSIS_NOT_READY", "Only a successful analysis with pseudocode can be approved.", 409)
        entry.context.approved = True
        return jsonify({"status": "approved", "analysis_id": payload["analysis_id"], "approved": True})

    @app.post("/api/review")
    def review():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"analysis_id", "corrections"} or not isinstance(payload.get("analysis_id"), str) or not isinstance(payload.get("corrections"), dict):
            return error("INVALID_REQUEST", "Send an analysis_id and constrained corrections for flagged elements.", 400)
        corrections = payload["corrections"]
        node_text = corrections.get("nodes", {})
        edge_labels = corrections.get("edges", {})
        if not isinstance(node_text, dict) or not isinstance(edge_labels, dict) or set(corrections) - {"nodes", "edges"}:
            return error("INVALID_REQUEST", "Corrections must contain only nodes and edges mappings.", 400)
        entry = store.get(payload["analysis_id"])
        if entry is None:
            return error("ANALYSIS_EXPIRED", "This analysis is unknown or expired. Upload the diagram again.", 404)
        reviewed = pipeline.review(entry.context, node_text, edge_labels)
        entry.context = reviewed
        return jsonify(AnalysisResponse(entry.context, payload["analysis_id"], store.expiry(entry)).to_dict())

    return app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
