"""All pipeline tunables, with environment overrides."""
from dataclasses import dataclass, field
import os

def integer(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value

@dataclass(frozen=True)
class Settings:
    MAX_UPLOAD_MB: int = field(default_factory=lambda: integer("MAX_UPLOAD_MB", 10))
    ALLOWED_EXTENSIONS: tuple[str, ...] = field(default_factory=lambda: tuple(item.strip().lower() for item in os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,svg,pdf").split(",")))
    TESSERACT_CMD: str | None = field(default_factory=lambda: os.getenv("TESSERACT_CMD") or None)
    OCR_LANGUAGE: str = field(default_factory=lambda: os.getenv("OCR_LANGUAGE", "eng"))
    OCR_MIN_CONFIDENCE: int = field(default_factory=lambda: integer("OCR_MIN_CONFIDENCE", 60))
    OCR_TIMEOUT_SECONDS: int = field(default_factory=lambda: integer("OCR_TIMEOUT_SECONDS", 15))
    WORKING_IMAGE_WIDTH: int = field(default_factory=lambda: integer("WORKING_IMAGE_WIDTH", 2000))
    PDF_RENDER_DPI: int = field(default_factory=lambda: integer("PDF_RENDER_DPI", 200))
    MAX_IMAGE_PIXELS: int = field(default_factory=lambda: integer("MAX_IMAGE_PIXELS", 25000000))
    MAX_WORKING_HEIGHT: int = field(default_factory=lambda: integer("MAX_WORKING_HEIGHT", 6000))
    MAX_NODES: int = field(default_factory=lambda: integer("MAX_NODES", 100))
    ARROW_ATTACH_DISTANCE: int = field(default_factory=lambda: integer("ARROW_ATTACH_DISTANCE", 24))
    # A connector whose closest approach to a node is within this many
    # thousandths of the page width touches it; farther contacts must also be
    # a genuine stroke end (not a line running past the node).
    CONNECTOR_TOUCH_PERMILLE: int = field(default_factory=lambda: integer("CONNECTOR_TOUCH_PERMILLE", 22))
    STORE_TTL_SECONDS: int = field(default_factory=lambda: integer("STORE_TTL_SECONDS", 1800))
    STORE_MAX_ENTRIES: int = field(default_factory=lambda: integer("STORE_MAX_ENTRIES", 50))
    STORE_MAX_BYTES: int = field(default_factory=lambda: integer("STORE_MAX_BYTES", 268435456))
    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024
