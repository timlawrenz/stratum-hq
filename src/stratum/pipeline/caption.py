"""Captioning pipeline — generate image descriptions via Ollama."""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from stratum.config import CAPTION_FILE, CAPTION_PROMPT
from stratum.pipeline.bucket import load_bucketed_image, parse_bucket_dims


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def ensure_single_paragraph(text: str) -> str:
    """Collapse newlines and excess whitespace into a single paragraph."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(part for part in text.split("\n") if part.strip())
    return " ".join(text.split()).strip()


# ---------------------------------------------------------------------------
# Ollama caption backend
# ---------------------------------------------------------------------------

class OllamaCaptionBackend:
    """Thin wrapper around the Ollama ``/api/generate`` endpoint."""

    def __init__(self, url: str, model_name: str) -> None:
        self.url = url
        self.model_name = model_name

    def generate(self, image: Image.Image, max_tokens: int = 500) -> str:
        """Send *image* (as base64 JPEG) to Ollama and return the caption."""
        buf = BytesIO()
        image.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": self.model_name,
            "prompt": CAPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()
        caption = resp.json().get("response", "")
        return ensure_single_paragraph(caption)


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def process(
    image_path: Path,
    output_dir: Path,
    backend: OllamaCaptionBackend,
    aspect_bucket: str | None = None,
    max_tokens: int = 500,
) -> bool:
    """Generate a caption for *image_path* and write it to *output_dir/caption.txt*.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        dims = parse_bucket_dims(aspect_bucket) if aspect_bucket else None
        if dims:
            img = load_bucketed_image(image_path, *dims)
        else:
            with Image.open(image_path) as im:
                img = im.convert("RGB")

        caption = backend.generate(img, max_tokens=max_tokens)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / CAPTION_FILE).write_text(caption, encoding="utf-8")
        return True

    except requests.RequestException as exc:
        eprint(f"warning: caption request failed for {image_path}: {exc}")
        return False
    except Exception as exc:
        eprint(f"warning: captioning failed for {image_path}: {exc}")
        return False
