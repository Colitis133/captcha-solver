"""Utilities for rendering canonical clean captcha targets from labels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


CANONICAL_FONT_CANDIDATES: Sequence[str] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "fonts/Arial.ttf",
    "fonts/arial.ttf",
)


@lru_cache(maxsize=32)
def _load_canonical_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in CANONICAL_FONT_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_clean_label(
    label: str,
    width: int,
    height: int,
    *,
    font_scale: float = 0.72,
    padding: int = 12,
) -> Image.Image:
    """Render `label` in a clean, single-font style matching the noisy captcha size."""
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    max_width = max(10, width - 2 * padding)
    font_size = max(12, int((height - 2 * padding) * font_scale))
    font = _load_canonical_font(font_size)

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    while text_width > max_width and font_size > 12:
        font_size -= 1
        font = _load_canonical_font(font_size)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    x = int((width - text_width) / 2 - bbox[0])
    y = int((height - text_height) / 2 - bbox[1])

    if x < padding:
        x = padding
    if x + text_width > width - padding:
        x = max(padding, width - padding - text_width)
    if y < padding:
        y = padding
    if y + text_height > height - padding:
        y = max(padding, height - padding - text_height)

    draw.text((x, y), label, font=font, fill=(0, 0, 0))
    return canvas


__all__ = ["render_clean_label"]
