"""Synthetic CAPTCHA generator (simple, dependency-light)

Generates fixed-size CAPTCHA images with random text, fonts (if available),
and modest geometric/photometric distortions to better mirror real-world CAPTCHAs.

Improvements over the original version (conservative / low-risk):
- variable text length (4-6)
- optional font loading from a local `fonts/` directory
- mild affine shear and small rotation
- background tint and occasional blotches/occlusions
- configurable width/height preserved
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import string
import os


DEFAULT_CHARS = string.ascii_uppercase + string.digits


def random_text(length: int = None, chars: str = DEFAULT_CHARS):
    if length is None:
        length = random.randint(4, 6)
    return ''.join(random.choice(chars) for _ in range(length))


def _find_local_fonts(dirpath="fonts"):
    """Return list of font file paths found under `dirpath` (relative to repo)."""
    if not os.path.isdir(dirpath):
        return []
    exts = ('.ttf', '.otf')
    fonts = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(exts):
                fonts.append(os.path.join(root, f))
    return fonts


def gen_captcha(text=None, width=160, height=60, fonts=None):
    """Generate a synthetic captcha PIL.Image in RGB.

    Args:
        text: optional label to render; if None a random label is created.
        width, height: image size
        fonts: list of file paths to truetype fonts; if None will try `./fonts/` then default PIL font.
    Returns:
        (PIL.Image, label)
    """
    if text is None:
        text = random_text()

    # attempt to collect fonts automatically if not provided
    if fonts is None:
        fonts = _find_local_fonts()

    # gentle off-white background to simulate scanning or compression
    bg_base = 245 + random.randint(-8, 8)
    img = Image.new('RGB', (width, height), color=(bg_base, bg_base, bg_base))
    draw = ImageDraw.Draw(img)

    # choose font
    font = None
    if fonts:
        try:
            font = ImageFont.truetype(random.choice(fonts), size=int(height * random.uniform(0.5, 0.72)))
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    # draw text with per-character jitter and slight darkness variation
    x = int(width * 0.05)
    for ch in text:
        y_jitter = random.randint(-6, 6)
        # getsize / getbbox depending on PIL version
        try:
            w, h = font.getsize(ch)
        except Exception:
            try:
                bbox = draw.textbbox((0, 0), ch, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = (int(height * 0.6), int(height * 0.6))
        # vary color a bit (dark gray range)
        shade = random.randint(0, 70)
        draw.text((x, max(0, (height - h) // 2 + y_jitter)), ch,
                  fill=(shade, shade, shade), font=font)
        x += w + random.randint(0, 6)

    # small gaussian blur sometimes
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.random() * 1.2))

    draw = ImageDraw.Draw(img)

    # add random interfering lines
    for _ in range(random.randint(1, 4)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        col = random.randint(30, 140)
        draw.line((x1, y1, x2, y2), fill=(col, col, col), width=random.randint(1, 2))

    # occasional blotches/occlusions (simulate stamps, smudges)
    if random.random() < 0.3:
        bx = random.randint(0, width - 10)
        by = random.randint(0, height - 10)
        bw = random.randint(6, 24)
        bh = random.randint(6, 24)
        draw.ellipse((bx, by, bx + bw, by + bh), fill=(random.randint(200, 255),) * 3)

    # add speckle noise (light)
    arr = np.array(img).astype(np.int16)
    if random.random() < 0.7:
        noise = (np.random.randn(*arr.shape) * random.uniform(6, 12)).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    else:
        # ensure uint8 for Image.fromarray
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # geometric warp: small shear + tiny rotation to simulate camera/scan
    pil = Image.fromarray(arr)
    if random.random() < 0.9:
        # affine shear parameters
        max_shear = 0.12
        shear = random.uniform(-max_shear, max_shear)
        # build affine matrix (a, b, c, d, e, f)
        a = 1.0
        b = shear
        c = 0
        d = 0.0
        e = 1.0
        f = 0
        try:
            pil = pil.transform(pil.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        except Exception:
            pass

    if random.random() < 0.6:
        rot = random.uniform(-6.0, 6.0)
        pil = pil.rotate(rot, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))

    return pil, text


if __name__ == "__main__":
    # quick demo
    img, label = gen_captcha()
    print("Label:", label)
    img.show()
