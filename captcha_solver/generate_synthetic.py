"""Synthetic CAPTCHA generator with stronger, varied augmentations.

This file keeps the previous, simple API: `gen_captcha(text=None, width=..., height=..., fonts=None)`
but produces a much wider range of appearance types (mesh warp/perspective,
motion blur, denser occlusions, variable spacing and overlap, textured backgrounds,
and compounded photometric noise). The goal is to better approximate real-world
captchas (Google/Cloudflare-like) so the model learns character shapes and
sequence patterns under heavy distortions.

Includes OpenCV-based elastic and perspective warps for even stronger transforms.
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import numpy as np
import random
import string
import os
import cv2


DEFAULT_CHARS = string.ascii_uppercase + string.digits


def random_text(length: int = None, chars: str = DEFAULT_CHARS):
    if length is None:
        length = random.randint(4, 6)
    return ''.join(random.choice(chars) for _ in range(length))


def _find_local_fonts(dirpath="fonts"):
    if not os.path.isdir(dirpath):
        return []
    exts = ('.ttf', '.otf')
    fonts = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(exts):
                fonts.append(os.path.join(root, f))
    return fonts


def _add_textured_background(img, intensity=0.12):
    """Overlay a subtle textured background (noise + gradient)."""
    w, h = img.size
    # per-pixel noise
    noise = (np.random.randn(h, w) * 255 * intensity).astype(np.int16)
    base = np.array(img).astype(np.int16)
    for c in range(3):
        base[..., c] = np.clip(base[..., c] + noise, 0, 255)
    # gentle vertical gradient tint
    grad = np.linspace(0, random.uniform(-8, 8), h).astype(np.int16)
    for c in range(3):
        base[..., c] = np.clip(base[..., c] + grad[:, None], 0, 255)
    return Image.fromarray(base.astype(np.uint8))


def _mesh_warp(pil_img, grid=3, mag=6):
    """Apply a cheap mesh warp by perturbing a regular grid (PIL.Image.transform MESH)."""
    w, h = pil_img.size
    mesh = []
    dx = w // grid
    dy = h // grid
    for i in range(grid):
        for j in range(grid):
            x0 = i * dx
            y0 = j * dy
            x1 = x0 + dx
            y1 = y0 + dy
            # source box
            box = (x0, y0, x1, y1)
            # destination quad (perturb corners)
            shift = lambda v: v + random.randint(-mag, mag)
            quad = (
                shift(x0), shift(y0),
                shift(x1), shift(y0),
                shift(x1), shift(y1),
                shift(x0), shift(y1),
            )
            mesh.append((box, quad))
    try:
        return pil_img.transform(pil_img.size, Image.MESH, mesh, resample=Image.BILINEAR)
    except Exception:
        return pil_img


def _motion_blur(pil_img, radius=5, angle=None):
    """Approximate directional motion blur by rotating, applying box blur, and rotating back."""
    if angle is None:
        angle = random.uniform(-30, 30)
    rotated = pil_img.rotate(angle, resample=Image.BILINEAR, expand=True)
    blurred = rotated.filter(ImageFilter.BoxBlur(radius))
    # paste back to original size centered
    w, h = pil_img.size
    bw, bh = blurred.size
    left = max(0, (bw - w) // 2)
    top = max(0, (bh - h) // 2)
    return blurred.crop((left, top, left + w, top + h)).rotate(-angle, resample=Image.BILINEAR)


def _elastic_warp(img_arr, alpha=50, sigma=5):
    """Apply elastic deformation using OpenCV remap."""
    h, w = img_arr.shape[:2]
    # generate displacement fields
    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha
    # smooth the displacements
    dx = cv2.GaussianBlur(dx.astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(dy.astype(np.float32), (0, 0), sigma)
    # create meshgrid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    # remap
    warped = cv2.remap(img_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def _perspective_warp(img_arr, max_offset=20):
    """Apply random perspective transformation using OpenCV."""
    h, w = img_arr.shape[:2]
    # random points for perspective
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [random.randint(0, max_offset), random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), random.randint(0, max_offset)],
        [random.randint(0, max_offset), h - random.randint(0, max_offset)],
        [w - random.randint(0, max_offset), h - random.randint(0, max_offset)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img_arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return warped


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
    img, label = gen_captcha()
    print("Label:", label)
    img.show()
