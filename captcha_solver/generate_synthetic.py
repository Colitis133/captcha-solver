"""Synthetic CAPTCHA generator (simple, dependency-light)

Generates fixed-size CAPTCHA images with random text, fonts (if available), warping and noise.
Used to expand small real datasets on-the-fly during training.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import string


DEFAULT_CHARS = string.ascii_uppercase + string.digits


def random_text(length: int = 5, chars: str = DEFAULT_CHARS):
    return ''.join(random.choice(chars) for _ in range(length))


def gen_captcha(text=None, width=160, height=60, fonts=None):
    """Generate a synthetic captcha PIL.Image in RGB.

    Args:
        text: optional label to render; if None a random label is created.
        width, height: image size
        fonts: list of file paths to truetype fonts; if None PIL default font used.
    Returns:
        (PIL.Image, label)
    """
    if text is None:
        text = random_text(5)

    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # choose font
    font = None
    if fonts:
        try:
            font = ImageFont.truetype(random.choice(fonts), size=int(height * 0.6))
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    # draw text with slight jitter per character
    x = 8
    for ch in text:
        y_jitter = random.randint(-6, 6)
        w, h = draw.textsize(ch, font=font)
        draw.text((x, max(0, (height - h) // 2 + y_jitter)), ch,
                  fill=(random.randint(0, 80),) * 3, font=font)
        x += w + random.randint(0, 6)

    # apply distortions
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.random() * 1.2))

    # add lines/noise
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(1, 3)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(random.randint(0, 120),) * 3, width=random.randint(1, 2))

    # add speckle noise
    arr = np.array(img).astype(np.uint8)
    if random.random() < 0.5:
        noise = (np.random.randn(*arr.shape) * 10).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img, text


if __name__ == "__main__":
    # quick demo
    img, label = gen_captcha()
    print("Label:", label)
    img.show()
