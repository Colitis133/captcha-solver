"""Synthetic CAPTCHA generator with stronger, varied augmentations.

This file keeps the previous, simple API: `gen_captcha(text=None, width=..., height=..., fonts=None)`
but produces a much wider range of appearance types (mesh warp/perspective,
motion blur, denser occlusions, variable spacing and overlap, textured backgrounds,
and compounded photometric noise). The goal is to better approximate real-world
captchas (Google/Cloudflare-like) so the model learns character shapes and
sequence patterns under heavy distortions.

Includes OpenCV-based elastic and perspective warps for even stronger transforms.
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageEnhance, ImageOps
import numpy as np
import random
import string
import os
import cv2
import enum
from io import BytesIO
from functools import lru_cache


DEFAULT_CHARS = string.ascii_uppercase + string.digits


class CaptchaStyle(enum.Enum):
    """Enum to define the 20 different CAPTCHA styles."""
    MIXED_FONTS_MAYHEM = 1
    BOLD_CHAOS = 2
    BLURRED_VISION = 3
    CLOUDFLARE_CLONE = 4
    GOOGLE_CLASSIC = 5
    HIGH_CONTRAST_INK = 6
    NIGHT_MODE = 7
    DUST_AND_SCRATCH = 8
    BLENDING_MADNESS = 9
    GRADIENT_STORM = 10
    THICK_SHADOWS = 11
    TILTED_TYPE = 12
    RAINY_GLASS = 13
    GRUNGE_TEXTURE = 14
    CARTOON_COMIC = 15
    ARTISTIC_CALLIGRAPHY = 16
    BROKEN_PIXELS = 17
    MOTION_BLUR_TEXT = 18
    DOUBLE_EXPOSURE = 19
    RANDOM_COLOR_EXPLOSION = 20
    MUTANT_HYBRID = 21


def random_text(length: int = None, chars: str = DEFAULT_CHARS):
    if length is None:
        length = random.randint(5, 7)
    return ''.join(random.choice(chars) for _ in range(length))


@lru_cache(maxsize=8)
def _find_local_fonts(dirpath="fonts"):
    if not os.path.isdir(dirpath):
        return []
    exts = ('.ttf', '.otf')
    fonts = []
    for root, _, files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(exts):
                fonts.append(os.path.join(root, f))
    valid_fonts = []
    for path in fonts:
        try:
            ImageFont.truetype(path, size=32)
            valid_fonts.append(path)
        except OSError:
            print(f"Warning: Skipping unsupported font file '{path}'.")
    return valid_fonts


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


# --- Style-Specific Helper Functions ---

def _apply_random_brightness_contrast(img):
    """Apply random brightness/contrast jitter."""
    if random.random() < 0.85:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.85:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.20))
    return img


def _random_hue_shift(img, max_shift=30):
    """Shift hue randomly within +/- max_shift degrees."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    hsv = np.array(img.convert('HSV'), dtype=np.uint8)
    shift = random.randint(-max_shift, max_shift)
    hsv[..., 0] = (hsv[..., 0].astype(int) + shift) % 256
    return Image.fromarray(hsv, mode='HSV').convert('RGB')


def _apply_random_perspective_tilt(pil_img, magnitude=0.18):
    """Apply a stronger perspective tilt to the image."""
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    offset = int(min(h, w) * magnitude)
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32([
        [random.randint(0, offset), random.randint(0, offset)],
        [w - 1 - random.randint(0, offset), random.randint(0, offset)],
        [random.randint(0, offset), h - 1 - random.randint(0, offset)],
        [w - 1 - random.randint(0, offset), h - 1 - random.randint(0, offset)]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(arr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))

def _sine_wave_distortion(pil_img, amplitude=5, frequency=0.05):
    """Apply a vertical sine wave distortion."""
    img_arr = np.array(pil_img)
    h, w = img_arr.shape[:2]
    
    # Create a mapping for y-coordinates
    y_map = np.arange(h)
    
    # Create a mapping for x-coordinates with sine wave offset
    x_coords = np.arange(w)
    x_map = np.tile(x_coords, (h, 1))
    
    for i in range(h):
        offset = int(amplitude * np.sin(2 * np.pi * i * frequency))
        x_map[i, :] += offset

    # Clip coordinates to be within image bounds
    x_map = np.clip(x_map, 0, w - 1).astype(np.float32)
    y_map = np.arange(h).reshape(-1, 1).astype(np.float32)
    y_map = np.tile(y_map, (1, w))

    # Remap the image
    distorted_arr = cv2.remap(img_arr, x_map, y_map, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(distorted_arr)


def _jpeg_artifacts(pil_img, quality=30):
    """Simulate JPEG compression artifacts."""
    buffer = BytesIO()
    pil_img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_glow(img, text_mask, glow_radius=5, glow_color=(0, 255, 0)):
    """Apply a glow effect around the text."""
    glow_mask = text_mask.filter(ImageFilter.GaussianBlur(glow_radius))
    glow_img = Image.new('RGB', img.size, glow_color)
    
    # Composite the glow using the blurred mask
    img.paste(glow_img, (0, 0), glow_mask)
    return img


# --- Main Style Implementations ---

def _style_mixed_fonts_mayhem(base_img, text, width, height, fonts):
    """Style 1: Each letter has a different font, color, and slight rotation."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    
    # Base image with light background
    bg_color = random.choice([(255, 255, 255), (240, 240, 240), (250, 245, 240)])
    img.paste(bg_color, (0, 0, width, height))

    x_pos = int(width * 0.05)
    for char in text:
        font_path = random.choice(fonts) if fonts else None
        font_size = int(height * random.uniform(0.6, 0.8))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        
        char_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        rotation = random.uniform(-15, 15)
        
        # Render char on a separate transparent surface to rotate it
        char_w, char_h = font.getbbox(char)[2], font.getbbox(char)[3]
        char_img = Image.new('RGBA', (char_w, char_h), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=char_color)
        
        rotated_char = char_img.rotate(rotation, expand=1, resample=Image.BILINEAR)
        
        y_pos = (height - rotated_char.height) // 2 + random.randint(-5, 5)
        img.paste(rotated_char, (x_pos, y_pos), rotated_char)
        
        x_pos += rotated_char.width + random.randint(-5, 2)

    # Apply Gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def _style_google_classic(base_img, text, width, height, fonts):
    """Style 5: Realistic fonts, sine-wave distortion, noise, and artifacts."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    
    # Light yellow/beige background
    bg_color = random.choice([(250, 245, 220), (245, 245, 230)])
    img.paste(bg_color, (0, 0, width, height))
    
    # Use a more standard font if available
    realistic_fonts = [f for f in fonts if "DejaVuSans" in f or "Arial" in f]
    font_path = random.choice(realistic_fonts) if realistic_fonts else (random.choice(fonts) if fonts else None)
    font_size = int(height * 0.7)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Draw text
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=(50, 50, 50))

    # Apply sine-wave distortion
    img = _sine_wave_distortion(img, amplitude=height/10, frequency=random.uniform(0.03, 0.05))

    # Add noise dots
    arr = np.array(img)
    noise = (np.random.randn(*arr.shape) * 15).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Simulate JPEG artifacts
    img = _jpeg_artifacts(img, quality=random.randint(40, 65))
    
    return img


def _style_bold_chaos(base_img, text, width, height, fonts):
    """Style 2: Thick, bold fonts with overlapping letters and slight blur."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((255, 255, 255), (0, 0, width, height))

    bold_fonts = [f for f in fonts if "Bold" in f or "Black" in f]
    font_path = random.choice(bold_fonts) if bold_fonts else (random.choice(fonts) if fonts else None)
    font_size = int(height * 0.85)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    x_pos = int(width * 0.05)
    for char in text:
        char_color = random.choice([(0, 0, 100), (100, 0, 0), (0, 0, 0)])
        
        # Get character size
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        
        draw.text((x_pos, (height - bbox[3]) / 2), char, font=font, fill=char_color)
        
        # Overlap letters
        x_pos += char_width - int(font_size * random.uniform(0.2, 0.4))

    # Slight blur to simulate ink bleeding
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
    
    return img


def _style_blurred_vision(base_img, text, width, height, fonts):
    """Style 3: Medium-weight fonts with strong Gaussian blur."""
    bg_gray = random.randint(180, 220)
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((bg_gray, bg_gray, bg_gray), (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    text_gray = random.randint(80, 120)
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=(text_gray, text_gray, text_gray))

    # Strong Gaussian blur
    blur_radius = random.uniform(1.5, 2.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return img


def _style_cloudflare_clone(base_img, text, width, height, fonts):
    """Style 4: Gray background, curved distortion, rotated chars, lines, and perspective warp."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    bg_gray = random.randint(220, 240)
    img.paste((bg_gray, bg_gray, bg_gray), (0, 0, width, height))

    x_pos = int(width * 0.1)
    for char in text:
        font_path = random.choice(fonts) if fonts else None
        font_size = int(height * random.uniform(0.7, 0.8))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        
        rotation = random.uniform(-15, 15)
        char_w, char_h = font.getbbox(char)[2], font.getbbox(char)[3]
        
        char_img = Image.new('RGBA', (char_w, char_h), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=(50, 50, 50))
        
        rotated_char = char_img.rotate(rotation, expand=1, resample=Image.BILINEAR)
        
        y_pos = (height - rotated_char.height) // 2
        img.paste(rotated_char, (x_pos, y_pos), rotated_char)
        
        x_pos += rotated_char.width + random.randint(-5, 0)

    # Add random arcs and lines
    for _ in range(random.randint(3, 6)):
        if random.random() < 0.5:
            # Line
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line([(x1, y1), (x2, y2)], fill=(150, 150, 150), width=1)
        else:
            # Arc
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
            start_angle, end_angle = random.randint(0, 360), random.randint(0, 360)
            draw.arc([(x1, y1), (x2, y2)], start_angle, end_angle, fill=(150, 150, 150), width=1)

    # Apply perspective warp
    arr = np.array(img)
    warped_arr = _perspective_warp(arr, max_offset=int(width * 0.1))
    
    return Image.fromarray(warped_arr)


def _style_high_contrast_ink(base_img, text, width, height, fonts):
    """Style 6: Black/white only with sharp edges."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((255, 255, 255), (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=(0, 0, 0))

    # Binarize the image to get sharp edges
    img = img.convert('L')
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    
    return img.convert('RGB')


def _style_night_mode(base_img, text, width, height, fonts):
    """Style 7: Dark background with bright neon text and glow."""
    bg_color = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 30))
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste(bg_color, (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.75)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    neon_color = random.choice([(50, 255, 50), (50, 200, 255), (255, 255, 100)])
    
    # Create a text mask for the glow effect
    text_mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(text_mask)
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    mask_draw.text((x_start, y_start), text, font=font, fill=255)

    # Apply glow
    img = _apply_glow(img, text_mask, glow_radius=random.uniform(2, 4), glow_color=neon_color)
    
    # Draw the main text on top of the glow
    draw.text((x_start, y_start), text, font=font, fill=neon_color)
    
    return img


def _style_dust_and_scratch(base_img, text, width, height, fonts):
    """Style 8: Adds random line/dot noise and semi-transparent scratches."""
    img = base_img.copy()
    bg_color = (255, 255, 255)
    img.paste(bg_color, (0, 0, width, height))

    # Sepia-like filter
    sepia_img = Image.new("RGB", (width, height), (240, 230, 200))
    img = Image.blend(img, sepia_img, 0.2)
    draw = ImageDraw.Draw(img)

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=(50, 50, 50))

    # Add dust (dots)
    for _ in range(int(width * height * 0.01)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        color = random.choice([(0, 0, 0), (255, 255, 255)])
        draw.point((x, y), fill=color)

    # Add scratches (lines)
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = x1 + random.randint(-width//2, width//2), y1 + random.randint(-height//2, height//2)
        color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
        
    return img


def _style_blending_madness(base_img, text, width, height, fonts):
    """Style 9: Text colors are very close to the background color."""
    # Dark blue text on slightly darker blue background
    bg_r, bg_g, bg_b = random.randint(0, 50), random.randint(0, 50), random.randint(100, 150)
    text_r = np.clip(bg_r + random.randint(-20, 20), 0, 255)
    text_g = np.clip(bg_g + random.randint(-20, 20), 0, 255)
    text_b = np.clip(bg_b + random.randint(-20, 20), 0, 255)
    
    bg_color = (bg_r, bg_g, bg_b)
    text_color = (text_r, text_g, text_b)

    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste(bg_color, (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=text_color)
    
    return img


def _style_thick_shadows(base_img, text, width, height, fonts):
    """Style 11: Bold text with a dark gray shadow offset."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((240, 240, 240), (0, 0, width, height))

    bold_fonts = [f for f in fonts if "Bold" in f or "Black" in f]
    font_path = random.choice(bold_fonts) if bold_fonts else (random.choice(fonts) if fonts else None)
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    
    # Draw shadow
    shadow_offset_x = random.randint(2, 4)
    shadow_offset_y = random.randint(2, 4)
    draw.text((x_start + shadow_offset_x, y_start + shadow_offset_y), text, font=font, fill=(100, 100, 100))
    
    # Draw main text
    draw.text((x_start, y_start), text, font=font, fill=(0, 0, 0))

    # Mild blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))
    
    return img


def _style_tilted_type(base_img, text, width, height, fonts):
    """Style 12: Characters individually rotated at random angles with a zigzag baseline."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((255, 255, 255), (0, 0, width, height))

    x_pos = int(width * 0.05)
    y_baseline = height // 2
    
    for char in text:
        font_path = random.choice(fonts) if fonts else None
        font_size = int(height * random.uniform(0.7, 0.8))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        
        rotation = random.uniform(-25, 25)
        char_w, char_h = font.getbbox(char)[2], font.getbbox(char)[3]
        
        char_img = Image.new('RGBA', (char_w, char_h), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=(0, 0, 0))
        
        rotated_char = char_img.rotate(rotation, expand=1, resample=Image.BILINEAR)
        
        # Zigzag baseline
        y_pos = y_baseline - rotated_char.height // 2 + random.randint(-height//6, height//6)
        img.paste(rotated_char, (x_pos, y_pos), rotated_char)
        
        x_pos += rotated_char.width - int(font_size * 0.1)

    # Add medium noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def _style_artistic_calligraphy(base_img, text, width, height, fonts):
    """Style 16: Script or cursive fonts with overlapping curves and faded ink color."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((250, 245, 240), (0, 0, width, height))

    script_fonts = [f for f in fonts if "Script" in f or "Vibes" in f]
    font_path = random.choice(script_fonts) if script_fonts else (random.choice(fonts) if fonts else None)
    font_size = int(height * 0.9)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    x_pos = int(width * 0.05)
    for char in text:
        color = random.choice([(50, 80, 50), (100, 60, 40)]) # Dark green, brown
        
        bbox = font.getbbox(char)
        char_width = bbox[2] - bbox[0]
        
        draw.text((x_pos, (height - bbox[3]) / 2 + random.randint(-3, 3)), char, font=font, fill=color)
        
        # Overlap for calligraphy effect
        x_pos += char_width - int(font_size * random.uniform(0.3, 0.5))
        
    return img


def _style_motion_blur_text(base_img, text, width, height, fonts):
    """Style 18: Applies horizontal or vertical motion blur."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((255, 255, 255), (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    x_pos = int(width * 0.1)
    for i, char in enumerate(text):
        # Each letter shifted differently
        offset = i * random.uniform(-2, 2)
        draw.text((x_pos + offset, (height - font.getbbox(char)[3]) / 2), char, font=font, fill=(0, 0, 0))
        x_pos += font.getlength(char)

    # Apply motion blur
    angle = random.choice([0, 90]) # Horizontal or vertical
    radius = random.uniform(3, 6)
    img = _motion_blur(img, radius=radius, angle=angle)
    
    return img


def _style_double_exposure(base_img, text, width, height, fonts):
    """Style 19: Two overlapping texts with different colors and transparency."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((255, 255, 255), (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2

    # First text (ghost)
    ghost_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    ghost_draw = ImageDraw.Draw(ghost_img)
    ghost_color = (150, 150, 200, 128) # Semi-transparent blue
    offset_x = random.randint(-5, 5)
    offset_y = random.randint(-5, 5)
    ghost_draw.text((x_start + offset_x, y_start + offset_y), text, font=font, fill=ghost_color)
    
    img.paste(ghost_img, (0, 0), ghost_img)

    # Second text (main)
    draw.text((x_start, y_start), text, font=font, fill=(0, 0, 0))
    
    return img


def _style_random_color_explosion(base_img, text, width, height, fonts):
    """Style 20: Background is random static noise, with thick text."""
    # Generate random color noise for the background
    noise_arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise_arr)
    draw = ImageDraw.Draw(img)

    bold_fonts = [f for f in fonts if "Bold" in f or "Black" in f]
    font_path = random.choice(bold_fonts) if bold_fonts else (random.choice(fonts) if fonts else None)
    font_size = int(height * 0.85)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    
    # Thick white or black text
    text_color = random.choice([(0, 0, 0), (255, 255, 255)])
    draw.text((x_start, y_start), text, font=font, fill=text_color)

    # Optional blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
    return img


def _style_mutant_hybrid(base_img, text, width, height, fonts):
    """Style 21: Validation-only hybrid that fuses multiple distortions unpredictably."""
    # Start from a textured background to ensure plenty of low-frequency chaos
    background = Image.new('RGB', (width, height), color=(random.randint(150, 220), random.randint(150, 220), random.randint(150, 220)))
    background = _add_textured_background(background, intensity=random.uniform(0.12, 0.22))

    # Build a richly styled text layer with per-character perturbations
    text_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    x_cursor = random.randint(int(width * 0.02), int(width * 0.08))
    for char in text:
        font_path = random.choice(fonts) if fonts else None
        font_size = int(height * random.uniform(0.6, 0.9))
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        char_img = Image.new('RGBA', (width * 2, height * 2), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        base_color = tuple(random.randint(10, 230) for _ in range(3))
        char_draw.text((int(width * 0.2), int(height * 0.2)), char, font=font, fill=base_color)

        # Apply random affine shear/rotation to the individual glyph
        shear = random.uniform(-0.4, 0.4)
        angle = random.uniform(-35, 35)
        char_img = char_img.transform(
            char_img.size,
            Image.AFFINE,
            (1, shear, 0, 0, 1, 0),
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0, 0)
        )
        char_img = char_img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0))

        # Crop tight around content to place it back on the canvas
        bbox = char_img.getbbox()
        if not bbox:
            continue
        char_img = char_img.crop(bbox)
        char_width, char_height = char_img.size

        y_offset = random.randint(int(-0.1 * height), int(0.15 * height))
        x_offset = random.randint(-5, 6)
        dest_x = max(0, min(width - char_width, x_cursor + x_offset))
        dest_y = max(-height // 6, min(height - char_height, (height - char_height) // 2 + y_offset))
        text_layer.alpha_composite(char_img, (dest_x, dest_y))
        x_cursor += int(char_width * random.uniform(0.75, 1.05))

    # Randomly jitter colors and brightness on the text layer
    text_rgb = text_layer.convert('RGB')
    text_rgb = _apply_random_brightness_contrast(text_rgb)
    if random.random() < 0.7:
        text_rgb = _random_hue_shift(text_rgb, max_shift=random.randint(15, 35))
    text_layer = text_rgb.convert('RGBA')

    # Composite background + text and add occlusions
    combined = Image.alpha_composite(background.convert('RGBA'), text_layer)
    occlusion = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    occ_draw = ImageDraw.Draw(occlusion)
    for _ in range(random.randint(3, 6)):
        stroke_rgb = [int(x) for x in np.random.randint(0, 255, 3)]
        stroke_color = tuple(stroke_rgb + [random.randint(90, 160)])
        start = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        occ_draw.line([start, end], fill=stroke_color, width=random.randint(2, 4))
    for _ in range(random.randint(1, 3)):
        radius = random.randint(height // 8, height // 4)
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        fill_rgb = [int(x) for x in np.random.randint(0, 255, 3)]
        fill = tuple(fill_rgb + [random.randint(60, 120)])
        occ_draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=fill, width=random.randint(2, 4))
    combined = Image.alpha_composite(combined, occlusion).convert('RGB')

    # Global photometric and geometric chaos borrowed from other styles
    combined = _apply_random_brightness_contrast(combined)
    if random.random() < 0.65:
        combined = _apply_random_perspective_tilt(combined, magnitude=random.uniform(0.15, 0.28))
    if random.random() < 0.6:
        arr = _elastic_warp(np.array(combined), alpha=random.uniform(35, 60), sigma=random.uniform(5.0, 7.5))
        combined = Image.fromarray(arr)
    if random.random() < 0.55:
        combined = _motion_blur(combined, radius=random.uniform(1.5, 3.8), angle=random.uniform(-60, 60))

    # High-frequency noise to finish the hybrid feel
    if random.random() < 0.8:
        arr = np.array(combined).astype(np.int16)
        noise = np.random.randint(-30, 31, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        combined = Image.fromarray(arr)

    if random.random() < 0.4:
        combined = _jpeg_artifacts(combined, quality=random.randint(30, 55))

    return combined


def _style_gradient_storm(base_img, text, width, height, fonts):
    """Style 10: Text and background filled with opposing gradients, plus wave distortion."""
    # Background gradient
    img = Image.new("RGB", (width, height))
    start_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    end_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for y in range(height):
        for x in range(width):
            # Simple linear gradient
            ratio = x / width
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            img.putpixel((x, y), (r, g, b))

    # Text with opposing gradient
    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    
    text_mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(text_mask)
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    mask_draw.text((x_start, y_start), text, font=font, fill=255)

    # Create gradient for text
    text_gradient = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            ratio = x / width
            r = int(end_color[0] * (1 - ratio) + start_color[0] * ratio) # Reversed
            g = int(end_color[1] * (1 - ratio) + start_color[1] * ratio)
            b = int(end_color[2] * (1 - ratio) + start_color[2] * ratio)
            text_gradient.putpixel((x, y), (r, g, b))

    img.paste(text_gradient, (0, 0), text_mask)

    # Apply wave-like distortion
    img = _sine_wave_distortion(img, amplitude=height/12, frequency=random.uniform(0.05, 0.08))
    
    return img


def _style_rainy_glass(base_img, text, width, height, fonts):
    """Style 13: Simulates viewing text through a rainy glass."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((230, 230, 240), (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=(100, 100, 110))

    # Apply a blur to the whole image first
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Add "water drop" distortions using elastic warp on a smaller scale
    arr = np.array(img)
    distorted_arr = _elastic_warp(arr, alpha=20, sigma=3)
    
    return Image.fromarray(distorted_arr)


def _style_grunge_texture(base_img, text, width, height, fonts):
    """Style 14: Uses a texture for the background."""
    # Create a noisy texture
    texture_arr = np.random.randint(180, 230, (height, width, 3), dtype=np.uint8)
    texture_img = Image.fromarray(texture_arr).filter(ImageFilter.GaussianBlur(1))
    
    img = texture_img.copy()
    draw = ImageDraw.Draw(img)

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    
    # Use high-contrast color for text
    text_color = (0, 0, 0) if np.mean(texture_arr) > 128 else (255, 255, 255)
    draw.text((x_start, y_start), text, font=font, fill=text_color)

    # Slight motion blur
    img = _motion_blur(img, radius=random.uniform(0.5, 1.5))
    
    return img


def _style_cartoon_comic(base_img, text, width, height, fonts):
    """Style 15: Comic-style fonts with outlines and cartoony colors."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    bg_color = random.choice([(255, 220, 100), (100, 200, 255), (255, 150, 80)])
    img.paste(bg_color, (0, 0, width, height))

    comic_fonts = [f for f in fonts if "Comic" in f or "Sans" in f] # Simple, bold fonts
    font_path = random.choice(comic_fonts) if comic_fonts else (random.choice(fonts) if fonts else None)
    font_size = int(height * 0.75)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    
    # Draw outline by drawing text multiple times with offsets
    outline_color = (0, 0, 0)
    for dx in [-2, 0, 2]:
        for dy in [-2, 0, 2]:
            if dx != 0 or dy != 0:
                draw.text((x_start + dx, y_start + dy), text, font=font, fill=outline_color)

    # Draw main text
    main_color = (255, 255, 255)
    draw.text((x_start, y_start), text, font=font, fill=main_color)
    
    return img


def _style_broken_pixels(base_img, text, width, height, fonts):
    """Style 17: Simulates low-quality compression and pixel dropout."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    img.paste((255, 255, 255), (0, 0, width, height))

    font_path = random.choice(fonts) if fonts else None
    font_size = int(height * 0.8)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    text_width = font.getlength(text)
    x_start = (width - text_width) / 2
    y_start = (height - font.getbbox(text)[3]) / 2
    draw.text((x_start, y_start), text, font=font, fill=(0, 0, 0))

    # Simulate blocky compression by downscaling and upscaling
    small_size = (width // 8, height // 8)
    img = img.resize(small_size, Image.NEAREST).resize((width, height), Image.NEAREST)

    # Pixel dropout
    arr = np.array(img)
    dropout_mask = np.random.rand(height, width) > 0.95
    arr[dropout_mask] = [255, 255, 255] # Drop pixels to white
    
    return Image.fromarray(arr)


# A dictionary to map styles to their functions
STYLE_GENERATORS = {
    CaptchaStyle.MIXED_FONTS_MAYHEM: _style_mixed_fonts_mayhem,
    CaptchaStyle.BOLD_CHAOS: _style_bold_chaos,
    CaptchaStyle.BLURRED_VISION: _style_blurred_vision,
    CaptchaStyle.CLOUDFLARE_CLONE: _style_cloudflare_clone,
    CaptchaStyle.GOOGLE_CLASSIC: _style_google_classic,
    CaptchaStyle.HIGH_CONTRAST_INK: _style_high_contrast_ink,
    CaptchaStyle.NIGHT_MODE: _style_night_mode,
    CaptchaStyle.DUST_AND_SCRATCH: _style_dust_and_scratch,
    CaptchaStyle.BLENDING_MADNESS: _style_blending_madness,
    CaptchaStyle.THICK_SHADOWS: _style_thick_shadows,
    CaptchaStyle.TILTED_TYPE: _style_tilted_type,
    CaptchaStyle.RAINY_GLASS: _style_rainy_glass,
    CaptchaStyle.GRUNGE_TEXTURE: _style_grunge_texture,
    CaptchaStyle.CARTOON_COMIC: _style_cartoon_comic,
    CaptchaStyle.ARTISTIC_CALLIGRAPHY: _style_artistic_calligraphy,
    CaptchaStyle.BROKEN_PIXELS: _style_broken_pixels,
    CaptchaStyle.MOTION_BLUR_TEXT: _style_motion_blur_text,
    CaptchaStyle.DOUBLE_EXPOSURE: _style_double_exposure,
    CaptchaStyle.RANDOM_COLOR_EXPLOSION: _style_random_color_explosion,
    CaptchaStyle.GRADIENT_STORM: _style_gradient_storm,
    CaptchaStyle.MUTANT_HYBRID: _style_mutant_hybrid,
}


def gen_captcha(text=None, width=160, height=60, fonts=None, style=None):
    """Generate a synthetic captcha PIL.Image in RGB.

    Args:
        text: optional label to render; if None a random label is created.
        width, height: image size
        fonts: list of file paths to truetype fonts; if None will try `./fonts/` then default PIL font.
        style: a CaptchaStyle enum member; if None, a random style is chosen.
    Returns:
        (PIL.Image, label)
    """
    if text is None:
        text = random_text()

    if fonts is None:
        fonts = _find_local_fonts()
        if not fonts:
            print("Warning: No fonts found in 'fonts/' directory. Using default.")

    if style is None:
        # For now, let's limit the choice to the implemented styles
        available_styles = [s for s in STYLE_GENERATORS.keys() if s != CaptchaStyle.MUTANT_HYBRID]
        style = random.choice(available_styles)

    # Create a base image that style functions can modify or replace
    img = Image.new('RGB', (width, height), color=(255, 255, 255))

    # Get the style generation function
    style_func = STYLE_GENERATORS.get(style)

    if style_func:
        # The style function is responsible for drawing and augmenting the image
        img = style_func(img, text, width, height, fonts)
    else:
        # Fallback to a simple implementation if style not found
        # This part can be removed once all styles are implemented
        font = ImageFont.load_default()
        draw.text((10, 10), text, font=font, fill=(0, 0, 0))
        print(f"Warning: Style '{style}' not implemented. Using fallback.")

    # Final common augmentations can be added here if needed,
    # but it's better to keep them within the style functions for uniqueness.

    return img, text


if __name__ == "__main__":
    # Example of generating a specific style
    print("Generating Google Classic style...")
    img, label = gen_captcha(style=CaptchaStyle.GOOGLE_CLASSIC)
    print("Label:", label)
    img.show()

    print("Generating Bold Chaos style...")
    img, label = gen_captcha(style=CaptchaStyle.BOLD_CHAOS)
    print("Label:", label)
    img.show()

    print("Generating Tilted Type style...")
    img, label = gen_captcha(style=CaptchaStyle.TILTED_TYPE)
    print("Label:", label)
    img.show()

    print("Generating Artistic Calligraphy style...")
    img, label = gen_captcha(style=CaptchaStyle.ARTISTIC_CALLIGRAPHY)
    print("Label:", label)
    img.show()

    print("Generating Broken Pixels style...")
    img, label = gen_captcha(style=CaptchaStyle.BROKEN_PIXELS)
    print("Label:", label)
    img.show()

    # Example of generating a random style from the implemented ones
    print("\nGenerating a random implemented style...")
    img, label = gen_captcha()
    print("Label:", label)
    img.show()
