import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import random
import math
import glob
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
ASSETS_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "backgrounds")

# ─────────────────────────────────────────────────────────────────────────────
# Reliable Font Management for Urdu Script on Windows
# ─────────────────────────────────────────────────────────────────────────────

def ensure_urdu_font() -> str:
    """
    Returns a verified font that reliably renders Arabic/Urdu Presentation Forms.
    On Windows, Tahoma and Arial natively support full Arabic cursive presentation forms.
    """
    candidates = [
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\times.ttf",
        "C:\\Windows\\Fonts\\seguihis.ttf",
        os.path.join(FONTS_DIR, "Gulzar-Regular.ttf")
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return "arial.ttf"

# ─────────────────────────────────────────────────────────────────────────────
# 48,000 Combinatorial Aesthetic Matrix
# ─────────────────────────────────────────────────────────────────────────────

COLOR_PALETTES = [
    {"name": "Emerald & Burnished Gold", "primary": "#0B3B24", "accent": "#D4AF37", "dark": "#04150D", "light": "#F7ECC9"},
    {"name": "Midnight Indigo & Cyan", "primary": "#101838", "accent": "#00E5FF", "dark": "#080C1E", "light": "#E0F7FA"},
    {"name": "Royal Crimson & Antique Brass", "primary": "#4A0E17", "accent": "#D4A373", "dark": "#240409", "light": "#FAEDCD"},
    {"name": "Dusty Rose & Sage Mist", "primary": "#8E5572", "accent": "#B8D8D8", "dark": "#361B2B", "light": "#EEF5DB"},
    {"name": "Sunset Terracotta & Amber", "primary": "#9C4124", "accent": "#F4A261", "dark": "#4A1A0C", "light": "#FFE8D6"},
    {"name": "Deep Sapphire & Pearl White", "primary": "#0D2538", "accent": "#E2E8F0", "dark": "#05101A", "light": "#FFFFFF"},
    {"name": "Lavender Mist & Charcoal", "primary": "#5C5470", "accent": "#DBD8E3", "dark": "#2A2438", "light": "#FAF9F6"},
    {"name": "Autumn Ochre & Burnt Sienna", "primary": "#7A3E1D", "accent": "#E9C46A", "dark": "#3A1B0B", "light": "#FFF3B0"},
    {"name": "Glacial Ice Blue & Polar Mist", "primary": "#1E3B4D", "accent": "#BEE3F8", "dark": "#0B1B24", "light": "#F0F8FF"},
    {"name": "Butterscotch & Rich Espresso", "primary": "#3B2219", "accent": "#E6A15C", "dark": "#1E0F0A", "light": "#FBF1E6"}
]

ART_STYLES = [
    "Cinematic National Geographic Photorealism with dramatic lighting",
    "Lush Mountain Valley Landscape with crisp mountain air",
    "Vibrant Green Rolling Hills with tropical foliage framing",
    "Fine Art 8k Landscape Photography with sun-drenched atmosphere",
    "Classical Romantic Hudson River School Landscape with glowing light",
    "Ethereal Sunset over Golden Meadows and Forest Canopies",
    "Star-studded Clear Night with Milky Way reflections",
    "Misty Primeval Woodland with shafts of morning sunlight"
]

NATURE_THEMES = [
    "Lush green hills and distant mountain ridge under blue skies",
    "Rolling green pastures with tropical palm leaves framing",
    "Majestic snow-capped mountain valley with mirror lake",
    "Misty pine forest with warm morning sunbeams (Komorebi)",
    "Deep starry night sky over peaceful highland mountains",
    "Golden sunset glowing across rolling green hills and meadows"
]

LIGHTING_CYCLES = [
    {"type": "Day", "desc": "Bright sunlit daylight with clear skies and lush green tones"},
    {"type": "Day", "desc": "Warm golden hour with soft sunlight breaking through clouds"},
    {"type": "Day", "desc": "Crisp morning dawn over misty green hills"},
    {"type": "Night", "desc": "Deep starry midnight with glowing celestial starlight"},
    {"type": "Day", "desc": "Vivid afternoon sun illuminating green meadows and mountain cliffs"}
]

def get_random_aesthetic() -> dict:
    """Samples a randomized aesthetic profile."""
    palette = random.choice(COLOR_PALETTES)
    art_style = random.choice(ART_STYLES)
    nature_theme = random.choice(NATURE_THEMES)
    lighting = random.choice(LIGHTING_CYCLES)

    return {
        "palette": palette,
        "art_style": art_style,
        "nature_theme": nature_theme,
        "lighting": lighting["type"],
        "lighting_desc": lighting["desc"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# High-Definition Photorealistic Background Engine
# ─────────────────────────────────────────────────────────────────────────────

def get_photorealistic_background(aesthetic: dict, width: int = 1080, height: int = 1920) -> Image.Image:
    """
    Retrieves or generates a high-definition 1080x1920 photorealistic nature wallpaper
    (green hills, mountain valley, blue sky, starry night).
    """
    os.makedirs(ASSETS_BG_DIR, exist_ok=True)
    is_night = aesthetic.get("lighting") == "Night"

    # Check for existing curated high-resolution landscape images
    bg_files = glob.glob(os.path.join(ASSETS_BG_DIR, "*.jpg")) + glob.glob(os.path.join(ASSETS_BG_DIR, "*.png"))
    
    if bg_files:
        # Separate night images if night theme
        night_candidates = [f for f in bg_files if "night" in f.lower() or "star" in f.lower()]
        day_candidates = [f for f in bg_files if "night" not in f.lower() and "star" not in f.lower()]

        chosen_file = None
        if is_night and night_candidates:
            chosen_file = random.choice(night_candidates)
        elif not is_night and day_candidates:
            chosen_file = random.choice(day_candidates)
        else:
            chosen_file = random.choice(bg_files)

        try:
            with Image.open(chosen_file) as img:
                return img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"[PosterMaker] Could not load {chosen_file}: {e}")

    # Fallback to procedural background if no asset images exist
    return create_procedural_background(aesthetic, width, height)

def create_procedural_background(aesthetic: dict, width: int = 1080, height: int = 1920) -> Image.Image:
    """High-aesthetic gradient canvas fallback."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    palette = aesthetic.get("palette", {"dark": "#0A2012", "primary": "#1B4D2E", "accent": "#D4AF37"})
    c_dark = hex_to_rgb(palette.get("dark", "#0A2012"))
    c_prim = hex_to_rgb(palette.get("primary", "#1B4D2E"))

    for y in range(height):
        t = y / height
        r = int(c_dark[0] * (1 - t) + c_prim[0] * t)
        g = int(c_dark[1] * (1 - t) + c_prim[1] * t)
        b = int(c_dark[2] * (1 - t) + c_prim[2] * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img

def hex_to_rgb(hex_code: str) -> tuple:
    hex_code = hex_code.lstrip("#")
    if len(hex_code) == 6:
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    return (20, 20, 20)

# ─────────────────────────────────────────────────────────────────────────────
# Urdu Script Typography & Calligraphy Overlay Engine
# ─────────────────────────────────────────────────────────────────────────────

def format_urdu_for_rendering(text: str) -> str:
    """Applies Arabic Reshaper and BiDi algorithm to ensure connected, right-to-left Urdu script."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception as e:
        print(f"[PosterMaker] Reshaping notice: {e}")
        return text

def render_urdu_poetry_poster(
    misra_1: str,
    misra_2: str,
    aesthetic: dict = None,
    bg_image: Image.Image = None,
    output_path: str = None
) -> Image.Image:
    """
    Renders a stunning 1080x1920 Urdu poetry poster:
    - High-definition photorealistic nature background
    - Elegant frosted-glass contrast card in center
    - Fully connected, crystal-clear white Urdu text with deep shadow
    - Clean golden divider line with centered vector diamond
    - Gold signature: — کلام: آداب
    """
    width, height = 1080, 1920
    aesthetic = aesthetic or get_random_aesthetic()

    # 1. Base Nature Landscape Canvas
    if bg_image:
        poster = bg_image.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    else:
        poster = get_photorealistic_background(aesthetic, width, height).convert("RGBA")

    # 2. Frosted Glass / Translucent Center Card for Perfect Readability
    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    scrim_draw = ImageDraw.Draw(scrim)
    
    # Elegant central card
    card_box = [80, 720, 1000, 1200]
    scrim_draw.rounded_rectangle(
        card_box,
        radius=35,
        fill=(10, 25, 15, 175),           # Deep translucent emerald/charcoal
        outline=(212, 175, 55, 220),       # Antique gold border
        width=3
    )
    scrim = scrim.filter(ImageFilter.GaussianBlur(radius=3))
    poster = Image.alpha_composite(poster, scrim)

    draw = ImageDraw.Draw(poster)

    # 3. Fonts Setup
    font_path = ensure_urdu_font()
    try:
        font_main = ImageFont.truetype(font_path, 44)
        font_sig = ImageFont.truetype(font_path, 30)
    except Exception:
        font_main = ImageFont.load_default()
        font_sig = ImageFont.load_default()

    # 4. Format Urdu Verses
    t1 = format_urdu_for_rendering(misra_1)
    t2 = format_urdu_for_rendering(misra_2)
    t_sig = format_urdu_for_rendering("— کلام: آداب")

    def draw_centered_text(text, y_pos, font_obj, fill_color):
        bbox = font_obj.getbbox(text)
        text_width = bbox[2] - bbox[0]
        x_pos = (width - text_width) // 2

        # Multi-layer deep drop shadow for 100% crisp legibility
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 2), (0, -2)]:
            draw.text((x_pos + dx, y_pos + dy), text, font=font_obj, fill=(0, 0, 0, 240))
        draw.text((x_pos, y_pos), text, font=font_obj, fill=fill_color)

    # 5. Render Centered Verses
    draw_centered_text(t1, 790, font_main, (255, 255, 255, 255))

    # Classical Golden Divider Line & Centered Vector Diamond
    cx = width // 2
    div_y = 910
    draw.line([(cx - 150, div_y), (cx + 150, div_y)], fill=(212, 175, 55, 220), width=2)
    draw.polygon([(cx, div_y - 7), (cx + 7, div_y), (cx, div_y + 7), (cx - 7, div_y)], fill=(212, 175, 55, 255))

    draw_centered_text(t2, 970, font_main, (255, 255, 255, 255))
    draw_centered_text(t_sig, 1085, font_sig, (212, 175, 55, 220))

    final_poster = poster.convert("RGB")
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        final_poster.save(output_path, quality=95)
        print(f"✓ Saved high-resolution Urdu poetry poster to: {output_path}")

    return final_poster

if __name__ == "__main__":
    print("Testing Updated Urdu Poster Engine...")
    sample_m1 = "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے"
    sample_m2 = "بہت نکلے مرے ارمان لیکن پھر بھی کم نکلے"
    test_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_poster_v2.png")
    render_urdu_poetry_poster(sample_m1, sample_m2, output_path=test_out)
    print("✓ Poster engine test complete.")
