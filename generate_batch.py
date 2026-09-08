import os
import sys
import time
import base64
import random
from datetime import datetime

# Ensure UTF-8 console output in Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from poetry_db import record_post, is_verse_duplicate, get_history_summary
from agent import compose_unique_couplet, CLASSICAL_THEMES
from poster_maker import get_random_aesthetic, render_urdu_poetry_poster
from voice_reader import generate_poetry_recitation
from video_maker import assemble_poetry_reel
from crash_tracker import crash_protected, log_error_event
from backend import AdaabClient

OUTPUT_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

def ensure_output_dir():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

@crash_protected(stage_name="batch_post_generation")
def generate_single_post(post_index: int, total_posts: int, theme: str = None, client: AdaabClient = None) -> dict:
    """
    Executes the end-to-end post generation:
    1. Composes fresh, unique Urdu couplet
    2. Samples 48k aesthetic matrix
    3. Renders 9:16 portrait poster
    4. Generates female voice recitation (ur-PK-UzmaNeural)
    5. Assembles 1080x1920 video reel (0.5s intro delay + audio + 0.5s outro hold)
    6. Saves all 4 assets to local output/ directory
    7. Records into SQLite database
    """
    ensure_output_dir()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    post_uuid = f"post_{post_index:03d}_{timestamp_str}"
    post_folder = os.path.join(OUTPUT_BASE_DIR, post_uuid)
    os.makedirs(post_folder, exist_ok=True)

    print(f"\n[{post_index}/{total_posts}] Generating Post: {post_uuid}...")

    # Step 1: Compose unique Urdu poetry
    print("  -> Composing unique classical Urdu verse...")
    couplet_data = compose_unique_couplet(theme=theme, backend_client=client)
    misra_1 = couplet_data["misra_1"]
    misra_2 = couplet_data["misra_2"]
    print(f"     مصرع اول: {misra_1}")
    print(f"     مصرع دوم: {misra_2}")

    # Step 2: Sample from 48,000 Aesthetic Matrix
    aesthetic = get_random_aesthetic()
    print(f"  -> Aesthetic Selected: {aesthetic['palette']['name']} | {aesthetic['lighting']}")
    print(f"     Art Style: {aesthetic['art_style'][:40]}...")
    print(f"     Nature Theme: {aesthetic['nature_theme'][:40]}...")

    # File paths
    poster_path = os.path.join(post_folder, "poster.png")
    audio_path = os.path.join(post_folder, "audio.mp3")
    reel_path = os.path.join(post_folder, "reel.mp4")
    text_path = os.path.join(post_folder, "poetry.txt")

    # Step 3: Render 9:16 Urdu Typography Poster
    print("  -> Rendering high-resolution 1080x1920 Nastaliq poster...")
    render_urdu_poetry_poster(
        misra_1=misra_1,
        misra_2=misra_2,
        aesthetic=aesthetic,
        output_path=poster_path
    )

    # Step 4: Synthesize Expressive Female Urdu Voice (ur-PK-UzmaNeural)
    print("  -> Synthesizing female Urdu recitation audio (ur-PK-UzmaNeural)...")
    generate_poetry_recitation(
        misra_1=misra_1,
        misra_2=misra_2,
        output_path=audio_path,
        rate_offset="-15%"
    )

    # Step 5: Assemble 1080x1920 MP4 Video Reel (0.5s intro delay + audio + 0.5s outro hold)
    print("  -> Assembling 1080x1920 video reel (0.5s intro + audio + 0.5s outro)...")
    assemble_poetry_reel(
        poster_path=poster_path,
        audio_path=audio_path,
        output_video_path=reel_path,
        intro_delay_sec=0.5,
        outro_hold_sec=0.5
    )

    # Step 6: Save Comprehensive Metadata Text File
    metadata_content = (
        f"=================================================================\n"
        f" آداب (Adaab) - Urdu Poetry & Reel Metadata\n"
        f" Post UUID: {post_uuid} | Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"=================================================================\n\n"
        f"[Urdu Script / اردو متن]\n"
        f"{misra_1}\n"
        f"{misra_2}\n\n"
        f"[Roman Urdu Transliteration]\n"
        f"{couplet_data.get('roman_urdu', '')}\n\n"
        f"[English Poetic Translation]\n"
        f"{couplet_data.get('english_translation', '')}\n\n"
        f"[Theme & Aesthetics]\n"
        f"Theme: {couplet_data.get('theme', '')}\n"
        f"Color Palette: {aesthetic['palette']['name']}\n"
        f"Art Style: {aesthetic['art_style']}\n"
        f"Nature Theme: {aesthetic['nature_theme']}\n"
        f"Lighting Cycle: {aesthetic['lighting']} ({aesthetic['lighting_desc']})\n\n"
        f"[Suggested Social Hashtags]\n"
        f"{couplet_data.get('hashtags', '#UrduPoetry #Shayari #Adaab #Reels')}\n\n"
        f"[Generated Media Files]\n"
        f"Poster: {poster_path}\n"
        f"Audio:  {audio_path}\n"
        f"Video:  {reel_path}\n"
    )

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(metadata_content)

    # Step 7: Record into Local SQLite Database
    record_data = {
        "post_uuid": post_uuid,
        "misra_1": misra_1,
        "misra_2": misra_2,
        "theme": couplet_data.get("theme", "General"),
        "color_palette": aesthetic["palette"]["name"],
        "art_style": aesthetic["art_style"],
        "nature_theme": aesthetic["nature_theme"],
        "lighting": aesthetic["lighting"],
        "poster_path": poster_path,
        "audio_path": audio_path,
        "video_path": reel_path,
        "english_translation": couplet_data.get("english_translation", ""),
        "roman_urdu": couplet_data.get("roman_urdu", ""),
        "status": "completed"
    }
    row_id = record_post(record_data)
    print(f"✓ Post #{post_index} complete and logged to database (ID: {row_id})!")

    return {
        "post_uuid": post_uuid,
        "folder": post_folder,
        "poster_path": poster_path,
        "audio_path": audio_path,
        "video_path": reel_path,
        "text_path": text_path
    }

def run_batch_generation(count: int):
    """Orchestrates batch post generation and opens File Explorer upon completion."""
    print("=" * 68)
    print(" آداب (Adaab) - Autonomous Urdu Poetry & Reel Studio ".center(68, "="))
    print(" Multi-Modal Cloud Pipeline | Poem + Poster + Voice + Reel ".center(68, " "))
    print("=" * 68)
    print(f"\nStarting batch generation of {count} complete post(s)...")

    client = AdaabClient()
    successful = 0
    start_time = time.time()

    if count == 1:
        try:
            generate_single_post(post_index=1, total_posts=1, client=client)
            successful = 1
        except Exception as e:
            print(f"\n❌ Error generating post 1: {e}")
            log_error_event("batch_post_1", e)
    else:
        import concurrent.futures
        workers = min(count, 3)
        print(f"  [Concurrency] Pipelining generation across {workers} parallel workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate_single_post, i, count, None, client): i
                for i in range(1, count + 1)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                    successful += 1
                except Exception as e:
                    print(f"\n❌ Error generating post {idx}: {e}")
                    log_error_event(f"batch_post_{idx}", e)

    elapsed = round(time.time() - start_time, 1)
    print("\n" + "=" * 68)
    print(f" Batch Complete: {successful}/{count} posts created successfully in {elapsed}s!")
    print(f" Saved to: {OUTPUT_BASE_DIR}")
    print("=" * 68 + "\n")

    # Automatically pop open Windows File Explorer
    try:
        if sys.platform == "win32":
            os.startfile(OUTPUT_BASE_DIR)
            print("✓ Opened Output folder in Windows File Explorer.")
    except Exception as e:
        print(f"Notice: Could not automatically open Explorer: {e}")

def main():
    print("=" * 68)
    print(" آداب (Adaab) - 1-Click Batch Generator ".center(68, "="))
    print("=" * 68)
    
    # Check if number was passed as command line argument
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
        except ValueError:
            num = 1
    else:
        user_input = input("\nHow many posts would you like to generate? (default 1): ").strip()
        try:
            num = int(user_input) if user_input else 1
        except ValueError:
            print("Invalid number, defaulting to 1.")
            num = 1

    num = max(1, min(num, 50))  # Safeguard range [1, 50]
    run_batch_generation(num)

if __name__ == "__main__":
    main()
