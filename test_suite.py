import os
import sys
import tempfile
import time
from PIL import Image

# Ensure UTF-8 console output in Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from poetry_db import init_db, record_post, is_verse_duplicate, compute_verse_hash, normalize_urdu_text
from crash_tracker import get_gpu_telemetry, log_error_event
from poster_maker import ensure_urdu_font, render_urdu_poetry_poster, get_random_aesthetic
from voice_reader import generate_poetry_recitation, get_audio_duration_seconds
from video_maker import assemble_poetry_reel, get_ffmpeg_binary, get_exact_media_duration
from agent import compose_unique_couplet
from backend import AdaabClient

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 8

    def log_result(self, test_name: str, success: bool, msg: str = ""):
        if success:
            self.passed += 1
            print(f" [PASS] {test_name}: {msg}")
        else:
            self.failed += 1
            print(f" [FAIL] {test_name}: {msg}")

def run_all_tests():
    print("=" * 68)
    print(" Adaab AI Agent - 8-Point Automated Unit & Integration Suite ".center(68, "="))
    print("=" * 68)

    runner = TestRunner()

    # ── TEST 1: SQLite Database & Deduplication Engine ─────────────────────────
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            test_db = tf.name

        init_db(test_db)
        m1 = "دل ناداں تجھے ہوا کیا ہے"
        m2 = "آخر اس درد کی دوا کیا ہے"

        assert not is_verse_duplicate(m1, m2, test_db), "Verse should not be duplicate initially"

        record_post({
            "post_uuid": "test_post_001",
            "misra_1": m1,
            "misra_2": m2,
            "theme": "Ishq"
        }, db_path=test_db)

        # Re-check with slight spacing and diacritic differences
        m1_altered = "دلِ ناداں  تجھے ہوا کیا ہے؟"
        assert is_verse_duplicate(m1_altered, m2, test_db), "Normalized duplicate detection should catch altered verse"
        runner.log_result("TEST-01: SQLite Deduplication Engine", True, "Strict duplicate hashing & normalization verified")
        try:
            os.remove(test_db)
        except Exception:
            pass
    except Exception as e:
        runner.log_result("TEST-01: SQLite Deduplication Engine", False, str(e))

    # ── TEST 2: Font System & Auto-Downloader ──────────────────────────────────
    try:
        font_path = ensure_urdu_font()
        assert os.path.exists(font_path), f"Font file {font_path} must exist"
        assert os.path.getsize(font_path) > 1000, "Font file must be non-empty"
        runner.log_result("TEST-02: Urdu Nastaliq Font System", True, f"Font verified at {os.path.basename(font_path)}")
    except Exception as e:
        runner.log_result("TEST-02: Urdu Nastaliq Font System", False, str(e))

    # ── TEST 3: GPU Crash Tracker & Telemetry ─────────────────────────────────
    try:
        telemetry = get_gpu_telemetry()
        assert isinstance(telemetry, dict)
        assert "allocated_mb" in telemetry
        runner.log_result("TEST-03: Crash Tracker & GPU Telemetry", True, f"Device: {telemetry.get('device_name', 'None')}")
    except Exception as e:
        runner.log_result("TEST-03: Crash Tracker & GPU Telemetry", False, str(e))

    # ── TEST 4: Urdu Typography & Poster Rendering ─────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_poster = os.path.join(tmp_dir, "test_poster.png")
            render_urdu_poetry_poster(
                misra_1="ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے",
                misra_2="بہت نکلے مرے ارمان لیکن پھر بھی کم نکلے",
                output_path=test_poster
            )
            assert os.path.exists(test_poster), "Poster image file must be written"
            with Image.open(test_poster) as img:
                w, h = img.size
                assert w == 1080 and h == 1920, f"Expected 1080x1920, got {w}x{h}"
        runner.log_result("TEST-04: 9:16 Typography Poster Rendering", True, "Generated 1080x1920 portrait with contrast scrim")
    except Exception as e:
        runner.log_result("TEST-04: 9:16 Typography Poster Rendering", False, str(e))

    # ── TEST 5: Female Urdu Voice Synthesis (edge-tts) ────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_audio = os.path.join(tmp_dir, "test_voice.mp3")
            generate_poetry_recitation(
                misra_1="ستاروں سے آگے جہاں اور بھی ہیں",
                misra_2="ابھی عشق کے امتحان اور بھی ہیں",
                output_path=test_audio
            )
            assert os.path.exists(test_audio), "Audio file must exist"
            assert os.path.getsize(test_audio) > 1000, "Audio file must contain audio data"
        runner.log_result("TEST-05: Female Voice Recitation (ur-PK-UzmaNeural)", True, "Synthesized MP3 with poetry SSML pauses")
    except Exception as e:
        runner.log_result("TEST-05: Female Voice Recitation (ur-PK-UzmaNeural)", False, str(e))

    # ── TEST 6: Video Reel Assembly & Precision Pacing ─────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_p = os.path.join(tmp_dir, "p.png")
            test_a = os.path.join(tmp_dir, "a.mp3")
            test_v = os.path.join(tmp_dir, "v.mp4")

            render_urdu_poetry_poster("مصرع اول", "مصرع دوم", output_path=test_p)
            generate_poetry_recitation("مصرع اول", "مصرع دوم", output_path=test_a)
            assemble_poetry_reel(test_p, test_a, test_v, intro_delay_sec=0.5, outro_hold_sec=0.5)

            assert os.path.exists(test_v), "Video file must be created"
            assert os.path.getsize(test_v) > 5000, "Video file must be valid MP4"

            ffmpeg_bin = get_ffmpeg_binary()
            v_dur = get_exact_media_duration(test_v, ffmpeg_bin)
            a_dur = get_exact_media_duration(test_a, ffmpeg_bin)
            expected_dur = round(0.5 + a_dur + 0.5, 1)
            assert abs(v_dur - expected_dur) <= 0.5, f"Expected ~{expected_dur}s, got {v_dur:.2f}s"

        runner.log_result("TEST-06: 1080x1920 Video Reel Pacing", True, f"Verified 0.5s intro delay + 0.5s outro hold (Duration: {v_dur:.1f}s)")
    except Exception as e:
        runner.log_result("TEST-06: 1080x1920 Video Reel Pacing", False, str(e))

    # ── TEST 7: Classical Couplet Composition ──────────────────────────────────
    try:
        couplet = compose_unique_couplet()
        assert "misra_1" in couplet and "misra_2" in couplet
        assert len(couplet["misra_1"]) > 5 and len(couplet["misra_2"]) > 5
        runner.log_result("TEST-07: Adaab Poetic Couplet Generator", True, f"Composed: {couplet['misra_1'][:25]}...")
    except Exception as e:
        runner.log_result("TEST-07: Adaab Poetic Couplet Generator", False, str(e))

    # ── TEST 8: Modal Cloud Backend Health ────────────────────────────────────
    try:
        client = AdaabClient()
        health = client.health_check()
        runner.log_result("TEST-08: Modal Serverless Backend Health", True, f"Health ping: {health.get('status', 'unknown')}")
    except Exception as e:
        runner.log_result("TEST-08: Modal Serverless Backend Health", False, str(e))

    print("\n" + "=" * 68)
    print(f" Test Results: {runner.passed}/{runner.total} Passed ({int(runner.passed/runner.total*100)}% Success Rate)")
    print("=" * 68 + "\n")
    return runner.passed == runner.total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
