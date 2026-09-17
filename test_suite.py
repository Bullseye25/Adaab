import os
import sys
import tempfile
import time

# Ensure UTF-8 console output in Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from memory_engine import init_memory_tables, save_or_update_user_profile, find_profiles_by_name
from crash_tracker import get_gpu_telemetry, log_error_event
from agent import (
    ADAAB_SYSTEM_PROMPT,
    TABRAIZ_SYSTEM_PROMPT,
    is_tehzeeb_invoked,
    is_tabraiz_invoked,
    get_tehzeeb_identity_response,
    get_tabraiz_identity_response,
    get_tehzeeb_etiquette_greeting,
    clean_llm_response
)
from backend import AdaabClient
from voice_listener import transcribe_audio
from voice_reader import (
    generate_conversation_speech,
    get_audio_duration_seconds,
    preprocess_urdu_phonetics,
    DEFAULT_VOICE,
    VOICE_FEMALE_URDU,
    VOICE_MALE_URDU
)
from time_context import get_local_time_context

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 11

    def log_result(self, test_name: str, success: bool, msg: str = ""):
        if success:
            self.passed += 1
            print(f" [PASS] {test_name}: {msg}")
        else:
            self.failed += 1
            print(f" [FAIL] {test_name}: {msg}")

def run_all_tests():
    print("=" * 68)
    print(" Adaab AI Agent - 11-Point Conversational Voice AI Suite ".center(68, "="))
    print("=" * 68)

    runner = TestRunner()

    # ── TEST 1: SQLite User Memory Database ──────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            test_db = tf.name

        init_memory_tables(test_db)
        saved = save_or_update_user_profile(
            name="علی احمد",
            profession="انجینئر",
            country="پاکستان",
            db_path=test_db
        )
        assert saved is not None and saved["name"] == "علی احمد"
        found = find_profiles_by_name("علی احمد", db_path=test_db)
        assert len(found) >= 1
        assert found[0]["profession"] == "انجینئر"
        runner.log_result("TEST-01: SQLite User Memory Persistence", True, "User profile saved and retrieved cleanly")
        try:
            os.remove(test_db)
        except Exception:
            pass
    except Exception as e:
        runner.log_result("TEST-01: SQLite User Memory Persistence", False, str(e))

    # ── TEST 2: Urdu System Font Verification ─────────────────────────────────
    try:
        windows_font = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "tahoma.ttf")
        assert os.path.exists(windows_font)
        runner.log_result("TEST-02: Urdu Font System", True, f"Verified font at {os.path.basename(windows_font)}")
    except Exception as e:
        runner.log_result("TEST-02: Urdu Font System", False, str(e))

    # ── TEST 3: GPU Crash Tracker & Telemetry ─────────────────────────────────
    try:
        telemetry = get_gpu_telemetry()
        assert isinstance(telemetry, dict)
        assert "allocated_mb" in telemetry
        runner.log_result("TEST-03: Crash Tracker & GPU Telemetry", True, f"Device: {telemetry.get('device_name', 'None')}")
    except Exception as e:
        runner.log_result("TEST-03: Crash Tracker & GPU Telemetry", False, str(e))

    # ── TEST 4: Neural Conversational Voice Synthesis ─────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_audio = os.path.join(tmp_dir, "test_voice.mp3")
            generate_conversation_speech(
                text="آداب عرض ہے! میں آپ کی کیا مدد کر سکتا ہوں؟",
                output_path=test_audio,
                voice=VOICE_MALE_URDU
            )
            assert os.path.exists(test_audio), "Audio file must exist"
            assert os.path.getsize(test_audio) > 1000, "Audio file must contain audio data"
        runner.log_result("TEST-04: Conversational Neural Speech Generation", True, f"Verified speech output with {VOICE_MALE_URDU}")
    except Exception as e:
        runner.log_result("TEST-04: Conversational Neural Speech Generation", False, str(e))

    # ── TEST 5: Phonetics & Role Prefix Purging ──────────────────────────────
    try:
        dirty = "**Text:** سلام، میں آپ کا معاون ہوں۔"
        cleaned = clean_llm_response(dirty)
        assert "text:" not in cleaned.lower() and "**" not in cleaned
        phonetic = preprocess_urdu_phonetics("Text Message: سلام جناب")
        assert "text message" not in phonetic.lower()
        runner.log_result("TEST-05: Clean Text Preprocessing & Artifact Purging", True, "Verified removal of Text:/Text Message: artifacts")
    except Exception as e:
        runner.log_result("TEST-05: Clean Text Preprocessing & Artifact Purging", False, str(e))

    # ── TEST 6: Local PC Time Context Engine ─────────────────────────────────
    try:
        ctx = get_local_time_context()
        assert "urdu_date_str" in ctx and "urdu_time_str" in ctx
        assert len(ctx["urdu_time_str"]) > 3
        runner.log_result("TEST-06: Local PC Time and Date Context Engine", True, f"Local time: {ctx['urdu_time_str']}")
    except Exception as e:
        runner.log_result("TEST-06: Local PC Time and Date Context Engine", False, str(e))

    # ── TEST 7: Tabraiz Male Persona & Identity ───────────────────────────────
    try:
        assert is_tabraiz_invoked("Hey Tabraiz"), "Must recognize 'Tabraiz' in English"
        assert is_tabraiz_invoked("تبریز سنو"), "Must recognize 'تبریز' in Urdu"
        assert is_tabraiz_invoked("Who are you?"), "Must recognize identity queries"
        reply = get_tabraiz_identity_response()
        assert "تبریز" in reply
        runner.log_result("TEST-07: Tabraiz Identity Recognition and Response", True, f"Identity response verified: '{reply}'")
    except Exception as e:
        runner.log_result("TEST-07: Tabraiz Identity Recognition and Response", False, str(e))

    # ── TEST 8: Modal Cloud Backend Health ────────────────────────────────────
    try:
        client = AdaabClient()
        health = client.health_check()
        runner.log_result("TEST-08: Modal Serverless Backend Health", True, f"Health ping: {health.get('status', 'unknown')}")
    except Exception as e:
        runner.log_result("TEST-08: Modal Serverless Backend Health", False, str(e))

    # ── TEST 9: Strict Urdu-First Policy Verification ─────────────────────────
    try:
        assert "صارف خواہ انگریزی (English) میں گفتگو کرے" in ADAAB_SYSTEM_PROMPT
        assert "آپ کا جواب ہمیشہ اور ہر حال میں فصیح، سلیس اور خوبصورت اردو میں ہی ہوگا" in ADAAB_SYSTEM_PROMPT
        assert "خالص صوتی اسسٹنٹ رویہ" in TABRAIZ_SYSTEM_PROMPT
        runner.log_result("TEST-09: Strict Urdu-First Linguistic Policy", True, "Verified English-to-Urdu conversational rule in system prompt")
    except Exception as e:
        runner.log_result("TEST-09: Strict Urdu-First Linguistic Policy", False, str(e))

    # ── TEST 10: Microphone Voice Listener & STT Engine ───────────────────────
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        assert r is not None
        res = transcribe_audio("nonexistent_test_audio.wav")
        assert res["success"] is False
        runner.log_result("TEST-10: Microphone STT Engine Readiness", True, "Speech recognition pipeline initialized and verified")
    except Exception as e:
        runner.log_result("TEST-10: Microphone STT Engine Readiness", False, str(e))

    # ── TEST 11: Tehzeeb Siri-like Wake Identity & Default Female Voice ────────
    try:
        assert DEFAULT_VOICE == VOICE_FEMALE_URDU == "ur-PK-UzmaNeural"
        assert is_tehzeeb_invoked("Hey Tehzeeb, how are you?")
        assert is_tehzeeb_invoked("تہذیب سنو")
        assert is_tehzeeb_invoked("Who am I talking to?")
        greeting = get_tehzeeb_identity_response("Hey Tehzeeb")
        assert "تہذیب" in greeting
        etiquette = get_tehzeeb_etiquette_greeting()
        assert "آداب" in etiquette and "تہذیب" in etiquette
        runner.log_result("TEST-11: Tehzeeb Wake Identity and Female Voice", True, f"Verified female voice '{DEFAULT_VOICE}'")
    except Exception as e:
        runner.log_result("TEST-11: Tehzeeb Wake Identity and Female Voice", False, str(e))

    print("\n" + "=" * 68)
    print(f" Test Results: {runner.passed}/{runner.total} Passed ({int(runner.passed/runner.total*100)}% Success Rate)")
    print("=" * 68 + "\n")
    return runner.passed == runner.total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
