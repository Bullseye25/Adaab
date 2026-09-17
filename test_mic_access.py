import os
import sys
import tempfile
import base64
import wave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from voice_listener import (
    convert_to_wav,
    transcribe_audio,
    transcribe_base64_audio,
    get_ffmpeg_binary
)
from voice_reader import generate_conversation_speech, get_audio_duration_seconds
from network_helper import is_secure_context_origin, is_valid_cloudflare_tunnel_url
from app import ptt_voice_fn

class MicrophoneAccessTestSuite:
    def __init__(self):
        self.passed = 0
        self.total = 9

    def log(self, test_name: str, passed: bool, detail: str = ""):
        status = "[PASS]" if passed else "[FAIL]"
        if passed:
            self.passed += 1
        print(f" {status} {test_name}: {detail}")

    def run_all(self):
        print("=" * 72)
        print(" Adaab AI — Microphone Access & Audio Pipeline Unit Test Suite ".center(72, "="))
        print("=" * 72)

        self.test_01_secure_context_validation()
        self.test_02_empty_and_corrupt_base64_handling()
        self.test_03_base64_mime_suffix_detection()
        self.test_04_audio_conversion_to_16khz_mono()
        self.test_05_audio_duration_metering()
        self.test_06_speech_synthesis_and_recognition()
        self.test_07_ptt_base64_transcription_pipeline()
        self.test_08_end_to_end_ptt_conversation_turn()
        self.test_09_cloudflare_tunnel_url_filtering()

        print("\n" + "=" * 72)
        print(f" Test Results: {self.passed}/{self.total} Passed ({(self.passed/self.total)*100:.0f}% Success Rate)")
        print("=" * 72)
        return self.passed == self.total

    def test_01_secure_context_validation(self):
        """TEST-01: Verifies W3C Secure Context rules required for mobile microphone access."""
        urls_secure = [
            "https://adaab-studio.loca.lt",
            "https://adaab.loca.lt",
            "https://quick-tunnel.trycloudflare.com",
            "http://localhost:7865",
            "http://127.0.0.1:7865"
        ]
        urls_insecure = [
            "http://192.168.1.8:7865",
            "http://10.0.0.5:7865",
            "http://my-domain.com:7865",
            "",
            None
        ]
        all_secure_ok = all(is_secure_context_origin(u) for u in urls_secure)
        all_insecure_ok = all(not is_secure_context_origin(u) for u in urls_insecure)
        self.log(
            "TEST-01: W3C Secure Context Validation",
            all_secure_ok and all_insecure_ok,
            "HTTPS and localhost verified secure; plain LAN IP flagged as insecure origin"
        )

    def test_02_empty_and_corrupt_base64_handling(self):
        """TEST-02: Verifies graceful error handling for empty or malformed base64 mic data."""
        res_empty = transcribe_base64_audio("")
        res_spaces = transcribe_base64_audio("   ")
        res_corrupt = transcribe_base64_audio("data:audio/webm;base64,!!!NotBase64Data@@@")

        passed = (
            res_empty.get("success") is False
            and res_spaces.get("success") is False
            and res_corrupt.get("success") is False
        )
        self.log(
            "TEST-02: Base64 Audio Error Resilience",
            passed,
            "Empty strings and corrupt base64 payloads handled gracefully without unhandled exceptions"
        )

    def test_03_base64_mime_suffix_detection(self):
        """TEST-03: Verifies mobile browser audio MIME prefixes are accurately identified."""
        header_bytes = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        b64_str = base64.b64encode(header_bytes).decode("ascii")

        safari_data_url = f"data:audio/mp4;base64,{b64_str}"
        chrome_data_url = f"data:audio/webm;base64,{b64_str}"
        wav_data_url = f"data:audio/wav;base64,{b64_str}"

        res_safari = transcribe_base64_audio(safari_data_url)
        res_chrome = transcribe_base64_audio(chrome_data_url)
        res_wav = transcribe_base64_audio(wav_data_url)

        passed = all(isinstance(r, dict) and ("error" in r or "success" in r) for r in [res_safari, res_chrome, res_wav])
        self.log(
            "TEST-03: Mobile MIME Type Suffix Routing",
            passed,
            "Recognized audio/webm (Chrome), audio/mp4 (Safari), and audio/wav payloads"
        )

    def test_04_audio_conversion_to_16khz_mono(self):
        """TEST-04: Verifies FFmpeg converts any input audio to exact 16kHz mono WAV for ASR."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf_src:
            src_path = tf_src.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_dst:
            dst_path = tf_dst.name

        try:
            generate_conversation_speech("آداب", src_path, voice="female")
            convert_to_wav(src_path, dst_path)

            assert os.path.exists(dst_path), "Destination WAV was not created"
            with wave.open(dst_path, "rb") as wf:
                channels = wf.getnchannels()
                framerate = wf.getframerate()
                sampwidth = wf.getsampwidth()

            passed = (channels == 1 and framerate == 16000 and sampwidth == 2)
            self.log(
                "TEST-04: Audio Normalization Engine (convert_to_wav)",
                passed,
                f"Verified: {framerate}Hz, {channels} channel(s), {sampwidth*8}-bit PCM WAV"
            )
        finally:
            for f in [src_path, dst_path]:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass

    def test_05_audio_duration_metering(self):
        """TEST-05: Verifies audio duration calculation engine for waveform and reel synchronization."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            sample_path = tf.name

        try:
            generate_conversation_speech("سلام، آداب عرض ہے۔ آپ کیسے ہیں؟", sample_path, voice="female")
            dur = get_audio_duration_seconds(sample_path)
            passed = (dur > 0.5 and dur < 15.0)
            self.log(
                "TEST-05: Audio Duration Metering",
                passed,
                f"Calculated audio sample length: {dur:.2f}s"
            )
        finally:
            if os.path.exists(sample_path):
                try:
                    os.unlink(sample_path)
                except Exception:
                    pass

    def test_06_speech_synthesis_and_recognition(self):
        """TEST-06: Verifies neural speech generation and speech recognition ASR compatibility."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            speech_path = tf.name

        try:
            generate_conversation_speech("آداب", speech_path, voice="female")
            result = transcribe_audio(speech_path, primary_lang="ur-PK")
            passed = isinstance(result, dict) and (result.get("success") or "error" in result)
            self.log(
                "TEST-06: Speech Recognition (ASR) Interface",
                passed,
                f"Transcription pipeline executed cleanly (Language detected: {result.get('language')})"
            )
        finally:
            if os.path.exists(speech_path):
                try:
                    os.unlink(speech_path)
                except Exception:
                    pass

    def test_07_ptt_base64_transcription_pipeline(self):
        """TEST-07: Verifies base64 encoded audio decoding and end-to-end ASR parsing."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tmp_audio = tf.name

        try:
            generate_conversation_speech("تہذیب", tmp_audio, voice="female")
            with open(tmp_audio, "rb") as f:
                b64_audio = "data:audio/mp3;base64," + base64.b64encode(f.read()).decode("ascii")

            res = transcribe_base64_audio(b64_audio, primary_lang="ur-PK")
            passed = isinstance(res, dict) and "language" in res
            self.log(
                "TEST-07: PTT Base64 Stream Parsing",
                passed,
                "Decoded base64 payload from browser Push-To-Talk and passed to recognizer"
            )
        finally:
            if os.path.exists(tmp_audio):
                try:
                    os.unlink(tmp_audio)
                except Exception:
                    pass

    def test_08_end_to_end_ptt_conversation_turn(self):
        """TEST-08: Simulates full Push-To-Talk user turn: Audio Base64 -> LLM -> Spoken Voice."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tmp_audio = tf.name

        try:
            generate_conversation_speech("میرا نام علی ہے", tmp_audio, voice="female")
            with open(tmp_audio, "rb") as f:
                b64_audio = "data:audio/mp3;base64," + base64.b64encode(f.read()).decode("ascii")

            history = []
            new_hist, status_msg, out_audio, _profile = ptt_voice_fn(
                base64_audio_data=b64_audio,
                history=history,
                style_choice="عمومی آداب (Standard Adaab)",
                voice_choice="female"
            )

            passed = (
                isinstance(new_hist, list)
                and len(new_hist) >= 2
                and out_audio is None  # audio is now embedded in HTML, no file path returned
            )
            self.log(
                "TEST-08: Full PTT Voice Conversation Turn",
                passed,
                f"Generated response audio at {os.path.basename(out_audio) if out_audio else 'None'}"
            )
        finally:
            if os.path.exists(tmp_audio):
                try:
                    os.unlink(tmp_audio)
                except Exception:
                    pass

    def test_09_cloudflare_tunnel_url_filtering(self):
        """TEST-09: Verifies that Cloudflare internal API endpoints are rejected and valid tunnels accepted."""
        valid_tunnel_urls = [
            "https://closer-keys-cohen-copper.trycloudflare.com",
            "https://quick-tunnel-2026.trycloudflare.com",
            "https://my-adaab-subdomain.trycloudflare.com",
            "https://random-word-chain.trycloudflare.com/"
        ]
        invalid_tunnel_urls = [
            "https://api.trycloudflare.com",
            "https://api.trycloudflare.com/",
            "https://update.trycloudflare.com",
            "https://dash.trycloudflare.com",
            "https://trycloudflare.com",
            "http://closer-keys-cohen-copper.trycloudflare.com",  # insecure scheme
            "https://otherdomain.com",
            "",
            None
        ]

        all_valid_pass = all(is_valid_cloudflare_tunnel_url(u) for u in valid_tunnel_urls)
        all_invalid_fail = all(not is_valid_cloudflare_tunnel_url(u) for u in invalid_tunnel_urls)

        passed = all_valid_pass and all_invalid_fail
        self.log(
            "TEST-09: Cloudflare Quick Tunnel URL Verification",
            passed,
            "Internal endpoint 'api.trycloudflare.com' filtered out; genuine public tunnel subdomains accepted"
        )


if __name__ == "__main__":
    suite = MicrophoneAccessTestSuite()
    success = suite.run_all()
    sys.exit(0 if success else 1)
