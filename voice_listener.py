import os
import sys
import shutil
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def get_ffmpeg_binary() -> str:
    """Finds FFmpeg executable."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def convert_to_wav(input_audio_path: str, output_wav_path: str) -> str:
    """Converts any audio format (webm, mp3, ogg, m4a) to 16kHz mono WAV for optimal ASR."""
    ffmpeg_bin = get_ffmpeg_binary()
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_audio_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_wav_path

def transcribe_audio(audio_path: str, primary_lang: str = "ur-PK") -> dict:
    """
    Transcribes spoken audio from microphone or audio file:
    - Supports Urdu ('ur-PK') and English ('en-US')
    - Automatically detects language and converts accurately
    - 100% free, zero API key required
    Returns: {"text": str, "language": str, "success": bool}
    """
    if not os.path.exists(audio_path):
        return {"text": "", "language": "unknown", "success": False, "error": "Audio file not found"}

    import speech_recognition as sr

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    # Ensure WAV format
    temp_wav = audio_path if audio_path.lower().endswith(".wav") else audio_path + "_converted.wav"
    converted = False
    if not audio_path.lower().endswith(".wav"):
        try:
            convert_to_wav(audio_path, temp_wav)
            converted = True
        except Exception as e:
            print(f"[VoiceListener] Audio conversion notice: {e}")
            temp_wav = audio_path

    try:
        with sr.AudioFile(temp_wav) as source:
            # Do not discard initial speech frames with long ambient noise adjustment
            audio_data = recognizer.record(source)

        # 1. Primary transcription in Urdu (ur-PK)
        transcript_ur = ""
        try:
            transcript_ur = recognizer.recognize_google(audio_data, language="ur-PK").strip()
        except Exception:
            transcript_ur = ""

        # 2. Only if Urdu transcription returned empty, try English fallback
        transcript_en = ""
        if not transcript_ur:
            try:
                transcript_en = recognizer.recognize_google(audio_data, language="en-US").strip()
            except Exception:
                transcript_en = ""

        final_text = transcript_ur or transcript_en
        detected_lang = "ur" if transcript_ur else ("en" if transcript_en else "unknown")

        if final_text:
            return {
                "text": final_text,
                "language": detected_lang,
                "success": True
            }
        else:
            return {
                "text": "",
                "language": "unknown",
                "success": False,
                "error": "No audible speech detected"
            }

    except sr.UnknownValueError:
        return {"text": "", "language": "unknown", "success": False, "error": "Could not understand audio"}
    except Exception as e:
        return {"text": "", "language": "unknown", "success": False, "error": str(e)}
    finally:
        if converted and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

def transcribe_base64_audio(base64_data: str, primary_lang: str = "ur-PK") -> dict:
    """
    Decodes base64-encoded audio (WebM/WAV from browser Push-To-Talk button) and transcribes it.
    """
    if not base64_data or not base64_data.strip():
        return {"text": "", "language": "unknown", "success": False, "error": "Empty audio data"}

    import base64
    import tempfile

    raw_b64 = base64_data
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        audio_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        return {"text": "", "language": "unknown", "success": False, "error": f"Base64 decode failed: {e}"}

    suffix = ".webm"
    if "audio/mp4" in base64_data or "audio/m4a" in base64_data or "audio/aac" in base64_data:
        suffix = ".mp4"
    elif "audio/wav" in base64_data:
        suffix = ".wav"
    elif "audio/ogg" in base64_data:
        suffix = ".ogg"
    elif len(audio_bytes) > 8:
        if audio_bytes[:4] == b"RIFF":
            suffix = ".wav"
        elif audio_bytes[4:8] == b"ftyp":
            suffix = ".mp4"
        elif audio_bytes[:4] == b"\x1aE\xdf\xa3":
            suffix = ".webm"
        elif audio_bytes[:4] == b"OggS":
            suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        temp_file = tf.name
        tf.write(audio_bytes)

    try:
        res = transcribe_audio(temp_file, primary_lang=primary_lang)
        return res
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

if __name__ == "__main__":
    print("Voice Listener initialized and ready.")

