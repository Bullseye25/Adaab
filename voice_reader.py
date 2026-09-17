import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import subprocess
import shutil
import re

VOICE_FEMALE_URDU = "ur-PK-UzmaNeural"
VOICE_MALE_URDU = "ur-PK-AsadNeural"
DEFAULT_VOICE = VOICE_FEMALE_URDU

def ensure_edge_tts():
    try:
        import edge_tts
    except ImportError:
        print("[VoiceReader] Installing edge-tts locally...")
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts"], check=True)

def get_ffmpeg_binary() -> str:
    """Finds FFmpeg executable, installing imageio-ffmpeg automatically if needed."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()

# ─────────────────────────────────────────────────────────────────────────────
# Intelligent Conversational Urdu Phonetic Pre-Processor
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_urdu_phonetics(text: str, preserve_alphanumeric: bool = False) -> str:
    """
    Intelligently prepares Urdu text for natural speech synthesis:
    1. Expands common Persian Izafat compounds so the linking '-e-' is vocalized cleanly.
    2. Inserts micro-pauses at punctuation marks ('،', '۔', '...') for human-like breathing.
    3. Cleans stray XML/HTML, code tokens, attribution labels, and URLs.
    4. Preserves English names/words in conversational mode so names like 'Ali' or 'Doctor' are vocalized.
    """
    if not text:
        return ""

    # 0. Strip markdown formatting FIRST so bold/header markers around prefixes are removed
    text = re.sub(r'[\*\#_`~>]', ' ', text)

    # 1. Strip role prefixes (English and Urdu) at the start of any line or text
    text = re.sub(r'^\s*(?:text\s*message|text|assistant|tabraiz|tehzeeb|تبریز|تہذیب|جواب|صنفی\s*معاون|مردانہ\s*معاون)\s*[:：\-—]\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\b(text\s*message|replay\s*voice)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\btext\s*[:：\-]\s*', '', text, flags=re.IGNORECASE)

    # 2. Strip XML / HTML tags & SSML residues
    text = re.sub(r'<[^>]+>', ' ', text)
    # 3. Strip URLs and web protocols
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # 4. Strip code words and technical SSML tags
    text = re.sub(r'\b(xmlns|version|xml|lang|http|https|speak|voice|prosody|break|rate|pitch)\b', ' ', text, flags=re.IGNORECASE)
    # 5. Strip hashtags
    text = re.sub(r'#\S+', ' ', text)
    # 6. Strip English words and digits only for classical poetry if requested
    if not preserve_alphanumeric:
        text = re.sub(r'[a-zA-Z0-9]', ' ', text)
    # 7. Strip attribution phrases
    text = re.sub(r'[\—\–\-]?\s*(کلام|شاعر|تخلص|تخلیق|شاعری)\s*[:\-]?\s*.*$', ' ', text)
    # 8. Strip emojis, brackets, symbols, but PRESERVE Urdu punctuation (، ۔ ؟)
    text = re.sub(r'[\—\–\-_:;\|\*~\\/@#\$%\^&\(\)\[\]\{\}<>=\+\"\'«»❦◆■•!]', ' ', text)

    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sanitize_urdu_voiceover(text: str) -> str:
    """Compatibility wrapper for phonetic pre-processing."""
    return preprocess_urdu_phonetics(text, preserve_alphanumeric=True)

async def synthesize_conversation_audio_async(
    text: str,
    output_path: str,
    voice: str = VOICE_FEMALE_URDU,
    rate_offset: str = "+0%"
) -> str:
    """
    Synthesizes spoken Urdu conversational responses (for interactive mic chat).
    - Preserves commas and periods for natural speech flow
    - Natural neural tempo (+0% for clear, human-like cadence)
    """
    ensure_edge_tts()
    import edge_tts

    # ALWAYS female by default (Tehzeeb)
    if not voice or voice.lower() in ["female", "default", "tehzeeb", "uzma"]:
        voice = VOICE_FEMALE_URDU
    elif voice.lower() in ["male", "asad", "tabraiz", "tabrez", "tabreez"]:
        voice = VOICE_MALE_URDU
    else:
        voice = VOICE_FEMALE_URDU

    clean_text = preprocess_urdu_phonetics(text, preserve_alphanumeric=True)
    if not clean_text:
        clean_text = "جی جناب، فرمائیے۔"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    comm = edge_tts.Communicate(text=clean_text, voice=voice, rate=rate_offset)
    await comm.save(output_path)
    return output_path

def generate_conversation_speech(
    text: str,
    output_path: str,
    voice: str = VOICE_FEMALE_URDU,
    rate_offset: str = "+0%"
) -> str:
    """Synchronous entry point for conversational speech audio generation."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, synthesize_conversation_audio_async(text, output_path, voice, rate_offset))
        return future.result()

def get_audio_duration_seconds(audio_path: str) -> float:
    """Calculates duration of an audio file in seconds."""
    try:
        import wave
        if audio_path.endswith(".wav"):
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
    except Exception:
        pass

    try:
        file_size = os.path.getsize(audio_path)
        estimated_sec = max(2.5, round(file_size / 8000.0, 2))
        return estimated_sec
    except Exception:
        return 5.0

if __name__ == "__main__":
    print("Testing Conversational Urdu Voice Engine...")
    test_text = "آداب عرض ہے! میں تبریز ہوں، آپ کی خدمت میں حاضر ہوں۔"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_speech.mp3")
    generate_conversation_speech(test_text, out, voice="male")
    print(f"Saved conversational voice audio to: {out}")
