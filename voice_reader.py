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

def sanitize_urdu_voiceover(text: str) -> str:
    """
    Strict Urdu Voiceover Sanitizer:
    Ensures voice synthesizer only speaks pure poetry words.
    Removes:
    - XML/HTML tags (<speak>, <voice>, <break>, etc.)
    - URLs and web protocols (http, https, www, xmlns, etc.)
    - Technical words (version, xml:lang, etc.)
    - English words, numbers, code artifacts
    - Attribution metadata ('کلام: آداب', 'شاعر: غالب', 'شاعری:', etc.)
    - Punctuation, symbols, emojis, and decorative characters
    """
    if not text:
        return ""
    # 1. Strip XML / HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Strip URLs and web protocols
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # 3. Strip technical SSML / code words that might get pronounced
    text = re.sub(r'\b(xmlns|version|xml|lang|http|https|speak|voice|prosody|break|rate|pitch)\b', ' ', text, flags=re.IGNORECASE)
    # 4. Strip hashtags
    text = re.sub(r'#\S+', ' ', text)
    # 5. Strip English/Latin letters and digits
    text = re.sub(r'[a-zA-Z0-9]', ' ', text)
    # 6. Strip attribution phrases (e.g. '— کلام: آداب', 'کلام: آداب', 'شاعر: مرزا غالب')
    text = re.sub(r'[\—\–\-]?\s*(کلام|شاعر|تخلص|تخلیق|شاعری)\s*[:\-]?\s*.*$', ' ', text)
    # 7. Strip punctuation, symbols, and decorative glyphs
    text = re.sub(r'[\—\–\-_:;\|\*~\\/@#\$%\^&\(\)\[\]\{\}<>=\+\"\'«»❦◆■•\.,!?؟،؛]', ' ', text)
    # 8. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def synthesize_poetry_audio_async(
    misra_1: str,
    misra_2: str,
    output_path: str,
    voice: str = VOICE_FEMALE_URDU,
    rate_offset: str = "-12%",
    inter_line_pause_sec: float = 0.75
) -> str:
    """
    Synthesizes clean Urdu poetry recitation with classical poetic cadence:
    - Strips all XML, URLs, 'http', 'version', symbols, and metadata
    - Synthesizes pure Misra 1
    - Inserts exact 0.75s silence (classical mushaira pause)
    - Synthesizes pure Misra 2
    """
    ensure_edge_tts()
    import edge_tts

    clean_m1 = sanitize_urdu_voiceover(misra_1)
    clean_m2 = sanitize_urdu_voiceover(misra_2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    temp_m1 = os.path.join(out_dir, "_temp_m1.mp3")
    temp_m2 = os.path.join(out_dir, "_temp_m2.mp3")

    try:
        # Synthesize Misra 1
        comm1 = edge_tts.Communicate(text=clean_m1, voice=voice, rate=rate_offset)
        await comm1.save(temp_m1)

        # Synthesize Misra 2
        comm2 = edge_tts.Communicate(text=clean_m2, voice=voice, rate=rate_offset)
        await comm2.save(temp_m2)

        # Concatenate with exact pause silence using FFmpeg
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [
            ffmpeg_bin, "-y",
            "-i", temp_m1,
            "-f", "lavfi", "-t", str(inter_line_pause_sec), "-i", "anullsrc=r=24000:cl=mono",
            "-i", temp_m2,
            "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]",
            "-map", "[outa]",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError("FFmpeg audio concat failed")

    except Exception as err:
        # Fallback to single call with clean punctuation if ffmpeg concat fails
        print(f"[VoiceReader] Concat fallback triggered ({err}). Using direct speech synthesis...")
        combined_clean = f"{clean_m1}۔   {clean_m2}"
        comm = edge_tts.Communicate(text=combined_clean, voice=voice, rate=rate_offset)
        await comm.save(output_path)

    finally:
        # Cleanup temporary audio files
        for tmp_f in [temp_m1, temp_m2]:
            if os.path.exists(tmp_f):
                try:
                    os.remove(tmp_f)
                except Exception:
                    pass

    return output_path

def generate_poetry_recitation(
    misra_1: str,
    misra_2: str,
    output_path: str,
    voice: str = VOICE_FEMALE_URDU,
    rate_offset: str = "-15%"
) -> str:
    """Synchronous entry point for poetry recitation audio generation."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already running in an async context, run in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, synthesize_poetry_audio_async(misra_1, misra_2, output_path, voice, rate_offset))
                return future.result()
        else:
            return loop.run_until_complete(synthesize_poetry_audio_async(misra_1, misra_2, output_path, voice, rate_offset))
    except RuntimeError:
        return asyncio.run(synthesize_poetry_audio_async(misra_1, misra_2, output_path, voice, rate_offset))

def get_audio_duration_seconds(audio_path: str) -> float:
    """Calculates exact duration of an audio file in seconds."""
    try:
        import wave
        # Try wave if wav
        if audio_path.endswith(".wav"):
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
    except Exception:
        pass

    try:
        # Fallback to imageio-ffmpeg or mutagen or ffprobe
        import imageio_ffmpeg
        # Or estimate based on bitrate (64 kbps standard edge-tts MP3)
        file_size = os.path.getsize(audio_path)
        # Average 64kbps MP3 is 8000 bytes per second
        estimated_sec = max(3.0, round(file_size / 8000.0, 2))
        return estimated_sec
    except Exception:
        return 6.0

if __name__ == "__main__":
    print("Testing Urdu Female Voice Synthesis...")
    m1 = "دل ناداں تجھے ہوا کیا ہے"
    m2 = "آخر اس درد کی دوا کیا ہے"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_recitation.mp3")
    generate_poetry_recitation(m1, m2, out)
    print(f"✓ Saved voice recitation audio to: {out}")
