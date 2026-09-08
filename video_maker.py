import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import subprocess
import shutil

def get_ffmpeg_binary() -> str:
    """Finds FFmpeg executable, installing imageio-ffmpeg automatically if needed."""
    # Check if system ffmpeg exists
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("[VideoMaker] Installing imageio-ffmpeg for standalone video encoding...")
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()

def get_exact_media_duration(file_path: str, ffmpeg_bin: str) -> float:
    """Extracts exact audio/video duration using ffmpeg."""
    try:
        cmd = [ffmpeg_bin, "-i", file_path]
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, errors="ignore")
        # Look for Duration: 00:00:05.12
        import re
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", proc.stderr)
        if match:
            hours = int(match.group(1))
            mins = int(match.group(2))
            secs = float(match.group(3))
            return hours * 3600 + mins * 60 + secs
    except Exception:
        pass
    # Rough fallback estimate based on filesize
    return max(4.0, os.path.getsize(file_path) / 8000.0)

def assemble_poetry_reel(
    poster_path: str,
    audio_path: str,
    output_video_path: str,
    intro_delay_sec: float = 0.5,
    outro_hold_sec: float = 0.5,
    fps: int = 30
) -> str:
    """
    Assembles a high-definition 1080x1920 MP4 poetry reel:
    - 0.5s visual silence at start
    - Audio plays in full
    - 0.5s visual lingering hold after voiceover finishes
    - Total duration = 0.5 + audio_duration + 0.5
    - Subtle cinematic pan/zoom (Ken Burns effect)
    """
    ffmpeg_bin = get_ffmpeg_binary()
    audio_dur = get_exact_media_duration(audio_path, ffmpeg_bin)
    total_duration = round(intro_delay_sec + audio_dur + outro_hold_sec, 2)
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    print(f"[VideoMaker] Audio duration: {audio_dur:.2f}s | Intro: {intro_delay_sec}s | Outro: {outro_hold_sec}s")
    print(f"[VideoMaker] Rendering 1080x1920 reel (Total Duration: {total_duration:.2f}s)...")

    # FFmpeg filter:
    # 1. Image looped for total_duration with subtle smooth zoom (1.0 -> 1.03)
    # 2. Audio delayed by intro_delay_sec using adelay filter (delays both L and R channels by intro_delay_sec * 1000 ms)
    delay_ms = int(intro_delay_sec * 1000)
    
    # We use a robust filtergraph:
    # Video: zoompan filter for subtle cinematic motion
    filter_complex = (
        f"[0:v]scale=1080:1920,zoompan=z='min(zoom+0.0004,1.03)':d={int(total_duration * fps)}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps={fps}[v];"
        f"[1:a]adelay={delay_ms}|{delay_ms},apad[a]"
    )

    cmd = [
        ffmpeg_bin,
        "-y",                             # Overwrite output
        "-loop", "1",
        "-t", str(total_duration),
        "-i", poster_path,
        "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "stillimage",
        "-threads", "0",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", str(total_duration),
        output_video_path
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"✓ Successfully generated 1080x1920 MP4 Video Reel at: {output_video_path}")
        return output_video_path
    except subprocess.CalledProcessError as e:
        print(f"[VideoMaker] Primary filter failed ({e.stderr[:200] if e.stderr else ''}), attempting fallback encoding...")
        # Simpler fallback encoding without zoompan
        fallback_filter = f"[1:a]adelay={delay_ms}|{delay_ms},apad[a]"
        cmd_fallback = [
            ffmpeg_bin,
            "-y",
            "-loop", "1",
            "-t", str(total_duration),
            "-i", poster_path,
            "-i", audio_path,
            "-filter_complex", fallback_filter,
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-t", str(total_duration),
            output_video_path
        ]
        subprocess.run(cmd_fallback, check=True)
        print(f"✓ Generated Video Reel via fallback encoder at: {output_video_path}")
        return output_video_path

if __name__ == "__main__":
    print("Testing Video Reel Maker...")
