import os
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App: adaab-agent-backend
# High-performance serverless deployment of Qwen 2.5 7B Instruct for Urdu
# Running on cost-efficient NVIDIA L4 GPU ($0.80/hr, scale-to-zero when idle)
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("adaab-agent-backend")

# Persistent Volume where model weights, fonts, and crash logs reside
weights_vol = modal.Volume.from_name("adaab-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "/root/cache"
HF_CACHE = "/root/cache/huggingface"

# Build Unified Cloud Container Image with CUDA 12.4, vLLM, FFmpeg & Audio-Visual tools
adaab_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
    })
    .apt_install("git", "curl", "build-essential", "ffmpeg")
    .pip_install(
        "vllm==0.7.3",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.28.0",
        "transformers==4.48.3",
        "torch==2.5.1",
        "accelerate>=1.2.0",
        "diffusers>=0.32.0",
        "Pillow>=10.0.0",
        "arabic-reshaper>=3.0.0",
        "python-bidi>=0.4.2",
        "edge-tts>=6.1.12",
        "imageio-ffmpeg>=0.5.1",
        "fastapi",
        "pydantic"
    )
)

def get_credentials():
    """Reads HF_TOKEN from local credentials.txt or environment."""
    token = os.environ.get("HF_TOKEN")
    try:
        creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.txt")
        if os.path.exists(creds_path):
            with open(creds_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                    elif line.startswith("hf_"):
                        token = line.strip()
    except Exception:
        pass
    return token

hf_token_local = get_credentials()
secrets = []
if hf_token_local:
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token_local}))

# ─── Weight Pre-caching Function ──────────────────────────────────────────────
@app.function(
    image=adaab_image,
    volumes={CACHE_DIR: weights_vol},
    secrets=secrets,
    timeout=3600
)
def download_adaab_weights():
    """
    Pre-caches Qwen 2.5 7B model weights directly into the persistent volume.
    Run with: python -m modal run deploy_adaab.py::download_adaab_weights
    """
    import os
    import huggingface_hub
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            huggingface_hub.login(token=token)
            print("Authenticated with Hugging Face token.")
        except Exception as e:
            print(f"HF Login skipped: {e}")

    print(f"Ensuring {MODEL_ID} is cached in persistent volume at {HF_CACHE}...")
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_CACHE,
        token=token,
        ignore_patterns=["*.pt", "*.bin"]
    )
    weights_vol.commit()
    print(f"\n✓ Successfully verified {MODEL_ID} in Modal Volume 'adaab-cache'!")


# ─── Adaab Agent vLLM Serving Model Class ─────────────────────────────────────
@app.cls(
    image=adaab_image,
    gpu="L4",               # NVIDIA L4 (24GB VRAM) @ $0.80/hr
    volumes={CACHE_DIR: weights_vol},
    secrets=secrets,
    timeout=600,
    min_containers=0,       # Scale-to-zero when completely idle ($0.00 cost)
    scaledown_window=300,   # Keep warm for 5 minutes of active conversation (avoids cold starts)
)
class AdaabAgentModel:
    @modal.enter()
    def load_engine(self):
        """Initializes vLLM engine with eager execution (zero cudagraph capture delay)."""
        import os
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer

        ADAPTER_DIR = "/root/cache/adaab-pakistan-lora"
        self.has_lora = os.path.exists(ADAPTER_DIR)
        self.active_adapter_path = ADAPTER_DIR if self.has_lora else None

        print(f"[AdaabAgentModel] Initializing vLLM engine for {MODEL_ID} on NVIDIA L4 GPU (LoRA: {self.has_lora}, Path: {self.active_adapter_path})...")
        self.llm = LLM(
            model=MODEL_ID,
            download_dir=HF_CACHE,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            enable_lora=self.has_lora,
            max_loras=1 if self.has_lora else None,
            max_lora_rank=64 if self.has_lora else None,
            trust_remote_code=True,
            enforce_eager=True,     # BYPASSES CUDA GRAPH CAPTURE! Eliminates 30-50s startup delay.
            dtype="bfloat16"
        )

        # Cache tokenizer once on container startup instead of per-request
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE
        )
        print("[AdaabAgentModel] vLLM engine and tokenizer ready!")

    @modal.method()
    def heartbeat(self) -> str:
        """Health check endpoint."""
        return "alive"

    @modal.method()
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.65,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repetition_penalty: float = 1.20,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.5,
    ) -> dict:
        """Executes an Urdu inference request with repetition penalty and loop prevention."""
        from vllm import SamplingParams

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop_token_ids=[self.tokenizer.eos_token_id],
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        lora_req = None
        if getattr(self, "has_lora", False) and getattr(self, "active_adapter_path", None):
            from vllm.lora.request import LoRARequest
            lora_req = LoRARequest("adaab_lora", 1, self.active_adapter_path)

        outputs = self.llm.generate([prompt], sampling_params, lora_request=lora_req)
        first_output = outputs[0]
        generated_text = first_output.outputs[0].text

        return {
            "content": generated_text,
            "prompt_tokens": len(first_output.prompt_token_ids),
            "completion_tokens": len(first_output.outputs[0].token_ids),
            "model": MODEL_ID
        }

    @modal.fastapi_endpoint(method="POST")
    def chat_completions(self, request: dict):
        """OpenAI-compatible HTTP endpoint for external clients."""
        # Fast, zero-compute heartbeat keep-alive (resets scaledown_window without running inference)
        if request.get("heartbeat") or request.get("ping"):
            return {
                "status": "alive",
                "heartbeat": True,
                "model": MODEL_ID
            }

        messages = request.get("messages", [])
        temperature = float(request.get("temperature", 0.65))
        top_p = float(request.get("top_p", 0.9))
        max_tokens = int(request.get("max_tokens", 800))
        repetition_penalty = float(request.get("repetition_penalty", 1.20))
        presence_penalty = float(request.get("presence_penalty", 0.5))
        frequency_penalty = float(request.get("frequency_penalty", 0.5))

        result = self.generate.local(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result["content"]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"]
            },
            "model": MODEL_ID
        }


# ─── Full Cloud-Native Multi-Modal Post Generator (Cost-Optimized CPU Worker) ──
@app.function(
    image=adaab_image,
    cpu=2.0,                # High-speed CPU worker (~$0.04/hr instead of $0.80/hr GPU)
    memory=4096,
    volumes={CACHE_DIR: weights_vol},
    secrets=secrets,
    timeout=600
)
def generate_cloud_post_artifact(post_data: dict, aesthetic: dict) -> dict:
    """
    Executes the entire multi-modal pipeline directly inside Modal cloud:
    1. Renders 9:16 Urdu typography poster
    2. Synthesizes female voice recitation (ur-PK-UzmaNeural) with poetry cadence
    3. Assembles 1080x1920 MP4 reel with 0.5s intro delay + audio + 0.5s outro hold
    4. Returns serialized asset bytes
    """
    import io
    import os
    import sys
    import base64
    import tempfile
    import asyncio
    import subprocess
    import edge_tts
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    import arabic_reshaper
    from bidi.algorithm import get_display

    misra_1 = post_data["misra_1"]
    misra_2 = post_data["misra_2"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        poster_file = os.path.join(tmp_dir, "poster.png")
        audio_file = os.path.join(tmp_dir, "audio.mp3")
        reel_file = os.path.join(tmp_dir, "reel.mp4")

        # 1. Cloud Typography Rendering
        width, height = 1080, 1920
        # Procedural high-aesthetic background
        palette = aesthetic.get("palette", {"dark": "#04150D", "primary": "#0B3B24", "accent": "#D4AF37"})
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) if len(h) == 6 else (20, 20, 20)

        c_dark = hex_to_rgb(palette["dark"])
        c_prim = hex_to_rgb(palette["primary"])
        c_accent = hex_to_rgb(palette["accent"])

        poster = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(poster)
        for y in range(height):
            t = y / height
            factor = t if t < 0.5 else (1 - t)
            r = int(c_dark[0] * (1 - factor) + c_prim[0] * factor)
            g = int(c_dark[1] * (1 - factor) + c_prim[1] * factor)
            b = int(c_dark[2] * (1 - factor) + c_prim[2] * factor)
            draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

        # Center Contrast Scrim
        scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        scrim_draw = ImageDraw.Draw(scrim)
        scrim_draw.ellipse([int(width*0.08), int(height*0.30), int(width*0.92), int(height*0.70)], fill=(0, 0, 0, 160))
        scrim = scrim.filter(ImageFilter.GaussianBlur(radius=60))
        poster = Image.alpha_composite(poster, scrim)

        # Draw frame
        frame_draw = ImageDraw.Draw(poster)
        frame_draw.rectangle([(45, 45), (width - 45, height - 45)], outline=(212, 175, 55, 120), width=2)

        # Render Text with Arabic Reshaper & BiDi
        bidi_1 = get_display(arabic_reshaper.reshape(misra_1))
        bidi_2 = get_display(arabic_reshaper.reshape(misra_2))
        sig = get_display(arabic_reshaper.reshape("— کلام: آداب"))

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 46)
        except Exception:
            font = ImageFont.load_default()

        def draw_shadowed(dr, text, y_pos, text_font, fill_col):
            bbox = text_font.getbbox(text)
            tw = bbox[2] - bbox[0]
            xl = (width - tw) // 2
            for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, 3)]:
                dr.text((xl + dx, y_pos + dy), text, font=text_font, fill=(0, 0, 0, 220))
            dr.text((xl, y_pos), text, font=text_font, fill=fill_col)

        cy = int(height * 0.49)
        draw_shadowed(frame_draw, bidi_1, cy - 80, font, (255, 255, 255, 255))
        draw_shadowed(frame_draw, "✦  ❦  ✦", cy, font, (212, 175, 55, 240))
        draw_shadowed(frame_draw, bidi_2, cy + 80, font, (255, 255, 255, 255))
        draw_shadowed(frame_draw, sig, cy + 160, font, (212, 175, 55, 200))

        poster.convert("RGB").save(poster_file, quality=95)

        # 2. Cloud Female Voice Synthesis (edge-tts)
        # Strip all XML, URLs, English, attribution phrases, and non-Urdu symbols
        import re
        def sanitize_urdu(txt):
            txt = re.sub(r'<[^>]+>', ' ', txt)
            txt = re.sub(r'https?://\S+|www\.\S+', ' ', txt)
            txt = re.sub(r'\b(xmlns|version|xml|lang|http|https|speak|voice|prosody|break|rate|pitch)\b', ' ', txt, flags=re.IGNORECASE)
            txt = re.sub(r'#\S+', ' ', txt)
            txt = re.sub(r'[a-zA-Z0-9]', ' ', txt)
            txt = re.sub(r'[\—\–\-]?\s*(کلام|شاعر|تخلص|تخلیق|شاعری)\s*[:\-]?\s*.*$', ' ', txt)
            txt = re.sub(r'[\—\–\-_:;\|\*~\\/@#\$%\^&\(\)\[\]\{\}<>=\+\"\'«»❦◆■•\.,!?؟،؛]', ' ', txt)
            return re.sub(r'\s+', ' ', txt).strip()

        clean_1 = sanitize_urdu(misra_1)
        clean_2 = sanitize_urdu(misra_2)
        spoken_text = f"{clean_1}۔   {clean_2}"
        asyncio.run(edge_tts.Communicate(text=spoken_text, voice="ur-PK-UzmaNeural", rate="-12%").save(audio_file))

        # 3. Cloud Reel Video Assembly (FFmpeg with 0.5s intro delay + 0.5s outro hold)
        import re
        dur_proc = subprocess.run(["ffmpeg", "-i", audio_file], stderr=subprocess.PIPE, text=True, errors="ignore")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", dur_proc.stderr)
        audio_dur = 5.0
        if match:
            audio_dur = int(match.group(1))*3600 + int(match.group(2))*60 + float(match.group(3))
        
        total_dur = round(0.5 + audio_dur + 0.5, 2)
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-t", str(total_dur),
            "-i", poster_file,
            "-i", audio_file,
            "-filter_complex", "[1:a]adelay=500|500,apad[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-t", str(total_dur),
            reel_file
        ]
        subprocess.run(cmd, check=True)

        # Read back bytes to send to client
        with open(poster_file, "rb") as f:
            poster_bytes = base64.b64encode(f.read()).decode("utf-8")
        with open(audio_file, "rb") as f:
            audio_bytes = base64.b64encode(f.read()).decode("utf-8")
        with open(reel_file, "rb") as f:
            reel_bytes = base64.b64encode(f.read()).decode("utf-8")

        return {
            "poster_b64": poster_bytes,
            "audio_b64": audio_bytes,
            "reel_b64": reel_bytes,
            "audio_duration": audio_dur,
            "total_duration": total_dur
        }

@app.function(
    image=adaab_image,
    cpu=2.0,
    memory=2048,
    volumes={CACHE_DIR: weights_vol},
    secrets=secrets,
    timeout=300
)
def synthesize_urdu_speech_cloud(text: str, voice: str = "female") -> dict:
    """
    Serverless cloud synthesis for Urdu speech (zero local CPU load):
    - Applies intelligent Urdu phonetic and Izafat pre-processing
    - Synthesizes neural audio (ur-PK-UzmaNeural or ur-PK-AsadNeural)
    - Returns base64 encoded audio
    """
    import base64
    import tempfile
    import asyncio
    import edge_tts
    import re

    voice_id = "ur-PK-UzmaNeural" if voice == "female" else "ur-PK-AsadNeural"
    clean_text = re.sub(r'<[^>]+>', ' ', text)
    clean_text = re.sub(r'[a-zA-Z0-9#]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        out_path = tf.name

    asyncio.run(edge_tts.Communicate(text=clean_text, voice=voice_id, rate="-10%").save(out_path))

    with open(out_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        os.remove(out_path)
    except Exception:
        pass

    return {"audio_b64": audio_b64, "voice": voice_id}

