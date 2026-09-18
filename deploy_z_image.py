import os
import io
import time
import uuid
import base64
import random
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal App: adaab-z-image-turbo
# Serverless deployment of Tongyi-MAI/Z-Image-Turbo on NVIDIA L4 GPU
# Generates high-fidelity 1024x1024 images in 8-9 steps (Guidance 0.0)
# Permanently persists generated images to Modal Volume in 'adaab_images' folder
# Scale-to-zero when idle ($0.00 compute cost)
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("adaab-z-image-turbo")

# Persistent Modal Volume for all Adaab generated images
images_vol = modal.Volume.from_name("adaab-images", create_if_missing=True)

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
OUTPUT_DIR = "/root/output/adaab_images"

# Build optimized container image with latest diffusers and PyTorch
z_image_container = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "transformers>=4.48.0",
        "git+https://github.com/huggingface/diffusers",
        "accelerate>=1.2.0",
        "sentencepiece",
        "protobuf",
        "Pillow>=10.0.0",
        "fastapi",
        "pydantic"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
    })
)

def get_credentials():
    """Reads HF_TOKEN from local credentials.txt or environment if present."""
    token = os.environ.get("HF_TOKEN")
    try:
        creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.txt")
        if os.path.exists(creds_path):
            with open(creds_path, "r", encoding="utf-8") as f:
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


@app.cls(
    image=z_image_container,
    gpu="L4",                # NVIDIA L4 (24GB VRAM) @ $0.80/hr
    volumes={"/root/output": images_vol},
    secrets=secrets,
    timeout=600,
    min_containers=0,        # Scale-to-zero when idle ($0.00 cost)
    scaledown_window=180,    # Warm window: 3 minutes
)
class ZImageTurboModel:
    @modal.enter()
    def load_pipeline(self):
        """Initializes the Z-Image-Turbo diffusion pipeline once on container startup."""
        import torch
        from diffusers import DiffusionPipeline

        print(f"[ZImageTurboModel] Loading {MODEL_ID} pipeline into NVIDIA L4 GPU...")
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False
        )
        self.pipe.to("cuda")
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass
        print("[ZImageTurboModel] Pipeline ready for ultra-fast turbo inference!")

    @modal.method()
    def heartbeat(self) -> dict:
        """Health-check endpoint."""
        return {"status": "alive", "model": MODEL_ID}

    def _generate_core(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        seed: int = None
    ) -> dict:
        """
        Executes turbo image generation and persists the result to Modal Volume
        under the dedicated '/root/output/adaab_images' folder.
        """
        import torch
        from PIL import Image

        # Ensure dedicated output folder exists in Modal volume
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Seed configuration
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator("cuda").manual_seed(int(seed))

        print(f"[ZImageTurboModel] Generating image for prompt: '{prompt[:70]}...' (Seed: {seed})")
        start_time = time.time()

        # Run turbo inference with inference_mode (guidance_scale=0.0 as required by turbo distillation)
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=0.0,
                generator=generator
            ).images[0]

        duration = round(time.time() - start_time, 2)
        print(f"[ZImageTurboModel] Image generated successfully in {duration}s!")

        # 1. Save to dedicated persistent Modal Volume folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        filename = f"adaab_img_{timestamp}_{uid}.png"
        cloud_path = os.path.join(OUTPUT_DIR, filename)

        image.save(cloud_path, format="PNG")
        images_vol.commit()  # Commit to cloud storage immediately
        print(f"[ZImageTurboModel] Persisted to Modal Volume at: {cloud_path}")

        # 2. Encode to base64 data URI for instant in-chat rendering
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        b64_str = base64.b64encode(buffered.getvalue()).decode("ascii")
        data_uri = f"data:image/png;base64,{b64_str}"

        return {
            "status": "success",
            "image_base64": data_uri,
            "filename": filename,
            "modal_storage_path": cloud_path,
            "folder": "adaab_images",
            "prompt": prompt,
            "seed": seed,
            "generation_time_sec": duration
        }

    @modal.method()
    def generate(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        seed: int = None
    ) -> dict:
        """Modal SDK method."""
        return self._generate_core(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed
        )

    @modal.fastapi_endpoint(method="POST")
    def generate_endpoint(self, request: dict):
        """Open HTTP POST endpoint for external or local client requests."""
        if request.get("heartbeat") or request.get("ping"):
            return {"status": "alive", "model": MODEL_ID}

        prompt = request.get("prompt", "").strip()
        if not prompt:
            return {"status": "error", "message": "Prompt cannot be empty"}

        height = int(request.get("height", 1024))
        width = int(request.get("width", 1024))
        steps = int(request.get("num_inference_steps", 9))
        seed = request.get("seed")

        return self._generate_core(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            seed=seed
        )


# Standalone test runner for CLI verification
@app.function(
    image=z_image_container,
    gpu="L4",
    volumes={"/root/output": images_vol},
    secrets=secrets,
    timeout=600
)
def test_generate(prompt: str = "A majestic royal Mughal palace courtyard in Lahore at golden hour, marble fountains, intricate Islamic geometric arches, cinematic lighting, photorealistic 8k"):
    """CLI test function: python -m modal run deploy_z_image.py::test_generate"""
    model = ZImageTurboModel()
    model.load_pipeline()
    result = model.generate(prompt=prompt)
    print("Test Image Generation Result:", {
        "status": result["status"],
        "filename": result["filename"],
        "modal_storage_path": result["modal_storage_path"],
        "folder": result["folder"],
        "duration": result["generation_time_sec"]
    })
    return result
