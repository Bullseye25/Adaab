"""
train_qwen_pakistan_lora.py
─────────────────────────────────────────────────────────────────────────────
Qwen 2.5 7B Instruct LoRA Fine-Tuning Recipe for Adaab Studio
─────────────────────────────────────────────────────────────────────────────
Fine-tunes Qwen 2.5 7B on the generated 50+ Pakistani Knowledge & Cultural
conversation dataset (ChatML format) using Parameter-Efficient Fine-Tuning (PEFT)
and 4-bit QLoRA.

Supports two execution modes:
  1. Modal Cloud GPU (Serverless NVIDIA L4 / A10G):
     Command: modal run train_qwen_pakistan_lora.py
  2. Local GPU (NVIDIA CUDA):
     Command: python train_qwen_pakistan_lora.py --local
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import json
import argparse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MASTER_DATASET_PATH = os.path.join(DATA_DIR, "master_training_dataset.jsonl")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "pakistan_knowledge_dataset.jsonl")
DATASET_PATH = MASTER_DATASET_PATH if os.path.exists(MASTER_DATASET_PATH) else DEFAULT_DATASET_PATH
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "qwen2.5_adaab_lora")
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Modal Cloud Training App Definition (NVIDIA L4 Scale-to-Zero)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import modal
    modal_available = True
except ImportError:
    modal_available = False

if modal_available:
    app = modal.App("adaab-qwen-finetuner")

    def get_credentials():
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
    modal_secrets = []
    if hf_token_local:
        modal_secrets.append(modal.Secret.from_dict({"HF_TOKEN": hf_token_local}))

    train_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2.0",
            "transformers>=4.44.0",
            "peft>=0.12.0",
            "trl>=0.9.6",
            "bitsandbytes>=0.43.0",
            "datasets>=2.20.0",
            "accelerate>=0.33.0"
        )
    )

    volume = modal.Volume.from_name("adaab-cache", create_if_missing=True)

    @app.function(
        image=train_image,
        gpu="L4",
        timeout=3600,
        volumes={"/root/cache": volume},
        secrets=modal_secrets
    )
    def train_qwen_modal(dataset_jsonl_content: str):
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import Dataset

        print("[Modal GPU] Initializing Qwen 2.5 7B QLoRA Fine-Tuning on NVIDIA L4...")

        # Parse in-memory dataset lines
        lines = [json.loads(line) for line in dataset_jsonl_content.strip().split("\n") if line.strip()]
        print(f"[Modal GPU] Loaded {len(lines)} training dialogue pairs.")

        dataset = Dataset.from_list(lines)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        HF_CACHE = "/root/cache/huggingface"
        hf_token = os.environ.get("HF_TOKEN")

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            cache_dir=HF_CACHE,
            token=hf_token,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            cache_dir=HF_CACHE,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        try:
            from trl import SFTConfig
            ConfigClass = SFTConfig
        except ImportError:
            ConfigClass = TrainingArguments

        training_args = ConfigClass(
            output_dir="/root/weights/qwen_lora_checkpoints",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
            optim="paged_adamw_8bit"
        )

        def format_chatml(example):
            formatted_texts = []
            for msg_list in example["messages"]:
                formatted = tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=False)
                formatted_texts.append(formatted)
            return {"text": formatted_texts}

        dataset = dataset.map(format_chatml, batched=True)

        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset,
            "peft_config": lora_config,
            "args": training_args,
        }
        if hasattr(training_args, "dataset_text_field"):
            training_args.dataset_text_field = "text"
            training_args.max_seq_length = 1024
        else:
            trainer_kwargs["dataset_text_field"] = "text"
            trainer_kwargs["max_seq_length"] = 1024

        trainer = SFTTrainer(**trainer_kwargs)

        print("[Modal GPU] Starting training...")
        trainer.train()

        final_adapter_path = "/root/cache/adaab-pakistan-lora"
        trainer.model.save_pretrained(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        volume.commit()
        print(f"[Modal GPU] LoRA Adapter saved to Modal Volume 'adaab-cache' at '{final_adapter_path}' successfully!")

    @app.local_entrypoint()
    def main():
        # Automatically compile and refresh master dataset if dataset_manager is available
        target_file = DATASET_PATH
        try:
            from dataset_manager import build_master_dataset
            target_file = build_master_dataset()
        except Exception:
            pass

        if not os.path.exists(target_file):
            print(f"Error: Dataset file not found at {target_file}.")
            return
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
        print("Submitting QLoRA fine-tuning job to Modal Cloud (NVIDIA L4) using " + str(target_file) + "...")
        train_qwen_modal.remote(content)
        print("[SUCCESS] Training completed successfully and adapter committed to Modal Volume 'adaab-cache'!")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Local Training Routine
# ─────────────────────────────────────────────────────────────────────────────
def run_local_training():
    print("=" * 70)
    print("==== Local Qwen 2.5 7B QLoRA Fine-Tuning ====")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at {DATASET_PATH}. Please run test_pakistan_knowledge_50.py first!")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        print(f"Required training packages missing: {e}")
        print("Install via: pip install torch transformers peft trl bitsandbytes datasets accelerate")
        return

    if not torch.cuda.is_available():
        print("Notice: CUDA GPU not detected locally. Qwen 2.5 7B fine-tuning requires an NVIDIA GPU (min 12GB VRAM).")
        print("Recommended: Run training on Modal cloud using: modal run train_qwen_pakistan_lora.py")
        return

    print(f"Loading base model: {BASE_MODEL_NAME} (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    raw_dataset = load_dataset("json", data_files=DATASET_PATH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=raw_dataset["train"],
        peft_config=lora_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args
    )

    print("Starting local training...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Local LoRA adapter saved to {OUTPUT_DIR} successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 7B on Pakistan Knowledge Dataset")
    parser.add_argument("--local", action="store_true", help="Run local GPU training instead of Modal")
    args = parser.parse_args()

    if args.local or not modal_available:
        run_local_training()
    else:
        print("To run on Modal cloud GPU, execute: modal run train_qwen_pakistan_lora.py")
        print("To run locally with NVIDIA GPU, execute: python train_qwen_pakistan_lora.py --local")
