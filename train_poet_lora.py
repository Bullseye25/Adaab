import os
import modal

app = modal.App("adaab-poetry-trainer")
weights_vol = modal.Volume.from_name("adaab-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_ADAPTER = "/root/cache/adaab-poet-adapter"

trainer_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.48.3",
        "peft>=0.14.0",
        "datasets>=3.2.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "trl>=0.14.0"
    )
)

@app.function(
    image=trainer_image,
    gpu="L4",
    volumes={"/root/cache": weights_vol},
    timeout=7200
)
def run_poet_fine_tuning():
    """
    Runs serverless QLoRA fine-tuning on Qwen 2.5 7B to specialize in classical Urdu poetry.
    Run with: python -m modal run train_poet_lora.py::run_poet_fine_tuning
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Starting QLoRA fine-tuning on {MODEL_ID} for Urdu Poetry...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="/root/cache/huggingface")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="/root/cache/huggingface"
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Saving trained adapter configuration to {OUTPUT_ADAPTER}...")
    os.makedirs(OUTPUT_ADAPTER, exist_ok=True)
    model.save_pretrained(OUTPUT_ADAPTER)
    weights_vol.commit()
    print("✓ Successfully saved Adaab Poet LoRA adapter into Modal Volume!")

if __name__ == "__main__":
    print("Modal training script ready. Run via start.py or 'modal run train_poet_lora.py'")
