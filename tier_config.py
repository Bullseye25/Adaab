# -*- coding: utf-8 -*-
import os
import json
from typing import List, Dict

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_config.json")

VALID_ENGINES = {
    "modal": {
        "name": "Modal Cloud GPU",
        "model": "Qwen 2.5 7B / LoRA",
        "device": "NVIDIA L4 GPU",
        "description": "Serverless Modal GPU running Qwen 2.5 with fine-tuned LoRA"
    },
    "ollama": {
        "name": "Ollama Cloud Oracle",
        "model": "gemma4:31b",
        "device": "Ollama Cloud",
        "description": "31-Billion parameter high-capacity conversational Oracle"
    },
    "gemini": {
        "name": "Google Gemini Oracle",
        "model": "gemini-2.5-flash",
        "device": "Google AI Cloud",
        "description": "Fast multimodal Google AI Studio Oracle"
    },
    "none": {
        "name": "Disabled",
        "model": "None",
        "device": "N/A",
        "description": "Tier disabled"
    }
}

DEFAULT_CONFIG = {
    "tier1": "modal",
    "tier2": "ollama",
    "tier3": "gemini"
}

PRESETS = {
    "1": {
        "title": "Modal GPU First (Qwen 2.5 / LoRA -> Ollama -> Gemini) [Recommended for Qwen Testing]",
        "tiers": ("modal", "ollama", "gemini")
    },
    "2": {
        "title": "Ollama Cloud First (Gemma 31B -> Gemini -> Modal GPU) [Maximum 31B Reasoning]",
        "tiers": ("ollama", "gemini", "modal")
    },
    "3": {
        "title": "Gemini Cloud First (Gemini 2.5 Flash -> Modal GPU -> Ollama)",
        "tiers": ("gemini", "modal", "ollama")
    },
    "4": {
        "title": "Modal GPU Only (Strict Qwen Evaluation with Zero Cloud Fallback)",
        "tiers": ("modal", "none", "none")
    }
}

def load_tier_config() -> Dict[str, str]:
    cfg = dict(DEFAULT_CONFIG)
    t1_env = os.environ.get("ADAAB_TIER1")
    t2_env = os.environ.get("ADAAB_TIER2")
    t3_env = os.environ.get("ADAAB_TIER3")
    if t1_env and t1_env.lower() in VALID_ENGINES:
        cfg["tier1"] = t1_env.lower()
    if t2_env and t2_env.lower() in VALID_ENGINES:
        cfg["tier2"] = t2_env.lower()
    if t3_env and t3_env.lower() in VALID_ENGINES:
        cfg["tier3"] = t3_env.lower()

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k in ["tier1", "tier2", "tier3"]:
                    val = str(saved.get(k, "")).lower()
                    if val in VALID_ENGINES:
                        cfg[k] = val
        except Exception as e:
            print(f"[TierConfig] Notice reading config file: {e}")
    return cfg

def save_tier_config(tier1: str, tier2: str, tier3: str) -> Dict[str, str]:
    t1 = tier1.strip().lower() if tier1 else "modal"
    t2 = tier2.strip().lower() if tier2 else "ollama"
    t3 = tier3.strip().lower() if tier3 else "gemini"
    if t1 not in VALID_ENGINES:
        t1 = "modal"
    if t2 not in VALID_ENGINES:
        t2 = "ollama"
    if t3 not in VALID_ENGINES:
        t3 = "gemini"
    cfg = {"tier1": t1, "tier2": t2, "tier3": t3}
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as ex:
        print(f"[TierConfig] Error saving config file: {ex}")
    os.environ["ADAAB_TIER1"] = t1
    os.environ["ADAAB_TIER2"] = t2
    os.environ["ADAAB_TIER3"] = t3
    return cfg

def get_active_tiers(cfg: Dict[str, str] = None) -> List[str]:
    if cfg is None:
        cfg = load_tier_config()
    tiers = []
    for k in ["tier1", "tier2", "tier3"]:
        eng = cfg.get(k, "none").lower()
        if eng in VALID_ENGINES and eng != "none":
            tiers.append(eng)
    return tiers or ["modal"]

def format_tier_summary(cfg: Dict[str, str] = None) -> str:
    if cfg is None:
        cfg = load_tier_config()
    parts = []
    for k in ["tier1", "tier2", "tier3"]:
        eng = cfg.get(k, "none")
        meta = VALID_ENGINES.get(eng, {})
        label = f"{meta.get('name', eng)} ({meta.get('model', '')})" if eng != "none" else "Disabled"
        idx = k.replace("tier", "Tier ")
        parts.append(f"{idx}: {label}")
    return " -> ".join(parts)
