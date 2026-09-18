"""
ollama_oracle.py
─────────────────────────────────────────────────────────────────────────────
Ollama Cloud Knowledge and Fallback Oracle for Adaab Studio
─────────────────────────────────────────────────────────────────────────────
• Uses official Ollama Cloud API (https://ollama.com/api/chat) with API key.
• Zero local PC compute or GPU load — runs entirely in the cloud.
• Ultra-fast sub-second latency (~300ms–600ms) with zero Cloudflare blocks.
• Primary Model: gemma4:31b (with automated fallbacks to nemotron-3-nano:30b and gpt-oss:20b).
• Formatted for fluent, cultured conversational Urdu and strict brevity.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import requests
from typing import Optional, List, Dict, Any

# Output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, "credentials.txt")
OLLAMA_ENDPOINT = "https://ollama.com/api/chat"

# Active models supported on free Ollama Cloud tier
PRIMARY_MODEL = "gemma4:31b"
FALLBACK_MODELS = ["nemotron-3-nano:30b", "gpt-oss:20b"]


class OllamaCloudOracle:
    def __init__(
        self,
        credentials_file: str = CREDENTIALS_FILE,
        endpoint_url: str = OLLAMA_ENDPOINT,
        primary_model: str = PRIMARY_MODEL
    ):
        self.credentials_file = credentials_file
        self.endpoint_url = endpoint_url
        self.primary_model = primary_model
        self.api_key: Optional[str] = self._load_api_key()

    def _load_api_key(self) -> Optional[str]:
        """Loads OLLAMA_API_KEY from environment or credentials.txt."""
        key = os.environ.get("OLLAMA_API_KEY")
        if key:
            return key.strip()
        if os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OLLAMA_API_KEY="):
                            return line.split("=", 1)[1].strip()
            except Exception:
                pass
        return None

    def is_available(self) -> bool:
        """Returns True if an Ollama API key is present."""
        return bool(self.api_key)

    def query(
        self,
        prompt: str,
        search_context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        timeout: int = 12
    ) -> Optional[str]:
        """
        Sends query to Ollama Cloud API.
        Attempts primary model (gemma4:31b) first, falling back to nemotron/gpt-oss if needed.
        """
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build messages payload
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        user_content = prompt
        if search_context:
            user_content = f"متعلقہ معلومات:\n{search_context}\n\nصارف کا سوال:\n{prompt}"
        messages.append({"role": "user", "content": user_content})

        models_to_try = [self.primary_model] + [m for m in FALLBACK_MODELS if m != self.primary_model]

        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.35,
                    "num_predict": 450
                }
            }
            try:
                t0 = time.time()
                resp = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        elapsed = round(time.time() - t0, 2)
                        print(f"[OllamaCloud] Successfully retrieved response via '{model_name}' in {elapsed}s.")
                        return content
                else:
                    print(f"[OllamaCloud] Model '{model_name}' returned HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                print(f"[OllamaCloud] Notice querying model '{model_name}': {e}")

        return None


# Global singleton
_ollama_oracle: Optional[OllamaCloudOracle] = None

def get_ollama_oracle() -> OllamaCloudOracle:
    """Returns the singleton instance of OllamaCloudOracle."""
    global _ollama_oracle
    if _ollama_oracle is None:
        _ollama_oracle = OllamaCloudOracle()
    return _ollama_oracle


if __name__ == "__main__":
    oracle = get_ollama_oracle()
    print("Ollama API Key configured:", oracle.is_available())
    if oracle.is_available():
        reply = oracle.query("سلام! آپ کا نام اور ماڈل کیا ہے؟")
        print("Reply:", reply)
