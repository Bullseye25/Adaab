"""
gemini_oracle.py
─────────────────────────────────────────────────────────────────────────────
Knowledge Oracle & Search Grounding with Rate-Limit Circuit Breaker
─────────────────────────────────────────────────────────────────────────────
• Uses Google Gemini (gemini-flash-latest / gemini-3.6-flash) as Knowledge Oracle.
• Integrates DuckDuckGo real-time web search for live context.
• Implements Rate-Limiting & Quota Circuit Breaker:
  - Rolling sliding window (max 10 RPM) to prevent quota exhaustion.
  - Automatically detects HTTP 429 (ResourceExhausted / Quota Limit) and 503 errors.
  - Immediately trips the circuit breaker: ceases all Gemini calls for a cooldown period.
  - Returns None gracefully, allowing seamless fallback to Modal GPU (Qwen 2.5).
"""

import os
import sys
import time
import re
import requests
from typing import Optional, List, Dict, Any

# Primary and resilient fallback Gemini models
PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODEL = "gemini-3-flash-preview"
TERTIARY_MODEL = "gemini-3.1-flash-lite"



class GeminiSearchOracle:
    def __init__(
        self,
        credentials_file: str = "H:/Adaab/credentials.txt",
        max_rpm: int = 10,
        cooldown_seconds: int = 600  # 10 minute cooldown when limit is reached
    ):
        self.credentials_file = credentials_file
        self.max_rpm = max_rpm
        self.cooldown_seconds = cooldown_seconds
        self.api_key: Optional[str] = self._load_api_key()
        
        # Rate-limiting state (sliding window)
        self._request_timestamps: List[float] = []
        
        # Circuit Breaker state
        self._circuit_open: bool = False
        self._circuit_tripped_time: float = 0.0
        self._circuit_reason: str = ""

    def _load_api_key(self) -> Optional[str]:
        """Loads GEMINI_API_KEY from environment or credentials.txt."""
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key.strip()

        if os.path.exists(self.credentials_file):
            try:
                with open(self.credentials_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip()
            except Exception as e:
                print(f"[GeminiOracle] Notice reading credentials: {e}")
        return None

    def is_available(self) -> bool:
        """Checks if the Gemini API is configured, healthy, and not rate-limited."""
        if not self.api_key:
            return False

        now = time.time()

        # Check Circuit Breaker
        if self._circuit_open:
            elapsed = now - self._circuit_tripped_time
            if elapsed > self.cooldown_seconds:
                # Enter Half-Open state (allow 1 trial request)
                print("[GeminiOracle] Cooldown period elapsed. Entering half-open trial state.")
                return True
            return False

        # Check Sliding RPM window
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60.0]
        if len(self._request_timestamps) >= self.max_rpm:
            print(f"[GeminiOracle] Sliding RPM limit ({self.max_rpm}/min) reached. Pausing Gemini calls.")
            return False

        return True

    def trip_circuit(self, reason: str):
        """Trips the circuit breaker to cease all Gemini calls immediately."""
        self._circuit_open = True
        self._circuit_tripped_time = time.time()
        self._circuit_reason = reason
        print(f"[GeminiOracle] CIRCUIT BREAKER TRIPPED! Reason: {reason}.")
        print(f"[GeminiOracle] As requested, ceasing all Gemini calls for {self.cooldown_seconds}s. Falling back to Modal GPU.")

    def reset_circuit(self):
        """Resets the circuit breaker to closed (normal operation)."""
        self._circuit_open = False
        self._circuit_tripped_time = 0.0
        self._circuit_reason = ""

    def get_circuit_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the circuit breaker."""
        now = time.time()
        return {
            "configured": bool(self.api_key),
            "circuit_open": self._circuit_open,
            "seconds_remaining_in_cooldown": max(0, int(self.cooldown_seconds - (now - self._circuit_tripped_time))) if self._circuit_open else 0,
            "tripped_reason": self._circuit_reason,
            "rpm_count_last_minute": len([t for t in self._request_timestamps if now - t < 60.0])
        }

    def search_web(self, query: str, max_results: int = 3) -> List[str]:
        """
        Lightweight, resilient real-time web search via DuckDuckGo HTML.
        Requires zero external libraries; returns clean snippet strings.
        """
        if not query:
            return []
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = "https://html.duckduckgo.com/html/"
        try:
            resp = requests.post(url, data={"q": query}, headers=headers, timeout=4)
            if resp.status_code != 200:
                return []
            
            snippets: List[str] = []
            matches = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            for m in matches[:max_results]:
                clean = re.sub(r'<[^>]+>', '', m).strip()
                clean = clean.replace('&quot;', '"').replace('&#x27;', "'").replace('&amp;', '&').replace('&nbsp;', ' ')
                if clean:
                    snippets.append(clean)
            return snippets
        except Exception as e:
            print(f"[GeminiOracle] Web search notice: {e}")
            return []

    def query(
        self,
        prompt: str,
        search_context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        timeout: int = 12
    ) -> Optional[str]:
        """
        Queries Gemini with rate-limit protection and circuit breaker handling.
        Returns the text response or None if rate-limited, quota exceeded, or timed out.
        """
        if not self.is_available():
            return None

        now = time.time()
        self._request_timestamps.append(now)

        # Prepare payload
        full_text = prompt
        if search_context:
            full_text = f"Search Context:\n{search_context}\n\nTask/Question:\n{prompt}"

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {
                "temperature": 0.35,
                "maxOutputTokens": 1500,
                "thinkingConfig": {
                    "thinkingBudget": 0
                }
            }
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}


        models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL, TERTIARY_MODEL]

        for model_id in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"
            try:
                resp = requests.post(url, json=payload, timeout=timeout)

                # Check 429 Quota / Rate-limit
                if resp.status_code == 429:
                    print(f"[GeminiOracle] Model {model_id} quota/rate-limited (HTTP 429). Trying fallback model...")
                    continue

                # Check 503 High Demand / Spikes
                if resp.status_code == 503:
                    print(f"[GeminiOracle] Model {model_id} high demand (503). Trying fallback...")
                    continue

                if resp.status_code != 200:
                    err_msg = resp.text[:200]
                    print(f"[GeminiOracle] API Error {resp.status_code} on {model_id}: {err_msg}")
                    continue

                # 200 OK -> parse candidate
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if "text" in p]
                reply = "\n".join(texts).strip()

                if reply:
                    # Successful response: reset circuit if was half-open
                    if self._circuit_open:
                        self.reset_circuit()
                    return reply

            except requests.exceptions.Timeout:
                print(f"[GeminiOracle] Request timed out on {model_id} after {timeout}s.")
                continue
            except requests.exceptions.RequestException as req_err:
                print(f"[GeminiOracle] Network error on {model_id}: {req_err}")
                continue
            except Exception as ex:
                print(f"[GeminiOracle] Unexpected error on {model_id}: {ex}")
                continue

        # All models failed or exhausted
        self.trip_circuit("All available Gemini models exhausted or rate-limited")

        # Check if ChatGPT Web Oracle fallback is available
        try:
            from chatgpt_browser_oracle import get_chatgpt_oracle
            chatgpt = get_chatgpt_oracle()
            if chatgpt.is_available():
                print("[GeminiOracle] Gemini quota exhausted. Seamlessly switching to ChatGPT Web Oracle fallback...")
                chatgpt_reply = chatgpt.query(
                    prompt=prompt,
                    search_context=search_context,
                    system_instruction=system_instruction
                )
                if chatgpt_reply:
                    print("[GeminiOracle] Successfully retrieved answer via ChatGPT Web Oracle fallback!")
                    return chatgpt_reply
        except Exception as fb_err:
            print(f"[GeminiOracle] ChatGPT fallback notice: {fb_err}")

        return None


# Global singleton instance
_oracle_instance: Optional[GeminiSearchOracle] = None

def get_gemini_oracle() -> GeminiSearchOracle:
    """Returns the singleton GeminiSearchOracle instance."""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = GeminiSearchOracle()
    return _oracle_instance
