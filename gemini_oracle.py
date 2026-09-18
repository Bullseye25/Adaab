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

    def is_gemini_healthy(self) -> bool:
        """Checks if Gemini specifically is configured, healthy, and not rate-limited."""
        if not self.api_key:
            return False

        now = time.time()

        # Check Circuit Breaker
        if self._circuit_open:
            elapsed = now - self._circuit_tripped_time
            if elapsed > self.cooldown_seconds:
                # Enter Half-Open state (allow 1 trial request)
                print("[GeminiOracle] Cooldown period elapsed. Entering half-open trial state for Gemini.")
                return True
            return False

        # Check Sliding RPM window
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60.0]
        if len(self._request_timestamps) >= self.max_rpm:
            return False

        return True

    def is_available(self) -> bool:
        """Returns True if either Gemini is healthy or ChatGPT fallback is available."""
        if self.is_gemini_healthy():
            return True
        try:
            from chatgpt_browser_oracle import get_chatgpt_oracle
            return get_chatgpt_oracle().is_available()
        except Exception:
            return False

    def trip_circuit(self, reason: str):
        """Trips the circuit breaker to cease all Gemini calls immediately."""
        self._circuit_open = True
        self._circuit_tripped_time = time.time()
        self._circuit_reason = reason
        print(f"[GeminiOracle] CIRCUIT BREAKER TRIPPED! Reason: {reason}.")
        print(f"[GeminiOracle] Ceasing Gemini calls for {self.cooldown_seconds}s. Routing all queries directly to ChatGPT.")

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

    def _query_chatgpt_fallback(
        self,
        prompt: str,
        search_context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        timeout: int = 25
    ) -> Optional[str]:
        """Seamlessly queries ChatGPT when Gemini is unavailable, rate-limited, or slow."""
        try:
            from chatgpt_browser_oracle import get_chatgpt_oracle
            chatgpt = get_chatgpt_oracle()
            if chatgpt.is_available():
                print("[GeminiOracle] Instantly switching to ChatGPT Oracle...")
                t0 = time.time()
                reply = chatgpt.query(
                    prompt=prompt,
                    search_context=search_context,
                    system_instruction=system_instruction,
                    timeout=timeout
                )
                if reply:
                    elapsed = round(time.time() - t0, 1)
                    print(f"[GeminiOracle] Successfully retrieved response via ChatGPT in {elapsed}s!")
                    return reply
                else:
                    print("[GeminiOracle] ChatGPT returned empty response.")
            else:
                print("[GeminiOracle] ChatGPT is not available.")
        except Exception as fb_err:
            print(f"[GeminiOracle] ChatGPT fallback notice: {fb_err}")
        return None

    def query(
        self,
        prompt: str,
        search_context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        timeout: int = 5
    ) -> Optional[str]:
        """
        Queries Gemini with fast 5s timeout and auto-failover to ChatGPT.
        If Gemini is cooling down, rate-limited (429), overloaded (503), or times out,
        it immediately and seamlessly routes the query to ChatGPT.
        """
        # If Gemini is currently cooling down or rate-limited, skip Gemini and go straight to ChatGPT
        if not self.is_gemini_healthy():
            status = self.get_circuit_status()
            rem = status.get("seconds_remaining_in_cooldown", 0)
            print(f"[GeminiOracle] Gemini cooling down ({rem}s remaining) or rate-limited. Quickly routing to ChatGPT...")
            return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)

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

        # Try Primary Gemini model with strict fast conversational timeout
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{PRIMARY_MODEL}:generateContent?key={self.api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)

            # Check 429 Quota / Rate-limit: DO NOT retry other Gemini models on same exhausted key!
            if resp.status_code == 429:
                print(f"[GeminiOracle] Model {PRIMARY_MODEL} quota/rate-limited (HTTP 429). Quickly switching to ChatGPT...")
                self.trip_circuit("Gemini quota/rate-limit (HTTP 429)")
                return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)

            # Check 503 High Demand / Server Spikes: immediately switch to ChatGPT!
            if resp.status_code == 503:
                print(f"[GeminiOracle] Model {PRIMARY_MODEL} high demand (503). Quickly switching to ChatGPT...")
                self.trip_circuit("Gemini high demand (503)")
                return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)

            if resp.status_code != 200:
                err_msg = resp.text[:120]
                print(f"[GeminiOracle] API Error {resp.status_code} on {PRIMARY_MODEL}: {err_msg}. Quickly switching to ChatGPT...")
                return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)

            # 200 OK -> parse candidate
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                texts = [p.get("text", "") for p in parts if "text" in p]
                reply = "\n".join(texts).strip()
                if reply:
                    if self._circuit_open:
                        self.reset_circuit()
                    return reply

        except requests.exceptions.Timeout:
            print(f"[GeminiOracle] Request timed out on {PRIMARY_MODEL} after {timeout}s. Quickly switching to ChatGPT...")
            return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)
        except requests.exceptions.RequestException as req_err:
            print(f"[GeminiOracle] Network error on {PRIMARY_MODEL}: {req_err}. Quickly switching to ChatGPT...")
            return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)
        except Exception as ex:
            print(f"[GeminiOracle] Unexpected error on {PRIMARY_MODEL}: {ex}. Quickly switching to ChatGPT...")
            return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)

        # If Gemini returned empty response or reached here, switch to ChatGPT
        return self._query_chatgpt_fallback(prompt, search_context, system_instruction, timeout=25)


# Global singleton instance
_oracle_instance: Optional[GeminiSearchOracle] = None

def get_gemini_oracle() -> GeminiSearchOracle:
    """Returns the singleton GeminiSearchOracle instance."""
    global _oracle_instance
    if _oracle_instance is None:
        _oracle_instance = GeminiSearchOracle()
    return _oracle_instance
