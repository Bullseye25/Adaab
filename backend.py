import os
import sys
import json
import time
import urllib.request
import urllib.error

# Default endpoint URL (populated automatically upon running modal deploy deploy_adaab.py)
ENDPOINT_URL = os.environ.get(
    "ADAAB_ENDPOINT_URL",
    "https://ammadraza27--adaab-agent-backend-adaabagentmodel-chat--66c2c9.modal.run"
)

def get_modal_client():
    """Returns local helper interface to Modal backend."""
    return AdaabClient(ENDPOINT_URL)

class AdaabClient:
    def __init__(self, endpoint_url: str = ENDPOINT_URL):
        self.endpoint_url = endpoint_url

    def health_check(self) -> dict:
        """Pings the Modal backend to check status."""
        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps({"messages": [{"role": "user", "content": "ping"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return {"status": "online", "code": resp.status}
        except Exception as e:
            return {"status": "offline_or_cold", "error": str(e)}

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ) -> dict:
        """
        Sends an inference request to the serverless Modal Qwen 2.5 7B backend.
        Includes cold-start container spin-up retry handling.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.endpoint_url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                # 180s timeout accommodates L4 GPU spin up from scale-to-zero
                with urllib.request.urlopen(req, timeout=180) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    choice = resp_data.get("choices", [{}])[0]
                    usage = resp_data.get("usage", {})
                    return {
                        "content": choice.get("message", {}).get("content", ""),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "model": resp_data.get("model", "Qwen/Qwen2.5-7B-Instruct")
                    }
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                last_error = f"HTTP {e.code}: {err_body}"
                print(f"[AdaabClient] Attempt {attempt + 1} HTTP Error: {last_error}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except Exception as e:
                last_error = str(e)
                print(f"[AdaabClient] Attempt {attempt + 1} Connection notice: Container spinning up ({last_error}). Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        raise RuntimeError(f"[AdaabClient] Cloud request failed after {max_retries + 1} attempts: {last_error}")

if __name__ == "__main__":
    print(f"Testing Adaab Client connection to: {ENDPOINT_URL}")
    client = AdaabClient()
    health = client.health_check()
    print("Health Status:", health)
