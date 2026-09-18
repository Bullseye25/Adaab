import os
import sys
import json
import time
import threading
import urllib.request
import urllib.error

def resolve_modal_endpoint() -> str:
    """
    Dynamically discovers the web endpoint URL for AdaabAgentModel
    from the currently active Modal profile/app.
    """
    env_url = os.environ.get("ADAAB_ENDPOINT_URL")
    if env_url:
        return env_url

    try:
        import modal
        cls = modal.Cls.from_name("adaab-agent-backend", "AdaabAgentModel")
        obj = cls()
        web_url_fn = getattr(obj.chat_completions, "get_web_url", None)
        if callable(web_url_fn):
            resolved = web_url_fn()
            if resolved:
                return resolved
    except Exception:
        pass

    return "https://ridaraza2499--adaab-agent-backend-adaabagentmodel-chat-c-40745c.modal.run"

def resolve_modal_image_endpoint() -> str:
    """
    Dynamically discovers the web endpoint URL for ZImageTurboModel
    from the active Modal app (adaab-z-image-turbo).
    """
    env_url = os.environ.get("ADAAB_IMAGE_ENDPOINT_URL")
    if env_url:
        return env_url

    try:
        import modal
        cls = modal.Cls.from_name("adaab-z-image-turbo", "ZImageTurboModel")
        obj = cls()
        web_url_fn = getattr(obj.generate_endpoint, "get_web_url", None)
        if callable(web_url_fn):
            resolved = web_url_fn()
            if resolved:
                return resolved
    except Exception:
        pass

    return "https://ridaraza2499--adaab-z-image-turbo-zimageturbomodel-gener-4fbafd.modal.run"

ENDPOINT_URL = resolve_modal_endpoint()
IMAGE_ENDPOINT_URL = resolve_modal_image_endpoint()

def get_modal_client():
    """Returns local helper interface to Modal backend."""
    return AdaabClient()

class AdaabClient:
    def __init__(self, endpoint_url: str = None, image_endpoint_url: str = None):
        self.endpoint_url = endpoint_url or resolve_modal_endpoint()
        self.image_endpoint_url = image_endpoint_url or resolve_modal_image_endpoint()
        self._heartbeat_thread = None
        self._stop_heartbeat_event = threading.Event()
        self._last_heartbeat_time = 0.0

    def heartbeat(self, timeout: float = 35.0) -> dict:
        """
        Sends a fast, zero-compute keep-alive ping to the Modal GPU container.
        Resets Modal's scaledown_window (300s) without running inference or generating tokens.
        """
        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps({"heartbeat": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._last_heartbeat_time = time.time()
                return {"status": "alive", "heartbeat": True, "code": resp.status, "model": data.get("model")}
        except Exception as e:
            return {"status": "offline_or_cold", "error": str(e)}

    def start_heartbeat(self, interval_sec: float = 48.0, verbose: bool = False):
        """
        Starts a background daemon thread that sends a keep-alive heartbeat every 45~50 seconds
        (default: 48s) as long as the CLI/app is active.
        Terminates automatically when the main Python CLI process exits.
        """
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._stop_heartbeat_event.clear()

        def _heartbeat_loop():
            # Initial ping to verify container state
            res = self.heartbeat(timeout=30.0)
            if verbose:
                print(f"[Heartbeat] Initial GPU keep-alive status: {res.get('status')}", flush=True)

            while not self._stop_heartbeat_event.is_set():
                # Sleep in 1-second slices so thread terminates immediately upon process exit/Ctrl+C
                for _ in range(int(interval_sec)):
                    if self._stop_heartbeat_event.is_set():
                        return
                    time.sleep(1.0)

                # Send lightweight keep-alive ping (45~50s cadence)
                res = self.heartbeat(timeout=15.0)
                if verbose:
                    t_str = time.strftime('%H:%M:%S')
                    print(f"[Heartbeat] GPU keep-alive ping ({t_str}): {res.get('status')}", flush=True)

        self._heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name="AdaabGPUHeartbeatThread",
            daemon=True
        )
        self._heartbeat_thread.start()
        print(f"[Adaab] GPU Heartbeat active (pinging every {int(interval_sec)}s to maintain warm GPU container)...", flush=True)

    def stop_heartbeat(self):
        """Stops the heartbeat background loop."""
        self._stop_heartbeat_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)

    def health_check(self) -> dict:
        """Pings the Modal backend to check status."""
        res = self.heartbeat(timeout=35.0)
        if res.get("status") == "alive":
            return {"status": "online", "code": 200, "model": res.get("model")}
        return {"status": "offline_or_cold", "error": res.get("error", "Unknown error")}


    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.65,
        top_p: float = 0.9,
        max_tokens: int = 800,
        repetition_penalty: float = 1.20,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.5,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ) -> dict:
        """
        Sends an inference request to the serverless Modal Qwen 2.5 7B backend.
        Includes cold-start container spin-up retry handling, presence/frequency penalties,
        and anti-repetition penalty for natural Urdu conversational flow.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
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

    def generate_image(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 9,
        seed: int = None,
        max_retries: int = 2,
        retry_delay: float = 5.0
    ) -> dict:
        """
        Dispatches an image generation request to the serverless Modal Z-Image-Turbo backend.
        Supports both direct Modal SDK invocation and HTTP endpoint fallback.
        Returns generated image base64, filename, and modal persistent storage path.
        """
        # 1. Attempt direct Modal SDK invocation if available
        try:
            import modal
            cls = modal.Cls.from_name("adaab-z-image-turbo", "ZImageTurboModel")
            obj = cls()
            print(f"[AdaabClient] Invoking ZImageTurboModel via Modal SDK...")
            res = obj.generate.remote(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed
            )
            if res and res.get("status") == "success":
                return res
        except Exception as sdk_err:
            print(f"[AdaabClient] Modal SDK invocation notice ({sdk_err}). Falling back to HTTP endpoint...")

        # 2. HTTP Endpoint fallback
        payload = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "seed": seed
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.image_endpoint_url,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=180) as response:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    if resp_data.get("status") == "success":
                        return resp_data
                    raise RuntimeError(resp_data.get("message") or "Unknown image generation failure")
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                last_error = f"HTTP {e.code}: {err_body}"
                print(f"[AdaabClient] Image attempt {attempt + 1} HTTP error: {last_error}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except Exception as e:
                last_error = str(e)
                print(f"[AdaabClient] Image attempt {attempt + 1} notice: Container spin-up ({last_error}). Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        raise RuntimeError(f"[AdaabClient] Image generation failed after {max_retries + 1} attempts: {last_error}")

if __name__ == "__main__":
    print(f"Testing Adaab Client connection to: {ENDPOINT_URL}")
    client = AdaabClient()
    health = client.health_check()
    print("Health Status:", health)
