"""
chatgpt_browser_oracle.py
─────────────────────────────────────────────────────────────────────────────
ChatGPT Web Browser Oracle & Session Cookie Manager for Adaab Studio
─────────────────────────────────────────────────────────────────────────────
• Uses authentic desktop Google Chrome / Microsoft Edge with Chrome DevTools
  Protocol (CDP) over WebSocket (via aiohttp).
• Completely bypasses Cloudflare anti-bot checks by using your real browser.
• Manages persistent browser profile in .chatgpt_profile and exports cookies
  to chatgpt_cookies.json.
• Acts as a seamless zero-cost fallback when Google Gemini Oracle quota limits
  (HTTP 429) are reached.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import asyncio
import subprocess
import requests
from typing import Optional, Dict, Any, List

# Output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(BASE_DIR, ".chatgpt_profile")
COOKIES_FILE = os.path.join(BASE_DIR, "chatgpt_cookies.json")
CDP_PORT = 9222

KNOWN_BROWSER_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
]


def find_browser_executable() -> Optional[str]:
    """Locates Google Chrome or Microsoft Edge executable on Windows."""
    for path in KNOWN_BROWSER_PATHS:
        if os.path.exists(path):
            return path
    return None


class ChatGPTBrowserOracle:
    def __init__(
        self,
        profile_dir: str = PROFILE_DIR,
        cookies_file: str = COOKIES_FILE,
        cdp_port: int = CDP_PORT
    ):
        self.profile_dir = profile_dir
        self.cookies_file = cookies_file
        self.cdp_port = cdp_port
        self.browser_path = find_browser_executable()
        self._browser_process: Optional[subprocess.Popen] = None

    def is_configured(self) -> bool:
        """Returns True if user profile directory or cookies exist."""
        return os.path.exists(self.profile_dir) or os.path.exists(self.cookies_file)

    def is_available(self) -> bool:
        """Checks if browser executable and configured profile are present."""
        return bool(self.browser_path) and self.is_configured()

    def is_browser_running(self) -> bool:
        """Checks if Chrome is currently running with remote debugging enabled."""
        try:
            r = requests.get(f"http://127.0.0.1:{self.cdp_port}/json/version", timeout=1.5)
            return r.status_code == 200
        except Exception:
            return False

    def ensure_browser_running(self, visible: bool = True) -> bool:
        """Ensures the browser is launched with CDP remote debugging and profile."""
        if self.is_browser_running():
            return True

        if not self.browser_path:
            print("[ChatGPT-Oracle] No supported Chromium browser (Chrome or Edge) found.")
            return False

        os.makedirs(self.profile_dir, exist_ok=True)
        cmd = [
            self.browser_path,
            f"--remote-debugging-port={self.cdp_port}",
            f"--user-data-dir={self.profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "https://chatgpt.com/"
        ]

        if not visible:
            cmd.append("--window-position=-2400,-2400")

        try:
            self._browser_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Wait up to 10 seconds for CDP endpoint to respond
            for _ in range(20):
                time.sleep(0.5)
                if self.is_browser_running():
                    return True
        except Exception as e:
            print(f"[ChatGPT-Oracle] Error launching browser: {e}")

        return self.is_browser_running()

    def get_chatgpt_tab(self) -> Optional[Dict[str, Any]]:
        """Finds or opens a ChatGPT page target via CDP."""
        try:
            r = requests.get(f"http://127.0.0.1:{self.cdp_port}/json", timeout=2)
            if r.status_code != 200:
                return None
            tabs = r.json()
            # Look for existing ChatGPT tab
            for tab in tabs:
                url = tab.get("url", "").lower()
                if "chatgpt.com" in url or "chat.openai.com" in url:
                    return tab

            # If not found, open a new ChatGPT tab
            new_req = requests.put(f"http://127.0.0.1:{self.cdp_port}/json/new?https://chatgpt.com/", timeout=4)
            if new_req.status_code == 200:
                time.sleep(2)
                return new_req.json()

            # Return first page tab if available
            for tab in tabs:
                if tab.get("type") == "page":
                    return tab
        except Exception as e:
            print(f"[ChatGPT-Oracle] Notice querying CDP tabs: {e}")
        return None

    def export_cookies(self) -> List[Dict[str, Any]]:
        """Exports cookies from ChatGPT session and writes to chatgpt_cookies.json."""
        if not self.ensure_browser_running():
            return []

        tab = self.get_chatgpt_tab()
        if not tab or not tab.get("webSocketDebuggerUrl"):
            return []

        ws_url = tab["webSocketDebuggerUrl"]

        async def _fetch():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    req = {
                        "id": 101,
                        "method": "Network.getCookies",
                        "params": {"urls": ["https://chatgpt.com", "https://openai.com"]}
                    }
                    await ws.send_str(json.dumps(req))
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("id") == 101:
                                return data.get("result", {}).get("cookies", [])
            return []

        try:
            cookies = asyncio.run(_fetch())
            if cookies:
                with open(self.cookies_file, "w", encoding="utf-8") as f:
                    json.dump(cookies, f, indent=2)
                print(f"[ChatGPT-Oracle] Exported {len(cookies)} session cookies to {os.path.basename(self.cookies_file)}.")
            return cookies
        except Exception as e:
            print(f"[ChatGPT-Oracle] Cookie export notice: {e}")
            return []

    def launch_login_session(self) -> bool:
        """
        Interactive login wizard for start.py CLI:
        Opens Chrome with persistent profile, allows user to log in,
        and saves session verification & cookies upon pressing Enter.
        """
        print("\n" + "=" * 70)
        print(" ChatGPT Web Oracle - Browser Session Setup & Login ".center(70, "="))
        print("=" * 70)
        if not self.browser_path:
            print("\n❌ Error: Google Chrome or Microsoft Edge was not detected on this system.")
            print("Please install Google Chrome from https://www.google.com/chrome/")
            return False

        print(f" • Browser Executable: {self.browser_path}")
        print(f" • Profile Directory:  {self.profile_dir}")
        print(f" • Target Service:     https://chatgpt.com/")
        print("=" * 70)
        print("\nLaunching browser window for one-time login...")

        if not self.ensure_browser_running(visible=True):
            print("❌ Failed to launch browser with remote debugging.")
            return False

        print("\n[INSTRUCTIONS]")
        print(" 1. In the opened browser window, log into your ChatGPT account.")
        print(" 2. Make sure you can see the ChatGPT home screen and message input box.")
        print(" 3. Return here and press [ENTER] to verify and save your session.\n")

        input("Press [ENTER] after logging into ChatGPT... ")

        print("\nVerifying session and exporting cookies...")
        tab = self.get_chatgpt_tab()
        if not tab:
            print("⚠️ Could not detect ChatGPT tab. Please ensure https://chatgpt.com/ is loaded.")
            return False

        cookies = self.export_cookies()
        print("\n" + "=" * 70)
        print(" ✓ ChatGPT Web Session Saved Successfully! ".center(70, "="))
        print("=" * 70)
        print(f" Session profile stored in: {self.profile_dir}")
        if cookies:
            print(f" Active cookies captured:   {len(cookies)}")
        print(" When Gemini Oracle quota is exhausted, Adaab will automatically")
        print(" use this ChatGPT browser session as an instant fallback.")
        print("=" * 70 + "\n")
        return True

    def query(
        self,
        prompt: str,
        search_context: Optional[str] = None,
        system_instruction: Optional[str] = None,
        timeout: int = 45
    ) -> Optional[str]:
        """
        Executes a prompt against ChatGPT via authentic browser automation.
        Types prompt using CDP Input.insertText, clicks send, waits for streaming
        to complete, and extracts the response text.
        """
        if not self.ensure_browser_running(visible=True):
            return None

        tab = self.get_chatgpt_tab()
        if not tab or not tab.get("webSocketDebuggerUrl"):
            return None

        ws_url = tab["webSocketDebuggerUrl"]

        # Build composite prompt
        full_query = prompt
        if search_context:
            full_query = f"Context:\n{search_context}\n\nQuestion:\n{prompt}"
        if system_instruction:
            full_query = f"Instruction: {system_instruction}\n\n{full_query}"

        # Clean string for injection
        escaped_query = full_query.strip()

        async def _run_interaction():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    msg_id = 1

                    async def send_cmd(method, params=None):
                        nonlocal msg_id
                        cur_id = msg_id
                        msg_id += 1
                        payload = {"id": cur_id, "method": method, "params": params or {}}
                        await ws.send_str(json.dumps(payload))
                        return cur_id

                    # 1. Bring page to front and focus
                    await send_cmd("Page.bringToFront")
                    await asyncio.sleep(0.5)

                    # 2. Check and focus prompt textarea
                    focus_js = """
                    (function() {
                        let ta = document.querySelector('#prompt-textarea p') ||
                                 document.querySelector('#prompt-textarea') ||
                                 document.querySelector('div[contenteditable="true"]') ||
                                 document.querySelector('textarea');
                        if (ta) {
                            ta.focus();
                            return true;
                        }
                        return false;
                    })()
                    """
                    f_id = await send_cmd("Runtime.evaluate", {"expression": focus_js, "returnByValue": True})
                    await asyncio.sleep(0.3)

                    # 3. Clear existing text in textarea
                    clear_js = """
                    (function() {
                        let ta = document.querySelector('#prompt-textarea p') ||
                                 document.querySelector('#prompt-textarea') ||
                                 document.querySelector('div[contenteditable="true"]');
                        if (ta) {
                            ta.innerText = '';
                            ta.dispatchEvent(new Event('input', { bubbles: true }));
                            return true;
                        }
                        return false;
                    })()
                    """
                    await send_cmd("Runtime.evaluate", {"expression": clear_js, "returnByValue": True})
                    await asyncio.sleep(0.2)

                    # 4. Insert text via native CDP input
                    await send_cmd("Input.insertText", {"text": escaped_query})
                    await asyncio.sleep(0.4)

                    # 5. Click Send Button or dispatch Enter key
                    click_send_js = """
                    (function() {
                        let btn = document.querySelector('button[data-testid="send-button"]') ||
                                  document.querySelector('button[aria-label="Send prompt"]') ||
                                  document.querySelector('button[data-testid="fruitjuice-send-button"]');
                        if (btn && !btn.disabled) {
                            btn.click();
                            return true;
                        }
                        return false;
                    })()
                    """
                    c_id = await send_cmd("Runtime.evaluate", {"expression": click_send_js, "returnByValue": True})
                    await asyncio.sleep(0.3)

                    # If button not clickable, dispatch Enter key
                    await send_cmd("Input.dispatchKeyEvent", {
                        "type": "keyDown",
                        "windowsVirtualKeyCode": 13,
                        "key": "Enter",
                        "code": "Enter",
                        "text": "\r"
                    })
                    await send_cmd("Input.dispatchKeyEvent", {
                        "type": "keyUp",
                        "windowsVirtualKeyCode": 13,
                        "key": "Enter",
                        "code": "Enter"
                    })

                    # 6. Wait for response generation to complete
                    start_time = time.time()
                    last_text = ""
                    stable_count = 0

                    poll_js = """
                    (function() {
                        let stopBtn = document.querySelector('button[data-testid="stop-button"]') ||
                                      document.querySelector('button[aria-label="Stop streaming"]');
                        let msgs = document.querySelectorAll('[data-message-author-role="assistant"]');
                        if (msgs.length === 0) {
                            return { hasMsg: false, streaming: Boolean(stopBtn), text: "" };
                        }
                        let lastMsg = msgs[msgs.length - 1];
                        let prose = lastMsg.querySelector('.markdown') || lastMsg.querySelector('.prose') || lastMsg;
                        return {
                            hasMsg: true,
                            streaming: Boolean(stopBtn),
                            text: prose.innerText ? prose.innerText.trim() : ""
                        };
                    })()
                    """

                    while (time.time() - start_time) < timeout:
                        await asyncio.sleep(0.8)
                        poll_id = await send_cmd("Runtime.evaluate", {"expression": poll_js, "returnByValue": True})
                        
                        # Read responses from WS
                        try:
                            msg = await asyncio.wait_for(ws.receive_str(), timeout=2.0)
                            data = json.loads(msg)
                            if data.get("id") == poll_id:
                                val = data.get("result", {}).get("result", {}).get("value", {})
                                has_msg = val.get("hasMsg", False)
                                is_streaming = val.get("streaming", False)
                                current_text = val.get("text", "")

                                if has_msg and current_text:
                                    if not is_streaming:
                                        if current_text == last_text:
                                            stable_count += 1
                                            if stable_count >= 2:
                                                # Completed and stable
                                                return current_text
                                        else:
                                            stable_count = 0
                                            last_text = current_text
                                    else:
                                        last_text = current_text
                                        stable_count = 0
                        except asyncio.TimeoutError:
                            pass

                    return last_text if last_text else None

        try:
            result = asyncio.run(_run_interaction())
            return result
        except Exception as ex:
            print(f"[ChatGPT-Oracle] Interaction notice: {ex}")
            return None


# Global singleton instance
_chatgpt_oracle_instance: Optional[ChatGPTBrowserOracle] = None

def get_chatgpt_oracle() -> ChatGPTBrowserOracle:
    """Returns the singleton ChatGPTBrowserOracle instance."""
    global _chatgpt_oracle_instance
    if _chatgpt_oracle_instance is None:
        _chatgpt_oracle_instance = ChatGPTBrowserOracle()
    return _chatgpt_oracle_instance
