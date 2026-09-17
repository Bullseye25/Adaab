"""
harvest_chatgpt_50.py
─────────────────────────────────────────────────────────────────────────────
Robust CDP Harvester for ChatGPT Web session.
Captures Message 1, prompts for Message 2, streams completion, parses all
50+ Q&A pairs, sanitizes Urdu formatting, and updates master training corpus.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import re
import asyncio
import aiohttp
import requests

# UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from chatgpt_browser_oracle import get_chatgpt_oracle
from dataset_manager import DATA_DIR, build_master_dataset, STANDARD_SYSTEM_PROMPT

OUTPUT_FILE = os.path.join(DATA_DIR, "chatgpt_50_questions_dataset.jsonl")


async def execute_cdp(ws, method: str, params: dict = None, req_id: int = 1) -> dict:
    payload = {"id": req_id, "method": method, "params": params or {}}
    await ws.send_str(json.dumps(payload))
    while True:
        msg = await ws.receive_str()
        data = json.loads(msg)
        if data.get("id") == req_id:
            return data.get("result", {})


async def harvest():
    oracle = get_chatgpt_oracle()
    tab = oracle.get_chatgpt_tab()
    if not tab:
        print("[Error] No ChatGPT tab found.")
        return

    ws_url = tab.get("webSocketDebuggerUrl")
    print(f"[ChatGPT] Connected to session: {tab.get('url')}")

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            # 1. Bring tab to front
            await execute_cdp(ws, "Page.bringToFront", req_id=10)

            # 2. Inspect current messages
            inspect_js = """
            (function() {
                let asstMsgs = Array.from(document.querySelectorAll('[data-message-author-role="assistant"]'));
                let stopBtn = document.querySelector('button[data-testid="stop-button"]') || document.querySelector('button[aria-label="Stop streaming"]');
                return JSON.stringify({
                    asstCount: asstMsgs.length,
                    isStreaming: Boolean(stopBtn),
                    texts: asstMsgs.map(m => (m.querySelector('.markdown') || m).innerText)
                });
            })()
            """
            res = await execute_cdp(ws, "Runtime.evaluate", {"expression": inspect_js, "returnByValue": True}, req_id=11)
            raw = res.get("result", {}).get("value", "{}")
            state = json.loads(raw)
            asst_count = state.get("asstCount", 0)
            texts = state.get("texts", [])
            print(f"[ChatGPT] Found {asst_count} assistant responses already generated.")

            # 3. If only 1 message, send prompt for batch 2
            if asst_count < 2:
                print("\n[ChatGPT] Requesting Batch 2 (Next 26 Questions)...")
                prompt_batch_2 = (
                    "بہت خوب! اب مزید 26 نئے، معلوماتی اور دلچسپ عمومی سوالات و جوابات "
                    "(Technology, Space, Health, Nature, Social Etiquette, History) اسی طرح تحریر کریں:\n\n"
                    "لازمی اصول:\n"
                    "1. مخاطب کے لیے ہمیشہ احترام سے 'آپ' کا صیغہ استعمال کریں ('تم' یا 'تو' ہرگز نہیں)۔\n"
                    "2. کوئی ایموجی مت لگائیں اور مارک ڈاؤن بولڈ مت کریں۔\n"
                    "3. ہر جواب 2 سے 3 آسان، سلیس اور قدرتی جملوں پر مشتمل ہو۔\n"
                    "4. فارمیٹ:\n"
                    "سوال: [یہاں سوال لکھیں]\n"
                    "جواب: [یہاں جواب لکھیں]\n"
                    "موضوع: [موضوع کا نام]"
                )

                # Focus prompt textarea
                focus_js = """
                (function() {
                    let ta = document.querySelector('#prompt-textarea p') ||
                             document.querySelector('#prompt-textarea') ||
                             document.querySelector('div[contenteditable="true"]');
                    if (ta) { ta.focus(); return true; }
                    return false;
                })()
                """
                await execute_cdp(ws, "Runtime.evaluate", {"expression": focus_js, "returnByValue": True}, req_id=20)
                await asyncio.sleep(0.3)

                # Insert text
                await execute_cdp(ws, "Input.insertText", {"text": prompt_batch_2.strip()}, req_id=21)
                await asyncio.sleep(0.5)

                # Click Send button
                click_js = """
                (function() {
                    let btn = document.querySelector('button[data-testid="send-button"]') ||
                              document.querySelector('button[aria-label="Send prompt"]') ||
                              document.querySelector('button[data-testid="fruitjuice-send-button"]');
                    if (btn && !btn.disabled) { btn.click(); return "CLICKED"; }
                    return "NOT_FOUND";
                })()
                """
                btn_res = await execute_cdp(ws, "Runtime.evaluate", {"expression": click_js, "returnByValue": True}, req_id=22)
                await asyncio.sleep(0.3)

                # Fallback Enter key
                await execute_cdp(ws, "Input.dispatchKeyEvent", {
                    "type": "keyDown", "windowsVirtualKeyCode": 13, "key": "Enter", "code": "Enter", "text": "\r"
                }, req_id=23)
                await execute_cdp(ws, "Input.dispatchKeyEvent", {
                    "type": "keyUp", "windowsVirtualKeyCode": 13, "key": "Enter", "code": "Enter"
                }, req_id=24)

                print("[ChatGPT] Batch 2 prompt dispatched. Waiting for streaming generation...")

                # 4. Stream wait loop
                start_wait = time.time()
                last_len = 0
                stable_cycles = 0

                while (time.time() - start_wait) < 240:
                    await asyncio.sleep(2.0)
                    poll_res = await execute_cdp(ws, "Runtime.evaluate", {"expression": inspect_js, "returnByValue": True}, req_id=30)
                    poll_data = json.loads(poll_res.get("result", {}).get("value", "{}"))
                    cur_asst_count = poll_data.get("asstCount", 0)
                    is_streaming = poll_data.get("isStreaming", False)
                    cur_texts = poll_data.get("texts", [])

                    if cur_asst_count >= 2:
                        msg2_text = cur_texts[1]
                        cur_len = len(msg2_text)
                        print(f"  [Generating Batch 2...] {cur_len} chars generated... (Streaming: {is_streaming})", end="\r", flush=True)

                        if not is_streaming:
                            if cur_len == last_len and cur_len > 500:
                                stable_cycles += 1
                                if stable_cycles >= 3:
                                    print(f"\n[ChatGPT] Batch 2 completed successfully ({cur_len} characters)!")
                                    texts = cur_texts
                                    break
                            else:
                                stable_cycles = 0
                                last_len = cur_len
                        else:
                            last_len = cur_len
                            stable_cycles = 0

            # 5. Parse Q&A pairs from all assistant messages
            print("\n[Parser] Extracting Q&A pairs from assistant messages...")
            all_pairs = []
            for m_idx, text in enumerate(texts, start=1):
                clean = re.sub(r'[\*\#\_]', '', text)
                blocks = clean.split("سوال:")
                for b in blocks:
                    b = b.strip()
                    if not b or "جواب:" not in b:
                        continue
                    parts = b.split("جواب:", 1)
                    q = parts[0].strip()
                    ans_part = parts[1].strip()

                    cat = "عمومی معلومات"
                    if "موضوع:" in ans_part:
                        ans_sub = ans_part.split("موضوع:", 1)
                        a = ans_sub[0].strip()
                        cat = ans_sub[1].split("\n")[0].strip()
                    else:
                        a = ans_part.strip()

                    # Sanitize
                    a = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf]', '', a)  # strip emojis
                    a = re.sub(r'\s+', ' ', a).strip()
                    q = re.sub(r'^\d+[\.\:\-]?\s*', '', q).strip()

                    if len(q) >= 6 and len(a) >= 20:
                        all_pairs.append({
                            "question": q,
                            "answer": a,
                            "category": cat
                        })

            print(f"[Parser] Total parsed and verified Q&A pairs from ChatGPT: {len(all_pairs)}")

            # 6. Save records to JSONL
            records = []
            for idx, item in enumerate(all_pairs, start=1):
                rec = {
                    "messages": [
                        {"role": "system", "content": STANDARD_SYSTEM_PROMPT},
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]}
                    ],
                    "metadata": {
                        "id": idx,
                        "category": item["category"],
                        "question": item["question"],
                        "source": "ChatGPT_Web_Session",
                        "response_length": len(item["answer"])
                    }
                }
                records.append(rec)

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"[OK] Successfully saved {len(records)} ChatGPT records to {OUTPUT_FILE}!")

            # 7. Refresh master training dataset
            master_path = build_master_dataset()
            print(f"[OK] Master training dataset refreshed at {master_path}!")


if __name__ == "__main__":
    asyncio.run(harvest())
