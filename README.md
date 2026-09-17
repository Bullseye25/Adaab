# آداب (Adaab) — Conversational Urdu Voice AI Assistant

An intelligent, culturally refined, voice-first Urdu AI assistant powered by **Qwen 2.5 7B Instruct** (with expandable Pakistan Knowledge LoRA), deployed serverlessly on **Modal.com** (NVIDIA L4 GPU) with an intelligent Multi-Oracle Cognitive Orchestrator.

All heavy computation and inference runs **100% in the cloud with zero local GPU requirements**.

---

## 🌟 Key Features

1. **Urdu-First Conversational Intelligence (تبریز / Tabraiz):**
   - Natural, conversational Pakistani Urdu persona that speaks with respectful everyday warmth (*آپ*, *جناب*).
   - Zero-onboarding policy: greets immediately without intrusive interrogation.
   - Comprehensive understanding of English, Roman Urdu, and Nastaliq Urdu with culturally authentic Urdu replies.

2. **Cognitive Orchestrator & Follow-Up Anaphora Resolution:**
   - Detects conversational context so colloquial follow-ups (e.g., *"batao na"*, *"aur?"*, *"wahan kon tha?"*) seamlessly resolve to the active topic.
   - Ground-truth Pakistan cultural knowledge retrieval covering 10 domains (Cuisine, History, Culture, Sports, Telecom, Geography, Heritage, etc.).
   - Episodic topic memory persisted asynchronously in SQLite (`adaab_history.db`).

3. **Multi-Tier Intelligence & Fallback Oracles:**
   - **Primary Serverless Engine:** Modal.com NVIDIA L4 GPU running Qwen 2.5 7B Instruct with vLLM eager execution and anti-degeneration penalties.
   - **Gemini Oracle Fallback:** High-speed real-time web search and knowledge grounding with circuit breaker protection.
   - **ChatGPT Browser Oracle:** Headless browser fallback via Chrome DevTools Protocol (CDP) for quota overflow.

4. **Real-Time Voice Pipeline:**
   - **Speech-to-Text (ASR):** Google Speech API configured for `ur-PK`.
   - **Neural Text-to-Speech (TTS):** Microsoft Azure Neural voice `ur-PK-AsadNeural` (Tabraiz) and `ur-PK-UzmaNeural` (Tehzeeb) via Edge-TTS.
   - Real-time client-side audio playback with in-memory Base64 data URIs.

5. **Expandable LoRA Fine-Tuning (`adaab-pakistan-lora`):**
   - Modular training dataset aggregator (`dataset_manager.py`).
   - 1-click cloud QLoRA fine-tuning on Modal NVIDIA L4 GPU (`train_qwen_pakistan_lora.py`).
   - Direct cloud backup to Google Drive with zero local disk footprint.

6. **Minimalist Responsive Web Interface:**
   - Antigravity-style clean conversational chat layout.
   - Fully optimized for iPhone and Android (touch-first, iOS safe areas, zero 300ms delays).
   - Cloudflare Quick Tunnel integration for global access with zero firewall blocks.

---

## 📁 Project Structure

```
H:\Adaab/
├── app.py                     # Minimalist conversational web UI (Gradio 6)
├── agent.py                   # Adaab conversational agent & system prompts
├── orchestrator.py            # Cognitive intent routing & follow-up resolution
├── backend.py                 # Modal serverless client & GPU keep-alive heartbeat
├── deploy_adaab.py            # Modal vLLM deployment (NVIDIA L4, Volume cache)
├── dataset_manager.py         # Modular LoRA training dataset manager
├── train_qwen_pakistan_lora.py# Cloud QLoRA fine-tuning script
├── gemini_oracle.py           # Multi-tier Gemini knowledge oracle & circuit breaker
├── chatgpt_browser_oracle.py  # CDP-based browser automation fallback oracle
├── voice_listener.py          # Audio transcription & ASR pipeline
├── voice_reader.py            # Neural TTS synthesis engine (Edge-TTS)
├── memory_engine.py           # SQLite episodic memory & topic tracker
├── time_context.py            # Real-time Pakistani date and time provider
├── network_helper.py          # Cloudflare tunnel & local SSL management
├── crash_tracker.py           # GPU telemetry & error tracking
├── start.py                   # Master English CLI launcher (18 options)
├── test_suite.py              # Automated unit test suite
├── data/
│   ├── master_training_dataset.jsonl    # Master compiled LoRA dataset
│   └── pakistan_knowledge_dataset.jsonl # Pakistan cultural knowledge pairs
└── assets/
    └── icons/                 # Clean UI icons (mic, send, speaker, etc.)
```

---

## 🚀 Getting Started

### 1. Launch Interactive CLI
Run the master controller:
```bash
python start.py
```
Options include:
- Launch Web UI (Local PC or Cloudflare Tunnel for Mobile)
- Deploy / Re-deploy to Modal NVIDIA L4 GPU
- Fine-tune Pakistan Knowledge LoRA on Cloud GPU
- Manage and expand LoRA training dataset
- Backup / Restore from Google Drive
- Run System Diagnostics & Test Suites

### 2. Launch Web Interface Directly
```bash
# Standard local launch
python app.py

# Launch with Cloudflare Quick Tunnel (Ideal for iPhone/Android access)
python app.py --tunnel
```

---

## 🧪 Testing

Run the automated test suite:
```bash
python test_suite.py
```
This validates SQLite memory persistence, Urdu font rendering, GPU telemetry, neural voice generation, time/date context, identity recognition, and speech recognition pipeline.

