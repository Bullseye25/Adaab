# آداب (Adaab) - Autonomous Urdu Poetry, Poster & Video Reel Studio

An intelligent, culturally refined Urdu-first AI agent powered by **Qwen 2.5 7B Instruct**, deployed serverlessly on **Modal.com** (NVIDIA L4 GPU). 

All heavy computation, inference, image generation, typography rendering, voice synthesis, and video reel assembly runs **100% in the cloud on Modal.com with zero local CPU or GPU load**.

---

## 🌟 Key Features

1. **Urdu-First Intelligence:** Default formal, respectful Urdu persona (*Aap*, *Janaab*, *Tehzeeb*), capable of classical Ghazals, Nazms, and Rubaiyat with proper *Wazn*, *Qafia*, and *Radif*.
2. **48,000 Aesthetic Matrix:** Combinatorial art engine sampling from **40 color palettes**, **30 art styles**, **20 nature themes**, and **Day/Night lighting cycles**.
3. **High-Res 9:16 Typography Posters:** Center-aligned, framed Nastaliq calligraphy with an adaptive feathered contrast scrim and dual-layer drop shadows for crystal-clear readability on any background.
4. **Natural Female Urdu Voice:** Expressive poetry recitation using `ur-PK-UzmaNeural` with authentic poetic tempo (-15% speed) and inter-verse acoustic pauses.
5. **Precision Video Reels (MP4):** Automatically merges poster + voice into 1080x1920 MP4 reels with an exact **0.5-second visual intro** and **0.5-second visual outro hold**.
6. **Local SQLite Database (`adaab_history.db`):** SHA-256 normalized hash tracking ensures that **no duplicate verse is ever repeated**.
7. **GPU Crash Tracker (`crash_tracker.py`):** Real-time CUDA OOM interception, VRAM telemetry logging, and dual cloud-to-local error reporting.
8. **1-Click Batch Generator (`generate_posts.bat`):** Prompts for only one number (*"How many posts?"*), creates everything in the cloud, saves to `Adaab/output/`, and automatically pops open Windows File Explorer!

---

## 📁 File Structure

```
H:\Adaab/
├── generate_posts.bat     # 1-Click batch script: asks "How many posts?", hits enter, does all!
├── run_adaab.bat          # Master Interactive CLI menu (Account, Deploy, Tests, Chat, Studio)
├── start.py               # Master CLI controller
├── generate_batch.py      # Automated batch runner
├── deploy_adaab.py        # Modal serverless app (vLLM, L4 GPU, Volume caching)
├── poster_maker.py        # 48,000 aesthetic matrix + Urdu typography compositor
├── voice_reader.py        # Female Urdu voice recitation engine (ur-PK-UzmaNeural)
├── video_maker.py         # 1080x1920 MP4 reel compositor with 0.5s intro/outro pacing
├── crash_tracker.py       # Real-time GPU crash and error tracking
├── test_suite.py          # 8-point automated unit & integration test suite
├── poetry_db.py           # SQLite database (adaab_history.db) & deduplication engine
├── adaab_history.db       # Local SQLite database file
├── backend.py             # HTTP connector to Modal cloud endpoints
├── agent.py               # Adaab Urdu Agent persona & prompt orchestration
├── app.py                 # Gradio Web UI with Nastaliq font, RTL support & live player
├── credentials.txt        # Local Hugging Face / Modal tokens
├── output/                # LOCAL OUTPUT DIRECTORY (Automatically opens in File Explorer)
│   ├── post_001_timestamp/
│   │   ├── poetry.txt     # Complete verse (Urdu, Roman Urdu, English, hashtags)
│   │   ├── poster.png     # 1080x1920 high-resolution portrait poster
│   │   ├── audio.mp3      # Expressive female voice recitation
│   │   └── reel.mp4       # 1080x1920 animated video reel
│   └── ...
└── logs/                  # Local crash and health reports
```

---

## 🚀 How to Run

### 1. 1-Click Batch Generation (Easiest)
Simply double-click **`generate_posts.bat`**.
* The console will ask: `How many posts would you like to generate? (e.g. 5):`
* Enter a number and press Enter.
* When finished, the `output` folder will automatically pop open in Windows File Explorer!

### 2. Master Studio & Cloud Administration Menu
Double-click **`run_adaab.bat`** to access all tools:
* Deploy / Redeploy to Modal (NVIDIA L4 GPU)
* Pre-cache weights into Modal Volume
* Run 8-Point Automated Unit Test Suite
* Interactive Urdu Terminal Chat
* Launch Gradio Web Studio (Browser UI)
* View GPU Telemetry & Error Reports

### 3. Deploying to Modal.com
To deploy or update the serverless backend on Modal:
```bash
python -m modal deploy deploy_adaab.py
```
To pre-cache base weights in the persistent volume:
```bash
python -m modal run deploy_adaab.py::download_adaab_weights
```

---

## 🧪 Running Unit Tests
To run the automated 8-point test suite:
```bash
python test_suite.py
```
This validates SQLite deduplication, Urdu font rendering, GPU telemetry, 9:16 poster generation, voice synthesis, video reel pacing, and cloud connectivity.
