import os
import sys
import subprocess
import webbrowser
import time

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

def install_package(package_name, pip_name=None):
    pip_name = pip_name or package_name
    print(f"Installing {package_name} package locally...")
    subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=True)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        from modal_account_manager import get_current_modal_profile
        active_modal = get_current_modal_profile()
    except Exception:
        active_modal = "ridaraza2499"

    print("=" * 72)
    print(" Adaab — Conversational Voice AI Studio ".center(72, "="))
    print(" Cloud Backend: Modal.com (NVIDIA L4 GPU) ".center(72, " "))
    print(" Policy: 0% Local GPU Used | 0% Google Storage (GCS) Cost ".center(72, " "))
    print("=" * 72)

    print("\n [1] Launch Web Studio (Push-to-Talk, Voice Chat, Mobile QR) [Default]")
    print(" [2] Modal.com Cloud GPU Backend (Inference Test, Deploy, Diagnostics)")
    print(" [3] Google Drive Cloud Backup (Backup/Restore Weights, 5TB Quota)")
    print(" [4] Pakistan Knowledge Engine & LoRA (200 Topics Sampler, Fine-Tune)")
    print(" [5] AI Cloud Oracles (Ollama Cloud, Modal GPU & Gemini Status)")
    print(" [6] Memory & Topics Database (View Topics, Reset Memory)")
    print(" [7] System Diagnostics & 12-Point Test Suite")
    print(" [0] Exit")

    choice = sys.argv[1].lower() if len(sys.argv) > 1 else (input("\nEnter choice [0-7] (default 1 for Web Studio): ").strip().lower() or "1")

    # ── Option 1: Web Studio (Default) ──
    if choice in ["1", "web", "studio", "frontend", "ui"]:
        print("\n" + "=" * 72)
        print(" Adaab Voice AI Web Studio ".center(72, "="))
        print(" Mobile Push-to-Talk | Tabraiz Conversational AI | Neon Waveform")
        print("=" * 72)
        print("\nLaunching Secure Mobile Tunnel...")
        try:
            from network_helper import terminate_all_stale_app_processes
            terminate_all_stale_app_processes()
        except Exception:
            pass
        try:
            import gradio
        except ImportError:
            install_package("gradio", "gradio>=5.0")
        cmd = [sys.executable, "-X", "utf8", "app.py", "--tunnel"]
        subprocess.run(cmd)

    # ── Module 2: Modal.com Cloud GPU Backend ──
    elif choice in ["2", "modal", "cloud", "backend", "7", "8", "9", "10"]:
        print("\n" + "=" * 70)
        print(" Modal.com Cloud GPU Backend (NVIDIA L4) ".center(70, "="))
        print(" Serverless GPU Container | 0% Local GPU Used ".center(70, " "))
        print("=" * 70)
        print(" 1. Test Live Cloud Backend Inference (Qwen 2.5 7B on L4 GPU)")
        print(" 2. Deploy / Redeploy Adaab Backend to Modal")
        print(" 3. View Live GPU Telemetry and Diagnostic Logs")
        print(" 4. Switch / Manage Modal Accounts & Google Drive Setup")
        print(" 5. Open Modal Cloud Dashboard in Web Browser")
        print(" 6. Pre-cache Clean Qwen 2.5 7B Weights into Modal Volume")
        print(" 0. Return to Main Menu")
        m_c = input("\nEnter choice [0-6] (default 1): ").strip() or "1"

        if m_c == "1":
            print("\n---> Testing live Modal backend inference...")
            from backend import AdaabClient
            client = AdaabClient()
            test_messages = [
                {"role": "system", "content": "You are Tabraiz, a polite conversational Urdu AI."},
                {"role": "user", "content": "السلام علیکم! آپ کون ہیں اور آپ کیا کر سکتے ہیں؟"}
            ]
            try:
                resp = client.chat_completion(test_messages)
                print("\n--- Response from Qwen 2.5 (7B) on Modal L4 GPU ---")
                print(resp["content"])
                print("---------------------------------------------------")
                print(f"Tokens: {resp['prompt_tokens']} prompt, {resp['completion_tokens']} completion")
                print("✓ Cloud inference is working flawlessly!")
            except Exception as e:
                print(f"\n❌ Cloud test failed: {e}")

        elif m_c == "2":
            print("\n---> Deploying Adaab Serving Backend to Modal.com (NVIDIA L4 GPU)...")
            try:
                import modal
            except ImportError:
                install_package("modal")
            cmd = [sys.executable, "-X", "utf8", "-m", "modal", "deploy", "deploy_adaab.py"]
            try:
                subprocess.run(cmd, check=True)
                print("\n✓ Adaab Backend deployed to Modal successfully!")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Deployment failed: {e}")

        elif m_c == "3":
            from crash_tracker import get_gpu_telemetry, get_latest_crash_report
            import json
            print("\n=== GPU Telemetry (Local PC vs Cloud Container) ===")
            print(json.dumps(get_gpu_telemetry(), indent=2))
            print("\n=== Recent Diagnostic & Crash Events ===")
            print(get_latest_crash_report())

        elif m_c == "4":
            from modal_account_manager import run_account_migration_flow
            run_account_migration_flow()

        elif m_c == "5":
            dashboard_url = "https://modal.com/apps"
            print(f"\nOpening Modal Dashboard: {dashboard_url}")
            webbrowser.open(dashboard_url)

        elif m_c == "6":
            print("\n---> Pre-caching clean Qwen 2.5 7B weights into Modal volume 'adaab-cache'...")
            cmd = [sys.executable, "-X", "utf8", "-m", "modal", "run", "deploy_adaab.py::download_adaab_weights"]
            subprocess.run(cmd)

    # ── Module 3: Google Drive Cloud Backup (5TB) ──
    elif choice in ["3", "gdrive", "backup", "restore", "13", "14", "16"]:
        print("\n" + "=" * 70)
        print(" Google Drive Cloud Backup & Storage Engine (5TB) ".center(70, "="))
        print(" Zero GCP Storage Fees | Personal Cloud Backup ")
        print("=" * 70)
        print(" 1. Backup Trained Model & Weights to Google Drive")
        print(" 2. Restore Model & Weights from Google Drive to Modal")
        print(" 3. Configure / Authenticate Google Drive & Check 5TB Quota")
        print(" 0. Return to Main Menu")
        g_c = input("\nEnter choice [0-3] (default 1): ").strip() or "1"

        if g_c == "1":
            from modal_gdrive_sync import run_cloud_backup
            run_cloud_backup()
        elif g_c == "2":
            from modal_gdrive_sync import run_cloud_restore
            run_cloud_restore()
        elif g_c == "3":
            from gdrive_manager import setup_gdrive_auth_interactive, show_gdrive_storage_status, is_gdrive_authenticated, get_gdrive_service
            if is_gdrive_authenticated():
                print("\nGoogle Drive is currently authenticated.")
                srv = get_gdrive_service()
                show_gdrive_storage_status(srv)
                reauth = input("Would you like to re-authenticate or configure a different Google account? [y/N]: ").strip().lower()
                if reauth == "y":
                    setup_gdrive_auth_interactive()
            else:
                setup_gdrive_auth_interactive()

    # ── Module 4: Pakistan Knowledge Engine & LoRA ──
    elif choice in ["4", "curriculum", "topics200", "200", "dataset", "train", "finetune", "11", "12", "19"]:
        print("\n" + "=" * 70)
        print(" Pakistan Knowledge Curriculum Engine & LoRA Fine-Tuning ".center(70, "="))
        print("=" * 70)
        print(" 1. Pakistan 200 Topics Curriculum Engine (Random 50 Sampler & ChatGPT Q&A)")
        print(" 2. Add New Q&A Topic Interactively (dataset_manager.py)")
        print(" 3. Rebuild & Validate Master Training Dataset")
        print(" 4. Fine-Tune Pakistan Knowledge LoRA on Modal GPU (NVIDIA L4)")
        print(" 0. Return to Main Menu")
        p_c = input("\nEnter choice [0-4] (default 1): ").strip() or "1"

        if p_c == "1":
            subprocess.run([sys.executable, "-X", "utf8", "pakistan_curriculum_engine.py"])
        elif p_c == "2":
            from dataset_manager import interactive_add_topic
            interactive_add_topic()
        elif p_c == "3":
            from dataset_manager import build_master_dataset, show_dataset_summary
            master_file = build_master_dataset()
            print(f"\n✓ Master dataset updated and validated at: {master_file}")
            show_dataset_summary()
        elif p_c == "4":
            print("\n---> Initializing Qwen 2.5 7B QLoRA Fine-Tuning on Modal GPU (NVIDIA L4)...")
            from dataset_manager import build_master_dataset
            dataset_path = build_master_dataset()
            if not os.path.exists(dataset_path):
                print(f"❌ Error: Dataset file not found at {dataset_path}.")
            else:
                cmd = [sys.executable, "-X", "utf8", "-m", "modal", "run", "train_qwen_pakistan_lora.py"]
                try:
                    subprocess.run(cmd, check=True)
                    print("\n✓ Model fine-tuning completed successfully! LoRA weights saved to Modal volume 'adaab-cache'.")
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ Fine-tuning run failed: {e}")

    # ── Module 5: AI Cloud Oracles (Ollama Cloud, Modal GPU & Gemini Free Tier) ──
    elif choice in ["5", "oracle", "gemini", "ollama", "4_old", "5_old", "6"]:
        print("\n" + "=" * 70)
        print(" AI Cloud Oracles & Knowledge Engines ".center(70, "="))
        print(" 100% Free Tier | Zero Local PC GPU/RAM Load | High Speed Failover ".center(70, " "))
        print("=" * 70)
        print(" 1. Test Ollama Cloud Oracle Live Inference (gemma4:31b — ~300ms)")
        print(" 2. Check Google Gemini Free Tier Status & Quota Diagnostics")
        print(" 3. Check Modal Cloud GPU Backend Status & Health (Qwen 2.5 7B)")
        print(" 0. Return to Main Menu")
        o_c = input("\nEnter choice [0-3] (default 1): ").strip() or "1"

        if o_c == "1":
            from ollama_oracle import get_ollama_oracle
            oracle = get_ollama_oracle()
            if not oracle.is_available():
                print("\n⚠️ OLLAMA_API_KEY is not configured in credentials.txt.")
            else:
                print("\n" + "=" * 70)
                print(" Ollama Cloud Oracle — Live Inference Test ".center(70, "="))
                print(f" Target Model: {oracle.primary_model} (Cloud Hosted)")
                print(" Policy: 100% Free Tier | 0% Local GPU/CPU Load")
                print("=" * 70)
                test_prompt = input("Enter test question (press Enter for default): ").strip()
                if not test_prompt:
                    test_prompt = "پاکستان کے کون سے کھانے سب سے زیادہ مقبول ہیں؟"
                print(f"\nQuerying Ollama Cloud API ({oracle.primary_model})...")
                t0 = time.time()
                reply = oracle.query(prompt=test_prompt, timeout=15)
                elapsed = round(time.time() - t0, 2)
                if reply:
                    print(f"\n--- Response from Ollama Cloud ({elapsed}s) ---")
                    print(reply)
                    print("-" * 50)
                else:
                    print("\n❌ Did not receive a response within timeout.")

        elif o_c == "2":
            from gemini_oracle import get_gemini_oracle
            oracle = get_gemini_oracle()
            print("\n" + "=" * 70)
            print(" Google Gemini Knowledge Oracle — Status & Diagnostics ".center(70, "="))
            print(" Policy: Google AI Studio Free Tier (Zero Dollar Billing Guarantee)")
            print("=" * 70)
            status = oracle.get_circuit_status()
            print(f" API Key Configured:      {status['configured']}")
            print(f" Circuit Breaker Tripped: {status['circuit_open']}")
            print(f" Cooldown Remaining:      {status['seconds_remaining_in_cooldown']} seconds")
            print(f" Requests in Last Min:    {status['rpm_count_last_minute']}")
            if status['tripped_reason']:
                print(f" Reason for Trip:         {status['tripped_reason']}")
            
            from ollama_oracle import get_ollama_oracle
            ollama = get_ollama_oracle()
            print(f" Ollama Cloud Fallback:   {ollama.is_available()} (gemma4:31b)")
            print("=" * 70)

            test_q = input("\nSend test prompt to Gemini Oracle? [y/N]: ").strip().lower()
            if test_q == "y":
                print("\nSending query to Gemini...")
                t0 = time.time()
                res = oracle.query("سلام! پاکستان کے کتنے صوبے ہیں؟", timeout=12)
                elapsed = round(time.time() - t0, 2)
                if res:
                    print(f"\nResponse ({elapsed}s):\n{res}")
                else:
                    print("\nGemini rate-limited or unavailable. Circuit status updated.")

        elif o_c == "3":
            print("\n---> Pinging Modal.com Cloud GPU Backend (NVIDIA L4)...")
            from backend import AdaabClient
            client = AdaabClient()
            try:
                t0 = time.time()
                health = client.health_check()
                elapsed = round(time.time() - t0, 2)
                print("\n" + "=" * 70)
                print(" Modal Cloud GPU Backend Status ".center(70, "="))
                print(f" Status:       {health.get('status', 'online')}")
                print(f" Device:       {health.get('device', 'NVIDIA L4 GPU')}")
                print(f" Latency:      {elapsed}s")
                print(" Platform:     Modal.com Serverless Cloud")
                print(" Policy:       0% Local PC GPU Used")
                print("=" * 70)
            except Exception as e:
                print(f"\n❌ Modal GPU check failed: {e}")

    # ── Module 6: Memory & Topics Database ──
    elif choice in ["6", "memory", "topics", "profiles", "17", "18"]:
        print("\n" + "=" * 70)
        print(" SQLite Episodic Memory & Topic Database ".center(70, "="))
        print("=" * 70)
        print(" 1. View Recorded Conversation Topics & User Memory")
        print(" 2. Clear / Reset Conversation and User Database")
        print(" 0. Return to Main Menu")
        mem_c = input("\nEnter choice [0-2] (default 1): ").strip() or "1"

        if mem_c == "1":
            from memory_engine import (
                get_active_discussion_topic,
                get_recent_conversation_context,
                format_who_did_you_talk_to_response
            )
            print("\n" + "=" * 70)
            print(" SQLite Episodic Memory & Topic Tracker ".center(70, "="))
            print("=" * 70 + "\n")
            
            ctx = get_active_discussion_topic()
            print("[Active Discussion Context]:")
            if ctx and ctx.get("active_topic"):
                print(f"  • Active Topic:  {ctx.get('active_topic')}")
                print(f"  • Last Query:    {ctx.get('last_query')}")
                last_resp = ctx.get('last_response', '')
                print(f"  • Last Response: {last_resp[:120]}..." if last_resp else "  • Last Response: None")
                print(f"  • Turn Count:    {ctx.get('turn_count')}")
            else:
                print("  • No active conversation session recorded yet.")
                
            print("\n[Recent Dialogue Turns]:")
            recent_turns = get_recent_conversation_context(limit=10)
            if recent_turns:
                for i, t in enumerate(recent_turns, 1):
                    top_label = t.get('detected_topic') or 'General'
                    print(f"  {i}. [{top_label}]")
                    print(f"     User:    {t.get('user_query')}")
                    print(f"     Tabraiz: {t.get('bot_response', '')[:100]}...\n")
            else:
                print("  • No dialogue turns recorded in database.\n")

            print("-" * 70)
            print(format_who_did_you_talk_to_response("Tabraiz"))

        elif mem_c == "2":
            print("\nWARNING: This will permanently wipe all conversation turns and profile records in adaab_history.db.")
            confirm = input("Are you sure you want to clear the memory database? [y/N]: ").strip().lower()
            if confirm == "y":
                from memory_engine import clear_user_memory
                clear_user_memory()
                print("✓ SQLite conversation and user memory cleared successfully!")
            else:
                print("Operation cancelled. Database untouched.")

    # ── Module 7: System Diagnostics & 12-Point Test Suite ──
    elif choice in ["7", "test", "tests", "diagnostics", "3"]:
        print("\n---> Running System Unit and Integration Suite...")
        from test_suite import run_all_tests
        run_all_tests()
        print("\n---> Running Microphone Access and Audio Pipeline Suite...")
        from test_mic_access import MicrophoneAccessTestSuite
        suite = MicrophoneAccessTestSuite()
        suite.run_all()
        print("\n---> Running Memory and Profiling Suite...")
        from test_memory_and_profiling import run_all_memory_profiling_tests
        run_all_memory_profiling_tests()

    # ── Option 0: Exit ──
    elif choice in ["0", "exit", "q"]:
        print("\nExiting Adaab...")
        sys.exit(0)

if __name__ == "__main__":
    main()
