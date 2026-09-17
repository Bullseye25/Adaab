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

    print("=" * 70)
    print(" Adaab — Conversational Voice AI Studio ".center(70, "="))
    print(" Serverless Qwen 2.5 7B on Modal (NVIDIA L4) | Scale-to-Zero ".center(70, " "))
    print("=" * 70)

    print("\n[Voice Assistant and Web Studio]")
    print(" 1. Launch Web Studio (Push-to-Talk, Voice Chat, Mobile QR) [Default]")
    print(" 2. Interactive Terminal Voice Chat with Tabraiz")
    print(" 3. Run Automated System Test Suite")

    print("\n[AI Knowledge Oracles (Gemini & ChatGPT)]")
    print(" 4. Setup ChatGPT Browser Session (Login & Save Session Cookies)")
    print(" 5. Test ChatGPT Web Oracle Live Inference")
    print(" 6. Check Google Gemini Oracle & Quota Status")

    print("\n[Modal GPU Cloud Backend & LoRA Training]")
    print(" 7. Deploy / Redeploy Adaab Backend to Modal (NVIDIA L4 GPU)")
    print(" 8. Test Live Cloud Backend and Verify Inference")
    print(" 9. View Live GPU Telemetry and Diagnostic Logs")
    print(" 10. Open Modal Cloud Dashboard in Browser")
    print(" 11. Manage / Add Topics to LoRA Training Dataset (dataset_manager.py)")
    print(" 12. Fine-Tune Pakistan Knowledge LoRA on Modal GPU (NVIDIA L4)")

    print("\n[Google Drive Cloud Backup & Migration (5TB)]")
    print(" 13. Backup Trained Model & Weights to Google Drive")
    print(" 14. Restore Model & Weights from Google Drive to Modal")
    print(" 15. Switch Modal.com Account & Auto-Setup from Google Drive")
    print(" 16. Configure / Authenticate Google Drive (5TB Quota Check)")

    print("\n[Memory and Topics Database]")
    print(" 17. View Recorded Conversation Topics & SQLite Memory")
    print(" 18. Clear / Reset Conversation and User Database")
    print(" 0. Exit")

    choice = input("\nEnter choice [0-18] (default 1 for Web Studio): ").strip().lower() or "1"

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

    # ── Option 2: Interactive Terminal Chat with Tabraiz ──
    elif choice in ["2", "chat", "tabraiz"]:
        from backend import AdaabClient
        from agent import handle_conversation_turn
        from time_context import get_local_time_context

        client = AdaabClient()
        client.start_heartbeat(interval_sec=48)
        t_ctx = get_local_time_context()
        print("\n" + "=" * 70)
        print(" Tabraiz — Conversational Voice & Reasoning Assistant ".center(70, "="))
        print(f" Local Clock: {t_ctx['urdu_full_str']}")
        print(" Casual Polite Urdu | Time-Aware | Grounded Knowledge")
        print(" (Type 'exit' to quit)")
        print("=" * 70 + "\n")

        history = []
        initial_greeting = "آداب! میں تبریز ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"
        print(f"[Tabraiz]: {initial_greeting}\n")
        history.append({"role": "assistant", "content": initial_greeting})
        active_profile = {}

        while True:
            try:
                user_msg = input("\n[You]: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    print("\n[Tabraiz]: Goodbye! It was a pleasure talking with you.")
                    break

                history.append({"role": "user", "content": user_msg})
                print("\n[Tabraiz is thinking...]")

                assistant_text, updated_prof = handle_conversation_turn(
                    user_message=user_msg,
                    history=history,
                    client=client,
                    persona="Tabraiz",
                    active_profile=active_profile
                )
                if updated_prof:
                    active_profile = updated_prof

                print(f"\n[Tabraiz]:\n{assistant_text}")
                history.append({"role": "assistant", "content": assistant_text})
            except KeyboardInterrupt:
                print("\nExiting session...")
                break
            except Exception as e:
                print(f"\nError: {e}")

    # ── Option 3: Run Automated Test Suite ──
    elif choice in ["3", "test", "tests"]:
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

    # ── Option 4: Setup ChatGPT Browser Session (Login & Save Session Cookies) ──
    elif choice in ["4", "chatgpt_login", "chatgpt_setup"]:
        from chatgpt_browser_oracle import get_chatgpt_oracle
        oracle = get_chatgpt_oracle()
        oracle.launch_login_session()

    # ── Option 5: Test ChatGPT Web Oracle Live Inference ──
    elif choice in ["5", "chatgpt_test", "chatgpt"]:
        from chatgpt_browser_oracle import get_chatgpt_oracle
        oracle = get_chatgpt_oracle()
        if not oracle.is_available():
            print("\n⚠️ ChatGPT session is not configured yet.")
            print("Please select Option 4 first to log into your ChatGPT account.")
        else:
            print("\n" + "=" * 70)
            print(" ChatGPT Web Oracle — Live Inference Test ".center(70, "="))
            print("=" * 70)
            test_prompt = input("Enter test question (press Enter for default): ").strip()
            if not test_prompt:
                test_prompt = "پاکستان کا دارالحکومت کیا ہے اور اس کی خاص بات کیا ہے؟"
            print(f"\nQuerying ChatGPT via browser session...")
            print(f"Prompt: {test_prompt}")
            t0 = time.time()
            reply = oracle.query(prompt=test_prompt, timeout=45)
            elapsed = round(time.time() - t0, 2)
            if reply:
                print("\n--- Response from ChatGPT Web Oracle ---")
                print(reply)
                print("-" * 50)
                print(f"Completed in {elapsed}s | Cookies and session verified.")
            else:
                print("\n❌ Did not receive a response within timeout. Please ensure the ChatGPT tab is ready.")

    # ── Option 6: Check Google Gemini Oracle & Quota Status ──
    elif choice in ["6", "gemini", "gemini_status"]:
        from gemini_oracle import get_gemini_oracle
        oracle = get_gemini_oracle()
        print("\n" + "=" * 70)
        print(" Google Gemini Knowledge Oracle — Status & Diagnostics ".center(70, "="))
        print("=" * 70)
        status = oracle.get_circuit_status()
        print(f" API Key Configured:     {status['configured']}")
        print(f" Circuit Breaker Tripped: {status['circuit_open']}")
        print(f" Cooldown Remaining:     {status['seconds_remaining_in_cooldown']} seconds")
        print(f" Requests in Last Min:   {status['rpm_count_last_minute']}")
        if status['tripped_reason']:
            print(f" Reason for Trip:        {status['tripped_reason']}")
        
        from chatgpt_browser_oracle import get_chatgpt_oracle
        cg = get_chatgpt_oracle()
        print(f" ChatGPT Fallback Ready: {cg.is_available()}")
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

    # ── Option 7: Deploy to Modal ──
    elif choice in ["7", "deploy"]:
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

    # ── Option 8: Test Cloud Backend ──
    elif choice in ["8", "test_modal"]:
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

    # ── Option 9: View GPU Telemetry & Crash Logs ──
    elif choice in ["9", "telemetry", "logs"]:
        from crash_tracker import get_gpu_telemetry, get_latest_crash_report
        print("\n=== Live GPU Telemetry ===")
        import json
        print(json.dumps(get_gpu_telemetry(), indent=2))
        print("\n=== Recent Crash and Diagnostic Events ===")
        print(get_latest_crash_report())

    # ── Option 10: Open Modal Dashboard ──
    elif choice in ["10", "dashboard"]:
        dashboard_url = "https://modal.com/apps"
        print(f"\nOpening Modal Dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)

    # ── Option 11: Manage & Add Topics to LoRA Training Dataset ──
    elif choice in ["11", "dataset", "topics_dataset", "add_topic"]:
        from dataset_manager import show_dataset_summary, interactive_add_topic, build_master_dataset
        show_dataset_summary()
        print("[Dataset Operations]")
        print(" 1. Add a New Topic / Q&A Pair Interactively")
        print(" 2. Rebuild & Validate Master Training Dataset")
        print(" 3. Return to Main Menu")
        sub_c = input("\nEnter choice [1-3] (default 1): ").strip() or "1"
        if sub_c == "1":
            interactive_add_topic()
        elif sub_c == "2":
            master_file = build_master_dataset()
            print(f"\n✓ Master dataset updated and validated at: {master_file}")
            show_dataset_summary()

    # ── Option 12: Fine-Tune Pakistan Knowledge LoRA on Modal GPU ──
    elif choice in ["12", "train", "finetune"]:
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

    # ── Option 13: Backup Trained Model & Weights to Google Drive ──
    elif choice in ["13", "backup", "gdrive_backup"]:
        from modal_gdrive_sync import run_cloud_backup
        run_cloud_backup()

    # ── Option 14: Restore Model & Weights from Google Drive to Modal ──
    elif choice in ["14", "restore", "gdrive_restore"]:
        from modal_gdrive_sync import run_cloud_restore
        run_cloud_restore()

    # ── Option 15: Switch Modal.com Account & Auto-Setup from Google Drive ──
    elif choice in ["15", "switch", "migrate", "account"]:
        from modal_account_manager import run_account_migration_flow
        run_account_migration_flow()

    # ── Option 16: Configure / Authenticate Google Drive (5TB) ──
    elif choice in ["16", "gdrive", "auth"]:
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

    # ── Option 17: View Conversation Topics & Memory ──
    elif choice in ["17", "memory", "topics", "profiles"]:
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

    # ── Option 18: Clear Memory Database ──
    elif choice in ["18", "clear_memory"]:
        print("\nWARNING: This will permanently wipe all conversation turns and profile records in adaab_history.db.")
        confirm = input("Are you sure you want to clear the memory database? [y/N]: ").strip().lower()
        if confirm == "y":
            from memory_engine import clear_user_memory
            clear_user_memory()
            print("✓ SQLite conversation and user memory cleared successfully!")
        else:
            print("Operation cancelled. Database untouched.")

    # ── Option 0: Exit ──
    elif choice in ["0", "exit", "q"]:
        print("\nExiting Adaab...")
        sys.exit(0)

if __name__ == "__main__":
    main()
