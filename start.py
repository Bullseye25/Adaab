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

    print("=" * 68)
    print(" آداب (Adaab) - Autonomous Urdu Poetry & Multi-Modal Studio ".center(68, "="))
    print(" Serverless Qwen 2.5 7B on Modal (NVIDIA L4) | Scale-to-Zero ($0.00 Idle) ".center(68, " "))
    print("=" * 68)

    print("\n[Studio & Generation Options]")
    print(" 1. Quick 1-Click Batch Generator (Poem + 9:16 Poster + Voice + Reel)")
    print(" 2. Start Interactive Urdu Terminal Chat (Conversation CLI)")
    print(" 3. Run 8-Point Automated Unit & Integration Test Suite")
    print(" 4. Start Adaab Web Studio (Browser UI with Nastaliq & Audio/Video Player)")

    print("\n[Modal.com Cloud Deployment & Storage]")
    print(" 5. Deploy / Redeploy Adaab Backend to Modal (NVIDIA L4 GPU)")
    print(" 6. Pre-cache Qwen 2.5 7B Weights into Modal Volume ('adaab-cache')")
    print(" 7. Test Deployed Cloud Backend & Verify Urdu Inference")
    print(" 8. Change / Setup Modal Account (runs 'modal setup')")
    print(" 9. Open Modal Deployment Dashboard in Browser")
    print(" 10. View Live GPU Telemetry & Crash Diagnostic Logs")

    print("\n[Local Outputs & Files]")
    print(" 11. Open Local Output Folder in Windows File Explorer")
    print(" 12. View SQLite Database History & Stats (adaab_history.db)")
    print(" 13. Clear / Reset SQLite History Database (Delete all recorded verses)")

    choice = input("\nEnter choice [1-13] (default 1): ").strip() or "1"

    # ── Option 1: Quick Batch Generator ──
    if choice == "1":
        from generate_batch import main as batch_main
        batch_main()

    # ── Option 2: Interactive Terminal Chat ──
    elif choice == "2":
        from backend import AdaabClient
        from agent import ADAAB_SYSTEM_PROMPT

        client = AdaabClient()
        print("\n" + "=" * 68)
        print(" آداب (Adaab) - Interactive Urdu Conversation ".center(68, "="))
        print(" گفتگو کے لیے سوال لکھیں یا شاعری کی فرمائش کریں۔ (Type 'exit' to quit)")
        print("=" * 68 + "\n")

        history = [{"role": "system", "content": ADAAB_SYSTEM_PROMPT}]
        while True:
            try:
                user_msg = input("\n[آپ / You]: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    print("\nآداب عرض ہے! خدا حافظ۔ (Farewell!)")
                    break

                history.append({"role": "user", "content": user_msg})
                print("\n[آداب / Adaab سوچ رہا ہے...]")
                resp = client.chat_completion(history, temperature=0.7, max_tokens=800)
                assistant_text = resp["content"]
                print(f"\n[آداب]:\n{assistant_text}")
                history.append({"role": "assistant", "content": assistant_text})
            except KeyboardInterrupt:
                print("\nExiting session...")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {e}")

    # ── Option 3: Run Unit Test Suite ──
    elif choice == "3":
        print("\n---> Running 8-Point Automated Unit & Integration Suite...")
        from test_suite import run_all_tests
        run_all_tests()

    # ── Option 4: Start Gradio Web Studio ──
    elif choice == "4":
        print("\n---> Starting Adaab Web Studio...")
        try:
            import gradio
        except ImportError:
            install_package("gradio", "gradio>=5.0")
        cmd = [sys.executable, "app.py"]
        subprocess.run(cmd)

    # ── Option 5: Deploy to Modal ──
    elif choice == "5":
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

    # ── Option 6: Pre-cache Weights into Modal Volume ──
    elif choice == "6":
        print("\n---> Caching Qwen 2.5 7B weights into Modal Volume 'adaab-cache'...")
        try:
            import modal
        except ImportError:
            install_package("modal")
        cmd = [sys.executable, "-X", "utf8", "-m", "modal", "run", "deploy_adaab.py::download_adaab_weights"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Weights cached in Modal volume successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Pre-caching failed: {e}")

    # ── Option 7: Test Cloud Backend ──
    elif choice == "7":
        print("\n---> Testing live Modal backend inference...")
        from backend import AdaabClient
        client = AdaabClient()
        test_messages = [
            {"role": "system", "content": "You are Adaab, a polite Urdu AI."},
            {"role": "user", "content": "السلام علیکم! اپنا مختصر تعارف کروائیں۔"}
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

    # ── Option 8: Modal Setup ──
    elif choice == "8":
        print("\n---> Setting up Modal account...")
        try:
            import modal
        except ImportError:
            install_package("modal")
        cmd = [sys.executable, "-X", "utf8", "-m", "modal", "setup"]
        try:
            subprocess.run(cmd, check=True)
            print("\n✓ Modal setup completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Modal setup failed: {e}")

    # ── Option 9: Open Dashboard ──
    elif choice == "9":
        dashboard_url = "https://modal.com/apps"
        print(f"\nOpening Modal Dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)

    # ── Option 10: View GPU Telemetry & Crash Logs ──
    elif choice == "10":
        from crash_tracker import get_gpu_telemetry, get_latest_crash_report
        print("\n=== Live GPU Telemetry ===")
        import json
        print(json.dumps(get_gpu_telemetry(), indent=2))
        print("\n=== Recent Crash & Diagnostic Events ===")
        print(get_latest_crash_report())

    # ── Option 11: Open Output Folder ──
    elif choice == "11":
        output_dir = os.path.join(script_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(output_dir)
            print(f"\n✓ Opened {output_dir} in File Explorer.")

    # ── Option 12: Database Stats ──
    elif choice == "12":
        from poetry_db import get_history_summary
        stats = get_history_summary()
        print("\n=== Adaab History Database Summary ===")
        print(f"Total Unique Posts Generated: {stats['total_posts']}")
        print("\nRecent Verses in Catalog:")
        for p in stats["recent_posts"]:
            print(f" • [{p['theme']}] {p['misra_1']} / {p['misra_2']} ({p['created_at']})")

    # ── Option 13: Clear Database ──
    elif choice == "13":
        print("\n⚠️ WARNING: This will permanently wipe all recorded verses and history in adaab_history.db.")
        confirm = input("Are you sure you want to clear the entire database? [y/N]: ").strip().lower()
        if confirm == "y":
            from poetry_db import clear_database
            if clear_database():
                print("✓ SQLite database cleared and reset successfully!")
            else:
                print("❌ Failed to clear database.")
        else:
            print("Operation cancelled. Database untouched.")

if __name__ == "__main__":
    main()
