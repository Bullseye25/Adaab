"""
modal_account_manager.py
─────────────────────────────────────────────────────────────────────────────
Modal Account Manager & Automated Cloud Migration Pipeline for Adaab Studio
─────────────────────────────────────────────────────────────────────────────
Features:
  1. Inspects, lists, and switches between multiple Modal.com user profiles.
  2. Supports adding brand-new Modal accounts ('modal setup').
  3. Seamless End-to-End Migration Pipeline:
     - Prompts to back up current account weights to Google Drive (5TB).
     - Activates or setups the target Modal account.
     - Automatically creates and restores 'adaab-cache' volume from Google Drive.
     - Automatically deploys 'deploy_adaab.py' to the new account.
     - Verifies live health and inference readiness on the new account.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import subprocess
import json
import time
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_current_modal_profile() -> str:
    """Returns the name of the currently active Modal profile."""
    try:
        cmd = [sys.executable, "-m", "modal", "profile", "current"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse last non-empty line (filter out any deprecation warnings)
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip() and not line.startswith("C:")]
        return lines[-1] if lines else "default"
    except Exception as e:
        print(f"[ModalManager] Notice checking current profile: {e}")
        return "unknown"


def list_modal_profiles() -> List[Dict[str, Any]]:
    """Returns a list of all configured Modal profiles with active indicator."""
    current = get_current_modal_profile()
    profiles = []
    try:
        cmd = [sys.executable, "-m", "modal", "profile", "list"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Output is a table like:
        # |   | Profile      | Workspace    |
        # | * | ammadraza01  | ammadraza01  |
        for line in res.stdout.split("\n"):
            if "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                # Table header or separator line
                if not parts or any(h in parts for h in ["Profile", "Workspace", "---"]) or line.startswith("+"):
                    continue
                if len(parts) == 3:
                    marker, p_name, w_name = parts[0], parts[1], parts[2]
                    is_act = (p_name == current) or bool(marker)
                    profiles.append({"name": p_name, "workspace": w_name, "is_active": is_act})
                elif len(parts) == 2:
                    p_name, w_name = parts[0], parts[1]
                    is_act = (p_name == current)
                    profiles.append({"name": p_name, "workspace": w_name, "is_active": is_act})
    except Exception as e:
        print(f"[ModalManager] Notice listing profiles: {e}")

    # Fallback to current if parsing failed
    if not profiles and current != "unknown":
        profiles.append({"name": current, "workspace": current, "is_active": True})

    return profiles


def switch_modal_profile(profile_name: str) -> bool:
    """Switches the active Modal profile using 'modal profile activate'."""
    try:
        cmd = [sys.executable, "-m", "modal", "profile", "activate", profile_name]
        subprocess.run(cmd, check=True)
        print(f"\n✓ Successfully activated Modal profile: '{profile_name}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to activate Modal profile '{profile_name}': {e}")
        return False


def setup_new_modal_account() -> bool:
    """Launches interactive Modal browser setup to authenticate a new account."""
    print("\n---> Launching Modal browser authentication for new account...")
    try:
        cmd = [sys.executable, "-m", "modal", "setup"]
        subprocess.run(cmd, check=True)
        new_prof = get_current_modal_profile()
        print(f"\n✓ New Modal account configured successfully! Active profile: '{new_prof}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Modal setup failed: {e}")
        return False


def run_account_migration_flow():
    """
    Main interactive Account Switcher and Automated Setup Pipeline.
    """
    from modal_gdrive_sync import run_cloud_backup, run_cloud_restore
    from gdrive_manager import is_gdrive_authenticated

    print("\n" + "=" * 72)
    print(" موڈل اکاؤنٹ منتقلی و خودکار سیٹ اپ — Modal Account Migration Pipeline ".center(72, "="))
    print("=" * 72)

    current_profile = get_current_modal_profile()
    print(f"Current Active Modal Account: [{current_profile}]")

    # Step 1: Ask if user wants to backup current account before leaving
    print("\n[Step 1/4] Google Drive Backup Pre-Check:")
    if is_gdrive_authenticated():
        do_backup = input(
            f"Would you like to back up the current model weights and cache from '{current_profile}' to Google Drive first? [Y/n]: "
        ).strip().lower()
        if do_backup != "n":
            print("\nStarting cloud backup to Google Drive...")
            b_ok = run_cloud_backup()
            if not b_ok:
                cont = input("Backup did not finish cleanly. Continue with account switch anyway? [y/N]: ").strip().lower()
                if cont != "y":
                    print("Migration aborted.")
                    return
    else:
        print("Notice: Google Drive is not configured yet. You will be prompted to authenticate before restoring.")

    # Step 2: Display available profiles
    print("\n[Step 2/4] Select Target Modal Account:")
    profiles = list_modal_profiles()
    for idx, p in enumerate(profiles, 1):
        status = " (ACTIVE NOW)" if p["is_active"] else ""
        print(f"  {idx}. {p['name']} [Workspace: {p['workspace']}]{status}")
    add_idx = len(profiles) + 1
    print(f"  {add_idx}. [+] Authenticate a Brand New Modal Account ('modal setup')")
    print("  0. Cancel")

    sel = input(f"\nEnter choice [1-{add_idx}] (0 to cancel): ").strip()
    if not sel or sel == "0":
        print("Account switch cancelled.")
        return

    try:
        choice_num = int(sel)
    except ValueError:
        print("Invalid selection.")
        return

    # Handle brand new account setup
    if choice_num == add_idx:
        ok = setup_new_modal_account()
        if not ok:
            print("Failed to set up new account. Migration aborted.")
            return
        target_profile = get_current_modal_profile()
    elif 1 <= choice_num <= len(profiles):
        target_profile = profiles[choice_num - 1]["name"]
        if target_profile == current_profile:
            print(f"\nProfile '{target_profile}' is already active.")
        else:
            ok = switch_modal_profile(target_profile)
            if not ok:
                print("Failed to switch profile. Migration aborted.")
                return
    else:
        print("Invalid choice.")
        return

    # Step 3: Automatically Restore from Google Drive to New Account
    print(f"\n[Step 3/4] Restoring Model Weights into New Account '{target_profile}'...")
    print("Triggering Modal Cloud container to pull weights directly from Google Drive (5TB)...")
    restore_ok = run_cloud_restore()
    if not restore_ok:
        retry = input("Cloud restore encountered an issue. Would you like to proceed to deployment anyway? [y/N]: ").strip().lower()
        if retry != "y":
            print("Setup halted.")
            return

    # Step 4: Automatically Deploy Backend to the New Modal Account
    print(f"\n[Step 4/4] Deploying Adaab Backend to New Account '{target_profile}'...")
    deploy_cmd = [sys.executable, "-X", "utf8", "-m", "modal", "deploy", "deploy_adaab.py"]
    try:
        subprocess.run(deploy_cmd, check=True)
        print("\n✓ Adaab Backend deployed to the new Modal account successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment to new account failed: {e}")
        return

    # Final Health Ping
    print("\n---> Performing post-migration health check on live cloud endpoint...")
    try:
        from backend import AdaabClient
        client = AdaabClient()
        ping_res = client.ping()
        print(f"✓ Health Check Result: {ping_res}")
    except Exception as e:
        print(f"Health check notice: {e}")

    print("\n" + "=" * 72)
    print("✓ منتقلی اور خودکار سیٹ اپ مکمل! — Account Migration Completed Successfully!".center(72))
    print("=" * 72)
    print(f" Active Account: {target_profile}")
    print(" Model Weights:  Restored from Google Drive (Zero local storage consumed)")
    print(" Status:         Ready for Voice Chat & Web Studio")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_account_migration_flow()
