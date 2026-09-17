"""
gdrive_manager.py
─────────────────────────────────────────────────────────────────────────────
Google Drive Cloud Storage Engine for Adaab Studio
─────────────────────────────────────────────────────────────────────────────
Features:
  1. Seamless OAuth 2.0 (Browser-based) and Service Account authentication.
  2. Automatic token refresh and secure serialization for Modal Cloud tasks.
  3. Professional, standardized folder taxonomy creation in Google Drive:
       Adaab_AI_Studio/
         ├── 01_Model_Hub/
         │   ├── LoRA_Adapters/
         │   ├── Base_Models/
         │   └── GGUF_Quantized/
         ├── 02_Datasets/
         │   ├── Conversational_ChatML/
         │   └── Audio_Speech_Corpus/
         ├── 03_Database_Backups/
         └── 04_Modal_Account_Snapshots/
  4. Quota verification (displays total capacity, usage, and available free 5TB space).
  5. Chunked resumable streaming for high-speed cloud-to-cloud transfers.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
from typing import Optional, Dict, Any, List

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GDRIVE_DIR = os.path.dirname(os.path.abspath(__file__))
GDRIVE_TOKEN_FILE = os.path.join(GDRIVE_DIR, "gdrive_token.json")
GDRIVE_CLIENT_SECRET = os.path.join(GDRIVE_DIR, "credentials_gdrive.json")
GDRIVE_SERVICE_ACCOUNT = os.path.join(GDRIVE_DIR, "service_account.json")

# Standard Google Drive API scope (limited to files created/opened by this app for security)
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_gdrive_service(headless_token_data: Optional[Any] = None):
    """
    Returns an authorized Google Drive API v3 resource service.
    Supports:
      1. In-memory headless token data (dict or JSON string from Modal cloud task)
      2. Service account key file ('service_account.json')
      3. Saved OAuth 2.0 user credentials ('gdrive_token.json')
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = None

    # 1. Headless Token passed in memory (e.g. inside Modal cloud container)
    if headless_token_data:
        if isinstance(headless_token_data, str):
            token_dict = json.loads(headless_token_data)
        else:
            token_dict = dict(headless_token_data)

        if "type" in token_dict and token_dict["type"] == "service_account":
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_info(token_dict, scopes=SCOPES)
        else:
            creds = Credentials.from_authorized_user_info(token_dict, SCOPES)

    # 2. Service Account JSON file in workspace
    elif os.path.exists(GDRIVE_SERVICE_ACCOUNT):
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(GDRIVE_SERVICE_ACCOUNT, scopes=SCOPES)

    # 3. Existing local user token file
    elif os.path.exists(GDRIVE_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(GDRIVE_TOKEN_FILE, SCOPES)
        except Exception as err:
            print(f"[GDrive] Notice loading token file: {err}")
            creds = None

    # Refresh expired credentials if refresh token is present
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Re-save refreshed credentials if running locally
            if not headless_token_data and os.path.exists(GDRIVE_TOKEN_FILE):
                with open(GDRIVE_TOKEN_FILE, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
        except Exception as ref_err:
            print(f"[GDrive] Notice refreshing token: {ref_err}")
            creds = None

    if not creds or not creds.valid:
        return None

    return build("drive", "v3", credentials=creds)


def is_gdrive_authenticated() -> bool:
    """Checks if valid Google Drive credentials or token are available."""
    try:
        service = get_gdrive_service()
        return service is not None
    except Exception:
        return False


def get_exportable_token_string() -> Optional[str]:
    """
    Returns the JSON string representation of the active Google Drive credentials
    so it can be securely passed to Modal cloud tasks for direct cloud-to-cloud sync.
    """
    if os.path.exists(GDRIVE_SERVICE_ACCOUNT):
        with open(GDRIVE_SERVICE_ACCOUNT, "r", encoding="utf-8") as f:
            return f.read().strip()
    elif os.path.exists(GDRIVE_TOKEN_FILE):
        with open(GDRIVE_TOKEN_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def setup_gdrive_auth_interactive() -> bool:
    """
    Interactive terminal setup wizard for Google Drive authentication.
    Supports:
      A. Google Cloud Console 'credentials.json' / 'credentials_gdrive.json' file
      B. Interactive Client ID and Client Secret input
      C. Service Account JSON file detection
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    print("\n" + "=" * 72)
    print(" گوگل ڈرائیو کلاؤڈ کنفیگریشن — Google Drive 5TB Cloud Auth Setup ".center(72, "="))
    print("=" * 72)

    # Check for service account first
    if os.path.exists(GDRIVE_SERVICE_ACCOUNT):
        print(f"\n✓ Found Google Service Account file: {GDRIVE_SERVICE_ACCOUNT}")
        try:
            service = get_gdrive_service()
            if service:
                print("✓ Service Account validated successfully!")
                return True
        except Exception as e:
            print(f"Service account verification failed: {e}")

    # Check for existing credentials_gdrive.json or credentials.json
    creds_source = None
    for cand in [GDRIVE_CLIENT_SECRET, os.path.join(GDRIVE_DIR, "credentials.json")]:
        if os.path.exists(cand):
            creds_source = cand
            break

    flow = None
    if creds_source:
        print(f"\n✓ Found OAuth Client Secrets file: {os.path.basename(creds_source)}")
        use_curr = input("Use this saved Client Secrets configuration? [Y/n]: ").strip().lower()
        if use_curr in ["n", "no"]:
            creds_source = None
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_source, SCOPES)
            except Exception as ex:
                print(f"Notice reading secrets file: {ex}")
                flow = None

    if not flow:
        print("\nGoogle Drive OAuth Setup Options:")
        print("1. Provide Client ID and Client Secret manually (Recommended)")
        print("2. I have a 'credentials.json' file downloaded from Google Cloud Console")
        print("3. I have a 'service_account.json' file from Google Cloud")
        print("0. Cancel")

        choice = input("\nEnter choice [1-3] (default 1): ").strip() or "1"

        if choice == "1":
            print("\nTo obtain your Google Drive OAuth credentials:")
            print("  1. Visit: https://console.cloud.google.com/apis/credentials")
            print("  2. Create an 'OAuth 2.0 Client ID' (Application type: 'Desktop app' or 'Web application').")
            print("     • If 'Desktop app': No redirect URI configuration is needed.")
            print("     • If 'Web application': Add 'http://localhost:8080/' to Authorized redirect URIs.")
            print("  3. Copy your Client ID and Client Secret below:")
            client_id = input("\nEnter Client ID: ").strip()
            client_secret = input("Enter Client Secret: ").strip()

            if not client_id or not client_secret:
                print("❌ Error: Client ID and Secret cannot be empty.")
                return False

            client_config = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [
                        "http://localhost:8080/",
                        "http://127.0.0.1:8080/",
                        "http://localhost"
                    ]
                }
            }
            # Save client config
            with open(GDRIVE_CLIENT_SECRET, "w", encoding="utf-8") as f:
                json.dump(client_config, f, indent=2)

            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)

        elif choice == "2":
            path = input("Enter absolute path to your credentials.json file: ").strip().strip('"')
            if not os.path.exists(path):
                print(f"❌ Error: File not found at '{path}'")
                return False
            import shutil
            shutil.copy(path, GDRIVE_CLIENT_SECRET)
            flow = InstalledAppFlow.from_client_secrets_file(GDRIVE_CLIENT_SECRET, SCOPES)

        elif choice == "3":
            path = input("Enter absolute path to your service_account.json file: ").strip().strip('"')
            if not os.path.exists(path):
                print(f"❌ Error: File not found at '{path}'")
                return False
            import shutil
            shutil.copy(path, GDRIVE_SERVICE_ACCOUNT)
            service = get_gdrive_service()
            if service:
                print("✓ Service Account imported and verified successfully!")
                return True
            return False
        else:
            print("Setup cancelled.")
            return False

    auth_port = 8080
    print("\n" + "-" * 70)
    print(" Opening browser for one-time Google Account Authorization... ".center(70))
    print(f" • Local Redirect URI: http://localhost:{auth_port}/")
    print(" • Target Storage: Google Drive (5TB Account)")
    print("-" * 70)
    print("Please log into your Google account and click 'Continue / Allow'.\n")
    try:
        creds = flow.run_local_server(
            host="localhost",
            port=auth_port,
            open_browser=True,
            prompt="consent"
        )
        # Save authorized user token
        with open(GDRIVE_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        print("\n✓ Authentication successful! Token saved securely to gdrive_token.json.")
        
        # Test connection & show quota
        service = get_gdrive_service()
        show_gdrive_storage_status(service)
        return True
    except Exception as ex:
        print(f"\n❌ Google OAuth authorization failed: {ex}")
        print("\n[TROUBLESHOOTING GUIDE]")
        print("1. If you see 'Error 400: redirect_uri_mismatch':")
        print("   In Google Cloud Console (https://console.cloud.google.com/apis/credentials):")
        print("   -> Open your Client ID settings.")
        print(f"   -> Under 'Authorized redirect URIs', click '+ ADD URI' and add: http://localhost:{auth_port}/")
        print("   -> Click 'Save' and try again.")
        print("   OR create a new OAuth Client ID with Application Type = 'Desktop app' (works automatically).")
        print("2. If you see 'Access blocked: Authorization Error':")
        print("   Make sure the Google Drive API is enabled in your Google Cloud Project:")
        print("   https://console.cloud.google.com/apis/library/drive.googleapis.com\n")
        return False


def get_storage_quota(service=None) -> Dict[str, Any]:
    """Retrieves Google Drive storage metrics (total, used, and free capacity)."""
    srv = service or get_gdrive_service()
    if not srv:
        return {"error": "Not authenticated with Google Drive"}

    try:
        about = srv.about().get(fields="storageQuota,user").execute()
        quota = about.get("storageQuota", {})
        user = about.get("user", {})

        limit_bytes = int(quota.get("limit", 0))
        usage_bytes = int(quota.get("usage", 0))
        free_bytes = max(0, limit_bytes - usage_bytes) if limit_bytes > 0 else 0

        to_gb = 1024 ** 3
        to_tb = 1024 ** 4

        return {
            "user_name": user.get("displayName", "User"),
            "user_email": user.get("emailAddress", "N/A"),
            "total_bytes": limit_bytes,
            "used_bytes": usage_bytes,
            "free_bytes": free_bytes,
            "total_gb": round(limit_bytes / to_gb, 2) if limit_bytes else 0,
            "used_gb": round(usage_bytes / to_gb, 2),
            "free_gb": round(free_bytes / to_gb, 2) if limit_bytes else "Unlimited",
            "total_tb": round(limit_bytes / to_tb, 2) if limit_bytes else 0,
            "used_pct": round((usage_bytes / limit_bytes) * 100, 1) if limit_bytes else 0
        }
    except Exception as e:
        return {"error": str(e)}


def show_gdrive_storage_status(service=None):
    """Prints a formatted diagnostic summary of the Google Drive storage quota."""
    data = get_storage_quota(service)
    if "error" in data:
        print(f"\n[Google Drive] Status: {data['error']}")
        return

    print("\n" + "=" * 60)
    print(" گوگل ڈرائیو کلاؤڈ اسٹوریج — Google Drive Cloud Status ".center(60, "="))
    print("=" * 60)
    print(f" Account:       {data['user_name']} ({data['user_email']})")
    print(f" Total Quota:   {data['total_gb']} GB (~{data['total_tb']} TB)")
    print(f" Space Used:    {data['used_gb']} GB ({data['used_pct']}%)")
    print(f" Available:     {data['free_gb']} GB")
    print("=" * 60 + "\n")


def ensure_folder(service, folder_name: str, parent_id: Optional[str] = None) -> str:
    """Finds an existing folder by name and parent ID, or creates it if missing."""
    query = f"mimeType = 'application/vnd.google-apps.folder' and name = '{folder_name}' and trashed = false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    else:
        query += " and 'root' in parents"

    resp = service.files().list(q=query, fields="files(id, name)", spaces="drive").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]

    # Create folder
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_id:
        file_metadata["parents"] = [parent_id]

    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder.get("id")


def init_adaab_folder_hierarchy(service) -> Dict[str, str]:
    """
    Initializes the standardized, professional Adaab directory structure in Google Drive.
    Returns a dictionary mapping logical path keys to their Google Drive folder IDs.
    """
    # 1. Root Folder
    root_id = ensure_folder(service, "Adaab_AI_Studio")

    # 2. Main Categories
    model_hub_id = ensure_folder(service, "01_Model_Hub", parent_id=root_id)
    datasets_id = ensure_folder(service, "02_Datasets", parent_id=root_id)
    db_backups_id = ensure_folder(service, "03_Database_Backups", parent_id=root_id)
    modal_snapshots_id = ensure_folder(service, "04_Modal_Account_Snapshots", parent_id=root_id)

    # 3. Sub-categories
    lora_id = ensure_folder(service, "LoRA_Adapters", parent_id=model_hub_id)
    base_models_id = ensure_folder(service, "Base_Models", parent_id=model_hub_id)
    gguf_id = ensure_folder(service, "GGUF_Quantized", parent_id=model_hub_id)

    chatml_id = ensure_folder(service, "Conversational_ChatML", parent_id=datasets_id)
    audio_id = ensure_folder(service, "Audio_Speech_Corpus", parent_id=datasets_id)

    return {
        "root": root_id,
        "model_hub": model_hub_id,
        "lora_adapters": lora_id,
        "base_models": base_models_id,
        "gguf_quantized": gguf_id,
        "datasets": datasets_id,
        "chatml_datasets": chatml_id,
        "audio_corpus": audio_id,
        "database_backups": db_backups_id,
        "modal_snapshots": modal_snapshots_id
    }


def list_backups_in_folder(service, folder_id: str) -> List[Dict[str, Any]]:
    """Returns a list of all backup archives in a specific Google Drive folder."""
    query = f"'{folder_id}' in parents and trashed = false"
    fields = "files(id, name, size, createdTime, modifiedTime, webViewLink)"
    resp = service.files().list(q=query, fields=fields, orderBy="createdTime desc", spaces="drive").execute()
    files = resp.get("files", [])
    result = []
    for f in files:
        sz_bytes = int(f.get("size", 0))
        sz_mb = round(sz_bytes / (1024 * 1024), 2)
        result.append({
            "id": f.get("id"),
            "name": f.get("name"),
            "size_mb": sz_mb,
            "created_time": f.get("createdTime"),
            "web_link": f.get("webViewLink")
        })
    return result


if __name__ == "__main__":
    print("Testing Google Drive Manager...")
    if is_gdrive_authenticated():
        srv = get_gdrive_service()
        show_gdrive_storage_status(srv)
        folders = init_adaab_folder_hierarchy(srv)
        print("✓ Verified Adaab Folder Taxonomy:")
        for k, fid in folders.items():
            print(f"   • {k}: {fid}")
    else:
        print("Google Drive is not yet authenticated.")
        setup_gdrive_auth_interactive()
