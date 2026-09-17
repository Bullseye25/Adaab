"""
modal_gdrive_sync.py
─────────────────────────────────────────────────────────────────────────────
Direct Cloud-to-Cloud Google Drive Backup & Restore for Modal.com (Adaab Studio)
─────────────────────────────────────────────────────────────────────────────
Transfers multi-gigabyte fine-tuned weights, adapters, and cached models
directly between Modal Persistent Volumes and Google Drive (5TB).

KEY BENEFIT: ZERO BYTES OF LARGE MODEL WEIGHTS ARE DOWNLOADED TO OR STORED
ON YOUR PERSONAL COMPUTER. High-speed cloud network performs the entire sync.

Commands:
  Backup:  modal run modal_gdrive_sync.py::run_cloud_backup
  Restore: modal run modal_gdrive_sync.py::run_cloud_restore
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import tarfile
from typing import Optional, Dict, Any, List

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal Cloud App Definition
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("adaab-gdrive-sync")

# Volume where Adaab models and checkpoints reside
weights_vol = modal.Volume.from_name("adaab-cache", create_if_missing=True)

# Lightweight container image for cloud archiving and Google Drive API transfers
sync_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "google-api-python-client>=2.150.0",
        "google-auth-httplib2>=0.2.0",
        "google-auth-oauthlib>=1.2.0"
    )
)

CACHE_DIR = "/root/cache"


@app.function(
    image=sync_image,
    volumes={CACHE_DIR: weights_vol},
    timeout=3600,
    cpu=2.0,
    memory=4096
)
def cloud_backup_to_gdrive(token_json_str: str, backup_tag: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes inside Modal Cloud container:
    1. Reads token and connects to Google Drive API.
    2. Packages model checkpoints and adapters into a compressed tarball.
    3. Streams the tarball directly into Google Drive (5TB space).
    """
    import io
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    print("[Modal Cloud] Initializing Direct Cloud-to-Cloud Backup to Google Drive...")

    # Authenticate with Google Drive
    token_dict = json.loads(token_json_str)
    if "type" in token_dict and token_dict["type"] == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(
            token_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
    else:
        creds = Credentials.from_authorized_user_info(
            token_dict, ["https://www.googleapis.com/auth/drive.file"]
        )

    service = build("drive", "v3", credentials=creds)

    # 1. Locate or create root 'Adaab_AI_Studio' and '01_Model_Hub/LoRA_Adapters' folders
    def get_or_create_folder(name: str, parent_id: Optional[str] = None) -> str:
        q = f"mimeType = 'application/vnd.google-apps.folder' and name = '{name}' and trashed = false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        else:
            q += " and 'root' in parents"
        res = service.files().list(q=q, fields="files(id, name)", spaces="drive").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            body["parents"] = [parent_id]
        return service.files().create(body=body, fields="id").execute().get("id")

    root_id = get_or_create_folder("Adaab_AI_Studio")
    model_hub_id = get_or_create_folder("01_Model_Hub", parent_id=root_id)
    lora_folder_id = get_or_create_folder("LoRA_Adapters", parent_id=model_hub_id)
    snapshots_folder_id = get_or_create_folder("04_Modal_Account_Snapshots", parent_id=root_id)

    # 2. Check files in volume
    print(f"[Modal Cloud] Inspecting volume at {CACHE_DIR}...")
    items = os.listdir(CACHE_DIR) if os.path.exists(CACHE_DIR) else []
    print(f"[Modal Cloud] Volume items: {items}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tag = backup_tag or "weights_and_adapters"
    archive_filename = f"adaab_backup_{tag}_{timestamp}.tar.gz"
    tmp_archive_path = f"/tmp/{archive_filename}"

    # 3. Create compressed tarball in cloud container memory/disk
    print(f"[Modal Cloud] Compressing volume files into {tmp_archive_path}...")
    start_compress = time.time()
    with tarfile.open(tmp_archive_path, "w:gz") as tar:
        for item in items:
            full_path = os.path.join(CACHE_DIR, item)
            tar.add(full_path, arcname=item)
    compress_sec = round(time.time() - start_compress, 2)
    file_size_mb = round(os.path.getsize(tmp_archive_path) / (1024 * 1024), 2)
    print(f"[Modal Cloud] Compressed {file_size_mb} MB in {compress_sec}s.")

    # 4. Upload directly to Google Drive via Resumable chunked upload
    print(f"[Modal Cloud] Uploading {archive_filename} ({file_size_mb} MB) directly to Google Drive...")
    media = MediaFileUpload(tmp_archive_path, mimetype="application/gzip", resumable=True, chunksize=10*1024*1024)
    target_parent = lora_folder_id if "lora" in tag.lower() else snapshots_folder_id
    file_metadata = {
        "name": archive_filename,
        "parents": [target_parent],
        "description": f"Adaab AI Studio trained weights & volume snapshot created at {timestamp}"
    }

    req = service.files().create(body=file_metadata, media_body=media, fields="id, name, webViewLink, size")
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            print(f"[Modal Cloud] Upload progress: {int(status.progress() * 100)}%")

    # Cleanup temp container archive
    if os.path.exists(tmp_archive_path):
        os.remove(tmp_archive_path)

    file_id = response.get("id")
    web_link = response.get("webViewLink")
    print(f"[Modal Cloud] ✓ Direct upload complete! File ID: {file_id}")
    print(f"[Modal Cloud] View in Google Drive: {web_link}")

    return {
        "status": "success",
        "file_id": file_id,
        "file_name": archive_filename,
        "size_mb": file_size_mb,
        "web_link": web_link,
        "timestamp": timestamp
    }


@app.function(
    image=sync_image,
    volumes={CACHE_DIR: weights_vol},
    timeout=3600,
    cpu=2.0,
    memory=4096
)
def cloud_restore_from_gdrive(token_json_str: str, file_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes inside Modal Cloud container on target account:
    1. Downloads latest backup archive directly from Google Drive.
    2. Extracts directly into the active Modal Volume (/root/cache).
    3. Commits the volume so the new Modal account is instantly ready.
    """
    import io
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    print("[Modal Cloud] Initializing Direct Cloud-to-Cloud Restore from Google Drive...")

    # Authenticate with Google Drive
    token_dict = json.loads(token_json_str)
    if "type" in token_dict and token_dict["type"] == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(
            token_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
    else:
        creds = Credentials.from_authorized_user_info(
            token_dict, ["https://www.googleapis.com/auth/drive.file"]
        )

    service = build("drive", "v3", credentials=creds)

    target_file_id = file_id

    # If no file_id specified, find the newest 'adaab_backup_*.tar.gz' file
    if not target_file_id:
        print("[Modal Cloud] Searching for latest Adaab backup in Google Drive...")
        q = "name contains 'adaab_backup_' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name, createdTime, size)", orderBy="createdTime desc", spaces="drive").execute()
        files = res.get("files", [])
        if not files:
            return {"status": "error", "message": "No Adaab backups found in Google Drive."}
        target_file_id = files[0]["id"]
        target_filename = files[0]["name"]
        print(f"[Modal Cloud] Selected latest backup: '{target_filename}' ({target_file_id})")
    else:
        target_filename = "adaab_backup_archive.tar.gz"

    # Download from Google Drive directly to container /tmp
    tmp_archive_path = f"/tmp/{target_filename}"
    print(f"[Modal Cloud] Downloading {target_filename} from Google Drive...")
    request = service.files().get_media(fileId=target_file_id)
    with open(tmp_archive_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=10*1024*1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"[Modal Cloud] Download progress: {int(status.progress() * 100)}%")

    file_size_mb = round(os.path.getsize(tmp_archive_path) / (1024 * 1024), 2)
    print(f"[Modal Cloud] Downloaded {file_size_mb} MB successfully.")

    # Extract directly into /root/cache
    print(f"[Modal Cloud] Extracting archive directly into {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with tarfile.open(tmp_archive_path, "r:gz") as tar:
        tar.extractall(path=CACHE_DIR)

    # Commit persistent volume
    weights_vol.commit()
    print("[Modal Cloud] ✓ Persistent volume 'adaab-cache' committed successfully!")

    # Cleanup temp container file
    if os.path.exists(tmp_archive_path):
        os.remove(tmp_archive_path)

    restored_items = os.listdir(CACHE_DIR)
    return {
        "status": "success",
        "file_id": target_file_id,
        "restored_items": restored_items,
        "size_mb": file_size_mb
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local Entrypoints (Called from start.py / run_adaab)
# ─────────────────────────────────────────────────────────────────────────────

def run_cloud_backup():
    """Local trigger: acquires Google Drive token and launches Modal cloud backup."""
    from gdrive_manager import get_exportable_token_string, is_gdrive_authenticated, setup_gdrive_auth_interactive

    if not is_gdrive_authenticated():
        print("\nGoogle Drive is not authenticated yet.")
        ok = setup_gdrive_auth_interactive()
        if not ok:
            return False

    token_str = get_exportable_token_string()
    if not token_str:
        print("❌ Error: Could not load Google Drive credentials.")
        return False

    print("\n---> Triggering Direct Cloud-to-Cloud Backup on Modal (NVIDIA Cloud)...")
    print("Zero bytes of model weights will be stored on your local PC.")
    try:
        with modal.enable_output():
            with app.run():
                result = cloud_backup_to_gdrive.remote(token_str)
        if result.get("status") == "success":
            print("\n" + "=" * 65)
            print("✓ کلاؤڈ بیک اپ کامیاب! — Google Drive Backup Complete".center(65))
            print("=" * 65)
            print(f" Archive Name:  {result.get('file_name')}")
            print(f" Size:          {result.get('size_mb')} MB")
            print(f" Google Drive:  {result.get('web_link')}")
            print("=" * 65 + "\n")
            return True
        else:
            print(f"\n❌ Cloud backup notice: {result}")
            return False
    except Exception as ex:
        print(f"\n❌ Cloud backup error: {ex}")
        return False


def run_cloud_restore(file_id: Optional[str] = None):
    """Local trigger: acquires Google Drive token and launches Modal cloud restore."""
    from gdrive_manager import get_exportable_token_string, is_gdrive_authenticated, setup_gdrive_auth_interactive

    if not is_gdrive_authenticated():
        print("\nGoogle Drive is not authenticated yet.")
        ok = setup_gdrive_auth_interactive()
        if not ok:
            return False

    token_str = get_exportable_token_string()
    if not token_str:
        print("❌ Error: Could not load Google Drive credentials.")
        return False

    print("\n---> Triggering Direct Cloud-to-Cloud Restore into Active Modal Volume...")
    print("Weights will be pulled directly from Google Drive into Modal's persistent volume.")
    try:
        with modal.enable_output():
            with app.run():
                result = cloud_restore_from_gdrive.remote(token_str, file_id=file_id)
        if result.get("status") == "success":
            print("\n" + "=" * 65)
            print("✓ کلاؤڈ بحالی کامیاب! — Google Drive Restore Complete".center(65))
            print("=" * 65)
            print(f" Restored Items: {result.get('restored_items')}")
            print(f" Total Size:     {result.get('size_mb')} MB")
            print("=" * 65 + "\n")
            return True
        else:
            print(f"\n❌ Cloud restore notice: {result}")
            return False
    except Exception as ex:
        print(f"\n❌ Cloud restore error: {ex}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modal Cloud Google Drive Sync")
    parser.add_argument("--backup", action="store_true", help="Run cloud backup to Google Drive")
    parser.add_argument("--restore", action="store_true", help="Run cloud restore from Google Drive")
    args = parser.parse_args()

    if args.backup:
        run_cloud_backup()
    elif args.restore:
        run_cloud_restore()
    else:
        print("Usage: python modal_gdrive_sync.py --backup | --restore")
