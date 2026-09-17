import os
import sys
import json
import traceback
from datetime import datetime
from functools import wraps

LOCAL_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOCAL_LOG_FILE = os.path.join(LOCAL_LOG_DIR, "error_report.log")
CLOUD_LOG_FILE = "/root/cache/logs/crash_events.jsonl"

def ensure_local_log_dir():
    os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

def get_gpu_telemetry() -> dict:
    """
    Captures live NVIDIA GPU telemetry via PyTorch.
    Works seamlessly on both Modal cloud containers and local GPU (if present).
    """
    telemetry = {
        "cuda_available": False,
        "device_name": "None",
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "total_memory_mb": 0.0
    }
    try:
        import torch
        if torch.cuda.is_available():
            telemetry["cuda_available"] = True
            telemetry["device_name"] = torch.cuda.get_device_name(0)
            telemetry["allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
            telemetry["reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            telemetry["max_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
            props = torch.cuda.get_device_properties(0)
            telemetry["total_memory_mb"] = round(props.total_memory / (1024 * 1024), 2)
    except Exception as e:
        telemetry["error"] = str(e)
    return telemetry

def log_error_event(stage: str, error: Exception, context: dict = None, is_cloud: bool = False):
    """
    Structured crash and error logger.
    Mirrors to local log file and cloud volume JSONL log.
    """
    ensure_local_log_dir()
    gpu_stats = get_gpu_telemetry()
    timestamp = datetime.now().isoformat()
    stack_trace = traceback.format_exc()

    event = {
        "timestamp": timestamp,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "gpu_telemetry": gpu_stats,
        "context": context or {},
        "stack_trace": stack_trace
    }

    # Format human-readable entry for local error_report.log
    log_text = (
        f"\n{'='*70}\n"
        f" [CRASH REPORT] {timestamp} | Stage: {stage}\n"
        f"{'='*70}\n"
        f"Error Type: {event['error_type']}\n"
        f"Message: {event['error_message']}\n"
        f"GPU Status: {gpu_stats['device_name']} | "
        f"Allocated: {gpu_stats['allocated_mb']}MB / {gpu_stats['total_memory_mb']}MB | "
        f"Reserved: {gpu_stats['reserved_mb']}MB\n"
        f"Context: {json.dumps(context or {}, ensure_ascii=False)}\n"
        f"--- Traceback ---\n{stack_trace}\n"
        f"{'='*70}\n"
    )

    try:
        with open(LOCAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_text)
    except Exception as e:
        print(f"[CrashTracker] Local file write failed: {e}")

    # If running inside Modal cloud container, append to persistent volume log
    if is_cloud or os.path.exists("/root/cache"):
        try:
            os.makedirs(os.path.dirname(CLOUD_LOG_FILE), exist_ok=True)
            with open(CLOUD_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[CrashTracker] Cloud volume log write failed: {e}")

    # Also log into SQLite telemetry
    try:
        from memory_engine import log_telemetry_event
        log_telemetry_event(
            event_type=f"CRASH_{stage.upper()}",
            details=json.dumps(context or {}, ensure_ascii=False),
            vram_allocated_mb=gpu_stats.get("allocated_mb", 0.0),
            vram_reserved_mb=gpu_stats.get("reserved_mb", 0.0),
            error_msg=f"{event['error_type']}: {event['error_message']}"
        )
    except Exception:
        pass

def crash_protected(stage_name: str = "operation"):
    """
    Decorator that wraps operations with instant GPU crash tracking,
    automatic VRAM cache clearing on CUDA OOM, and descriptive logging.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Handle CUDA OOM specially
                is_oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
                if is_oom:
                    print(f"\n⚠️ [GPU CRASH DETECTED] CUDA Out of Memory during {stage_name}!")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        print("✓ Emptied PyTorch CUDA cache.")
                    except Exception:
                        pass

                context = {
                    "function": func.__name__,
                    "args_repr": str(args)[:300],
                    "kwargs_keys": list(kwargs.keys()),
                    "is_oom": is_oom
                }
                log_error_event(stage=stage_name, error=exc, context=context)
                raise exc
        return wrapper
    return decorator

def get_latest_crash_report() -> str:
    """Reads the most recent entries from error_report.log."""
    if not os.path.exists(LOCAL_LOG_FILE):
        return "✓ No crash events recorded. System health is normal."
    try:
        with open(LOCAL_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-40:])
    except Exception as e:
        return f"Error reading log file: {e}"

if __name__ == "__main__":
    print("--- Testing Crash Tracker Telemetry ---")
    tele = get_gpu_telemetry()
    print(json.dumps(tele, indent=2))
    print("✓ Crash Tracker initialized.")
