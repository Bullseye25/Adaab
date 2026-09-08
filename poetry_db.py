import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sqlite3
import hashlib
import re
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaab_history.db")

def normalize_urdu_text(text: str) -> str:
    """
    Normalizes Urdu text for robust duplicate detection:
    - Removes Arabic/Urdu diacritics (Zabar, Zer, Pesh, Tashdeed, Tanween, Sukun, Khari Zabar)
    - Normalizes different forms of Ye, Kaf, He, Hamza
    - Strips punctuation and collapses whitespace
    """
    if not text:
        return ""
    # Remove diacritics (aerab / harakat)
    diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
    clean_text = diacritics_pattern.sub('', text)

    # Normalize letter variations
    replacements = {
        'ك': 'ک',      # Arabic kaf to Urdu kaf
        'ي': 'ی',      # Arabic ye to Urdu ye
        'ى': 'ی',
        'ة': 'ہ',      # Teh marbuta to He
        'ہ': 'ہ',
        'ھ': 'ھ',      # Do-chashmi he kept distinct
        'ء': '',       # Strip stand-alone hamza for deduplication matching
        '،': '',       # Urdu comma
        '۔': '',       # Urdu full stop
        '؟': '',       # Urdu question mark
        '!': '',
        ',': '',
        '.': '',
        '-': '',
        ':': '',
        '\n': ' ',
        '\r': '',
        '\t': ' '
    }
    for old, new in replacements.items():
        clean_text = clean_text.replace(old, new)

    # Collapse multiple whitespaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def compute_verse_hash(misra_1: str, misra_2: str) -> str:
    """Computes a unique SHA-256 fingerprint for a couplet."""
    norm1 = normalize_urdu_text(misra_1)
    norm2 = normalize_urdu_text(misra_2)
    combined = f"{norm1}|{norm2}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()

def init_db(db_path: str = DB_FILE):
    """Initializes the SQLite database with required tables and indexes."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS generated_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_uuid TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        misra_1 TEXT NOT NULL,
        misra_2 TEXT NOT NULL,
        normalized_hash TEXT UNIQUE NOT NULL,
        theme TEXT,
        color_palette TEXT,
        art_style TEXT,
        nature_theme TEXT,
        lighting TEXT,
        poster_path TEXT,
        audio_path TEXT,
        video_path TEXT,
        english_translation TEXT,
        roman_urdu TEXT,
        status TEXT DEFAULT 'completed'
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS system_telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT NOT NULL,
        vram_allocated_mb REAL,
        vram_reserved_mb REAL,
        details TEXT,
        error_message TEXT
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON generated_posts(normalized_hash);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_theme ON generated_posts(theme);")
    conn.commit()
    conn.close()

def is_verse_duplicate(misra_1: str, misra_2: str, db_path: str = DB_FILE) -> bool:
    """Returns True if this exact or normalized verse has ever been used before."""
    init_db(db_path)
    v_hash = compute_verse_hash(misra_1, misra_2)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM generated_posts WHERE normalized_hash = ?", (v_hash,))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def record_post(post_data: dict, db_path: str = DB_FILE) -> int:
    """Records a generated post into the database. Returns the new row ID."""
    init_db(db_path)
    v_hash = compute_verse_hash(post_data["misra_1"], post_data["misra_2"])
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO generated_posts (
        post_uuid, misra_1, misra_2, normalized_hash,
        theme, color_palette, art_style, nature_theme, lighting,
        poster_path, audio_path, video_path,
        english_translation, roman_urdu, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        post_data.get("post_uuid", datetime.now().strftime("post_%Y%m%d_%H%M%S")),
        post_data["misra_1"],
        post_data["misra_2"],
        v_hash,
        post_data.get("theme", "General"),
        post_data.get("color_palette", ""),
        post_data.get("art_style", ""),
        post_data.get("nature_theme", ""),
        post_data.get("lighting", ""),
        post_data.get("poster_path", ""),
        post_data.get("audio_path", ""),
        post_data.get("video_path", ""),
        post_data.get("english_translation", ""),
        post_data.get("roman_urdu", ""),
        post_data.get("status", "completed")
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def log_telemetry_event(event_type: str, details: str = "", vram_allocated_mb: float = 0.0,
                        vram_reserved_mb: float = 0.0, error_msg: str = "", db_path: str = DB_FILE):
    """Logs system telemetry or error events into SQLite."""
    try:
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO system_telemetry (event_type, vram_allocated_mb, vram_reserved_mb, details, error_message)
        VALUES (?, ?, ?, ?, ?)
        """, (event_type, vram_allocated_mb, vram_reserved_mb, details, error_msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Telemetry Error] Failed to log telemetry: {e}")

def get_history_summary(db_path: str = DB_FILE) -> dict:
    """Returns key metrics from the database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM generated_posts")
    total_posts = cursor.fetchone()[0]

    cursor.execute("SELECT post_uuid, misra_1, misra_2, theme, created_at FROM generated_posts ORDER BY id DESC LIMIT 5")
    recent = cursor.fetchall()
    conn.close()

    return {
        "total_posts": total_posts,
        "recent_posts": [
            {
                "uuid": r[0],
                "misra_1": r[1],
                "misra_2": r[2],
                "theme": r[3],
                "created_at": r[4]
            } for r in recent
        ]
    }

def clear_database(db_path: str = DB_FILE) -> bool:
    """Wipes all records from generated_posts and system_telemetry in the SQLite database."""
    try:
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM generated_posts;")
        cursor.execute("DELETE FROM system_telemetry;")
        conn.commit()
        conn.isolation_level = None
        cursor.execute("VACUUM;")
        conn.close()
        return True
    except Exception as e:
        print(f"[Database Error] Failed to clear database: {e}")
        return False

if __name__ == "__main__":
    init_db()
    print("✓ Local SQLite database initialized successfully at:", DB_FILE)
