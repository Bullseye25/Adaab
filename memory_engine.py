"""
=============================================================================
Adaab Studio: Dynamic SQLite User Memory & Personality Engine (memory_engine.py)
Features:
 1. Local SQLite User Profiles & Conversational History storage
 2. Onboarding state: Asks for Name, Profession, Country, Education
 3. Disambiguates between multiple individuals with the same name
 4. Key-Points Extraction & Personality Summarization Algorithm
 5. Generates evocative B1-B2 level Urdu personality and impression summaries
 6. Handles "Who did you talk to?" queries with rich historical recall
=============================================================================
"""

import os
import sys
import json
import sqlite3
import uuid
import re
from datetime import datetime

# Standard SQLite database file in project root
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaab_history.db")
DB_PATH = DB_FILE

def get_current_db_file() -> str:
    global DB_PATH, DB_FILE
    return DB_PATH or DB_FILE

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    target_path = db_path or get_current_db_file()
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_memory_tables(db_path: str = None):
    """Creates the user_profiles and user_conversations tables if they do not exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_uuid TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        display_name TEXT,
        profession TEXT,
        country TEXT,
        city TEXT,
        education TEXT,
        interests TEXT,
        personality_summary TEXT,
        appearance_impression TEXT,
        info_sharing_consent TEXT DEFAULT 'unspecified',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        conversation_count INTEGER DEFAULT 1
    );
    """)

    # Migration for existing databases
    try:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN info_sharing_consent TEXT DEFAULT 'unspecified'")
    except sqlite3.OperationalError:
        pass

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        session_uuid TEXT NOT NULL,
        assistant_persona TEXT DEFAULT 'Tehzeeb',
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ended_at TIMESTAMP,
        messages_json TEXT,
        key_points TEXT,
        topics TEXT,
        detected_mood TEXT,
        FOREIGN KEY (profile_id) REFERENCES user_profiles(id)
    );
    """)

    # Conversational turns memory (What is being talked about)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_turns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT DEFAULT 'default_session',
        user_query TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        detected_topic TEXT DEFAULT '',
        key_entities TEXT DEFAULT '',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Active conversation context & state (for anaphora / follow-up resolution)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_context (
        session_id TEXT PRIMARY KEY,
        active_topic TEXT DEFAULT '',
        last_entities TEXT DEFAULT '',
        last_query TEXT DEFAULT '',
        last_response TEXT DEFAULT '',
        turn_count INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()

# Initialize tables immediately on import
init_memory_tables()

# ─────────────────────────────────────────────────────────────────────────────
# User Profile Database Operations
# ─────────────────────────────────────────────────────────────────────────────

def save_or_update_user_profile(
    name: str,
    profession: str = None,
    country: str = None,
    education: str = None,
    city: str = None,
    interests: str = None,
    display_name: str = None,
    personality_summary: str = None,
    appearance_impression: str = None,
    info_sharing_consent: str = None,
    profile_id: int = None,
    db_path: str = None
) -> dict:
    """
    Saves a new user profile or updates an existing one in SQLite.
    Includes info_sharing_consent ('consented', 'declined', 'preferred_name_only', 'unspecified').
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    clean_name = name.strip() if name else "محترم دوست"
    display = display_name.strip() if display_name else clean_name

    if profile_id:
        # Update existing profile
        cursor.execute("""
        UPDATE user_profiles
        SET profession = COALESCE(?, profession),
            country = COALESCE(?, country),
            education = COALESCE(?, education),
            city = COALESCE(?, city),
            interests = COALESCE(?, interests),
            personality_summary = COALESCE(?, personality_summary),
            appearance_impression = COALESCE(?, appearance_impression),
            info_sharing_consent = COALESCE(?, info_sharing_consent),
            display_name = COALESCE(?, display_name),
            name = COALESCE(?, name),
            last_seen_at = CURRENT_TIMESTAMP,
            conversation_count = conversation_count + 1
        WHERE id = ?
        """, (profession, country, education, city, interests, personality_summary, appearance_impression, info_sharing_consent, display, clean_name, profile_id))
        target_id = profile_id
    else:
        # Check if identical profile already exists by name, country, and profession
        cursor.execute("""
        SELECT id FROM user_profiles
        WHERE LOWER(name) = LOWER(?) AND LOWER(COALESCE(country, '')) = LOWER(?) AND LOWER(COALESCE(profession, '')) = LOWER(?)
        """, (clean_name, (country or "").strip(), (profession or "").strip()))
        row = cursor.fetchone()

        if row:
            target_id = row["id"]
            cursor.execute("""
            UPDATE user_profiles
            SET education = COALESCE(?, education),
                city = COALESCE(?, city),
                interests = COALESCE(?, interests),
                personality_summary = COALESCE(?, personality_summary),
                appearance_impression = COALESCE(?, appearance_impression),
                info_sharing_consent = COALESCE(?, info_sharing_consent),
                display_name = COALESCE(?, display_name),
                last_seen_at = CURRENT_TIMESTAMP,
                conversation_count = conversation_count + 1
            WHERE id = ?
            """, (education, city, interests, personality_summary, appearance_impression, info_sharing_consent, display, target_id))
        else:
            # Create new profile
            new_uuid = str(uuid.uuid4())
            cursor.execute("""
            INSERT INTO user_profiles (
                profile_uuid, name, display_name, profession, country, city, education,
                interests, personality_summary, appearance_impression, info_sharing_consent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_uuid, clean_name, display, profession, country, city, education,
                interests, personality_summary, appearance_impression, info_sharing_consent or 'unspecified'
            ))
            target_id = cursor.lastrowid

    conn.commit()

    # Fetch and return updated profile
    cursor.execute("SELECT * FROM user_profiles WHERE id = ?", (target_id,))
    updated_row = dict(cursor.fetchone())
    conn.close()
    return updated_row

def evaluate_user_sharing_consent(consent: str) -> dict:
    """
    Python switch-case (match/case) evaluating user's information sharing status.
    Categorizes consent into:
      1. 'consented': User agreed and shared full info (Name, Profession, Country, etc.)
      2. 'declined': User refused / opted out of personal details for privacy
      3. 'preferred_name_only': User provided only what they should be called without personal info
      4. 'unspecified': Default / pending
    """
    status = (consent or "unspecified").strip().lower()
    match status:
        case "consented" | "shared" | "full":
            return {
                "key": "consented",
                "label": "مکمل کوائف منظور (Consented)",
                "allows_details": True,
                "badge": "✅ رازداری کی بجائے مکمل کوائف کا اشتراک پسند فرمایا",
                "description": "صارف نے خوش دلی سے اپنے نام اور تمام کوائف کا اشتراک فرمایا ہے۔"
            }
        case "declined" | "refused" | "private" | "no":
            return {
                "key": "declined",
                "label": "معلومات مخفی (Declined / Private)",
                "allows_details": False,
                "badge": "🔒 ذاتی معلومات کو پردۂ راز میں رکھنے کی ترجیح",
                "description": "صارف نے ذاتی کوائف مخفی رکھے ہیں اور پرائیویسی کو ترجیح دی ہے۔"
            }
        case "preferred_name_only" | "partial" | "name_only":
            return {
                "key": "preferred_name_only",
                "label": "صرف نامِ تخاطب (Preferred Name Only)",
                "allows_details": False,
                "badge": "🏷️ صرف باوقار گفتگو کے لیے پسندیدہ نام کا اشتراک",
                "description": "صارف نے صرف بہتر تخاطب کے لیے نام بتایا، ذاتی کوائف محفوظ ہیں۔"
            }
        case _:
            return {
                "key": "unspecified",
                "label": "غیر متعین (Unspecified)",
                "allows_details": False,
                "badge": "ℹ️ تاحال کوئی ترجیح متعین نہیں ہوئی",
                "description": "صارف کی رازداری یا اشتراک کی ترجیح ابھی غیر معین ہے۔"
            }

def get_profiles_with_sharing_switch(db_path: str = None) -> list[dict]:
    """
    Executes a SQL query with a switch-case (CASE statement) in SQLite
    to evaluate whether each user consented to share their information or opted out.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        id, profile_uuid, name, display_name, profession, country, education,
        personality_summary, appearance_impression, conversation_count,
        info_sharing_consent,
        CASE info_sharing_consent
            WHEN 'consented' THEN 'مکمل کوائف کا اشتراک منظور (Full Info Shared)'
            WHEN 'declined' THEN 'کوائف کا اشتراک مسترد / پرائیویسی ترجیح (Privacy Opt-Out / Declined)'
            WHEN 'preferred_name_only' THEN 'صرف تخاطب کا نام فراہم کیا (Preferred Call Name Only)'
            ELSE 'غیر واضح / ابتدائی مرحلہ (Unspecified)'
        END AS consent_status_label,
        CASE info_sharing_consent
            WHEN 'consented' THEN 1
            ELSE 0
        END AS can_share_details
    FROM user_profiles
    ORDER BY last_seen_at DESC;
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def find_profiles_by_name(name: str, db_path: str = None) -> list[dict]:
    """Finds all profiles matching a given name (case-insensitive)."""
    if not name:
        return []
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM user_profiles
    WHERE LOWER(name) LIKE LOWER(?) OR LOWER(display_name) LIKE LOWER(?)
    ORDER BY last_seen_at DESC
    """, (f"%{name.strip()}%", f"%{name.strip()}%"))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_profile_by_id(profile_id: int, db_path: str = None) -> dict:
    """Retrieves a single profile by primary key ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profiles WHERE id = ?", (profile_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_interacted_profiles(db_path: str = None) -> list[dict]:
    """Retrieves all stored user profiles ordered by most recent interaction."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profiles ORDER BY last_seen_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def record_conversation_session(
    profile_id: int,
    messages: list,
    assistant_persona: str = "Tehzeeb",
    key_points: str = None,
    topics: str = None,
    detected_mood: str = None,
    db_path: str = None
) -> int:
    """Records a complete conversation transcript session to SQLite."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    sess_uuid = str(uuid.uuid4())
    cursor.execute("""
    INSERT INTO user_conversations (
        profile_id, session_uuid, assistant_persona, messages_json, key_points, topics, detected_mood, ended_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        profile_id, sess_uuid, assistant_persona,
        json.dumps(messages, ensure_ascii=False),
        key_points, topics, detected_mood
    ))
    conn.commit()
    sess_id = cursor.lastrowid
    conn.close()
    return sess_id

# ─────────────────────────────────────────────────────────────────────────────
# Conversational Onboarding Parser & State Detector
# ─────────────────────────────────────────────────────────────────────────────

def is_full_name(name: str) -> bool:
    """Checks whether the provided name looks like a full name (at least 2 words)."""
    if not name:
        return False
    parts = name.strip().split()
    return len(parts) >= 2

def extract_profile_details_from_text(text: str, has_existing_name: bool = False) -> dict:
    """
    Extracts name, profession, country, and education from user messages.
    Supports English, Roman Urdu, and Native Urdu phrasing.
    When has_existing_name is True, direct input is never treated as a name,
    ensuring answers to onboarding questions (profession, city) are mapped accurately.
    """
    if not text:
        return {}
    
    extracted = {}
    lower = text.lower().strip()

    # 1. Name detection (only if name is not yet established)
    if not has_existing_name:
        name_patterns = [
            r"(?:my full name is|full name is|my name is|call me|this is)\s+([A-Za-z]+(?:\s+(?!(?:and|from|i am|i work|in|at)\b)[A-Za-z]+){0,3})",
            r"(?:mera poora naam|mera pura naam|mera mukammal naam|mera naam|mera nam)\s+([A-Za-z]+(?:\s+(?!(?:hai|hain|aur|se|me|mein)\b)[A-Za-z]+){0,3})",
            r"\bmujhe\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:kehte|bolte)\s+hain\b",
            r"(?:میرا پورا نام|میرا مکمل نام|میرا نام|میرا اسمِ گرامی|میرا اسم گرامی)\s+([\u0600-\u06FF]+(?:\s+(?!(?:ہے|کہتے|اور|سے|میں)\b)[\u0600-\u06FF]+){0,3})",
            r"مجھے\s+([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?)\s*(?:کہتے|پکارتے)\s*ہیں"
        ]
        for p in name_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                extracted["name"] = m.group(1).strip()
                break

        # Direct name input (e.g. user enters only their name like 'Ammad Raza' or 'علی' when asked)
        if not extracted.get("name"):
            words = text.strip().split()
            if 1 <= len(words) <= 3:
                clean = text.strip(" .،,!?:-")
                stopwords_en = r"\b(hi|hello|hey|salam|adaab|adab|bye|help|yes|no|haan|nahi|theek|good|kese|kaise|waqt|time|date|tarikh|tareekh|who|koun|kaun|urdu|poetry|shayari|tell|what|where|when|why|how|bato|batao|btao|na|bolo|bol|karo|bata|sunao|suno|kaho|aur|phir|next|more)\b"
                stopwords_ur = r"(آداب|سلام|کیسے|کب|کیوں|کہاں|وقت|تاریخ|کون|کیا|ہاں|نہیں|ٹھیک|شاعری|شعر|مدد|بتائیے|بتائیں|بتاؤ|سنائیے|سنائیں|سناؤ|سنو|بولو|کہو|فرمائیے|فرمائیں|کیجیے|کریں|معلومات|احوال|حال|کھیل|موسم|ہے|ہیں|تھا|تھی|تھے|کا|کی|کے|کو|سے|پر|تک|میں|نے|اور|پھر|نا)"
                if not re.search(stopwords_en, clean, re.IGNORECASE) and not re.search(stopwords_ur, clean):
                    if re.match(r"^[A-Za-z\s]+$", clean) or re.match(r"^[\u0600-\u06FF\s]+$", clean):
                        extracted["name"] = clean

    # 2. Profession detection
    prof_patterns = [
        r"(?:i am a|i work as a|my profession is|my job is)\s+([A-Za-z]+(?:\s+(?!(?:in|at|and|from)\b)[A-Za-z]+){0,3})",
        r"(?:mera pesha|mera shoba|main ek|mai aik|main|mai)\s+([A-Za-z]+(?:\s+(?!(?:hoon|hon|aur|se)\b)[A-Za-z]+){0,3})\s*(?:hoon|hon)",
        r"(?:میرا پیشہ|میرا شعبہ|میں ایک|میں)\s+([\u0600-\u06FF]+(?:\s+(?!(?:ہوں|کا کام|اور|سے)\b)[\u0600-\u06FF]+){0,3})\s*(?:ہوں|کا کام کرتا ہوں)"
    ]
    for p in prof_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            extracted["profession"] = m.group(1).strip()
            break

    # 3. Country / City detection
    loc_patterns = [
        r"(?:i live in|i am from|from)\s+([A-Za-z]+(?:\s+(?!(?:and|in|at)\b)[A-Za-z]+){0,2})",
        r"(?:main|mai)\s+([A-Za-z]+(?:\s+(?!(?:se|me|mein|aur)\b)[A-Za-z]+){0,2})\s*(?:se hoon|me rehta hoon|mein rehta hoon|se)",
        r"(?:میرا تعلق|میں)\s+([\u0600-\u06FF]+(?:\s+(?!(?:سے|میں|رہتا|مقیم|ہوں|اور)\b)[\u0600-\u06FF]+){0,2})\s*(?:سے ہے|سے ہوں|سے|میں رہتا ہوں|میں مقیم ہوں)"
    ]
    for p in loc_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            extracted["country"] = m.group(1).strip()
            break

    # 4. Education detection
    edu_patterns = [
        r"(?:i studied|my education is|degree in)\s+([A-Za-z]+(?:\s+(?!(?:and|in|at)\b)[A-Za-z]+){0,3})",
        r"(?:meri taleem|maine parha)\s+([A-Za-z]+(?:\s+(?!(?:hai|mein|aur)\b)[A-Za-z]+){0,3})\s*(?:hai|kiya|parha)",
        r"(?:میری تعلیم|میں نے)\s+([\u0600-\u06FF]+(?:\s+(?!(?:کی ڈگری|پڑھا|حاصل|اور)\b)[\u0600-\u06FF]+){0,3})\s*(?:کی ڈگری لی ہے|پڑھا ہے|کی تعلیم حاصل کی|کی تعلیم حاصل کی ہے)"
    ]
    for p in edu_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            extracted["education"] = m.group(1).strip()
            break

    # 5. Direct Details Mapping when name is already established
    if has_existing_name:
        clean_text = text.strip(" .،,!?:-")
        # Split by comma or 'from' or 'se' or 'aur'
        parts = re.split(r",|،|\bfrom\b|\bse\b|\baur\b|اور|سے", clean_text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            p1 = parts[0].strip()
            p2 = parts[1].strip()
            if not extracted.get("profession") and p1 and len(p1.split()) <= 4:
                extracted["profession"] = p1
            if not extracted.get("country") and p2 and len(p2.split()) <= 4:
                extracted["country"] = p2
        elif len(parts) == 1 and clean_text:
            common_locations = {
                "pakistan", "lahore", "karachi", "islamabad", "rawalpindi", "peshawar", "quetta",
                "multan", "faisalabad", "sialkot", "london", "uk", "usa", "canada", "uae", "dubai",
                "saudi arabia", "riyadh", "لاہور", "کراچی", "اسلام آباد", "پاکستان"
            }
            if clean_text.lower() in common_locations or any(loc in clean_text.lower() for loc in common_locations):
                if not extracted.get("country"):
                    extracted["country"] = clean_text
            else:
                if not extracted.get("profession"):
                    extracted["profession"] = clean_text

    return extracted

def is_declining_info_sharing(text: str) -> bool:
    """
    Detects if the user declines or opts out of providing their personal information
    (name, profession, country, education) due to privacy preferences.
    Handles English, Roman Urdu, and Nastaliq Urdu expressions.
    """
    if not text:
        return False
    lower = text.lower().strip()
    if lower in ["no", "nah", "nope", "skip", "pass", "private", "personal", "never", "نہیں", "نو", "چھوڑیں", "رہنے دیں"]:
        return True

    en_patterns = [
        r"\b(?:don'?t|do not|won'?t|will not|cannot|cant)\s+(?:want\s+to\s+)?(?:share|give|tell|disclose|provide)\b",
        r"\b(?:prefer\s+not\s+to\s+(?:say|share|tell)|rather\s+not\s+(?:say|share|tell))\b",
        r"\b(?:no\s+(?:personal\s+)?(?:info|information|details)|keep\s+(?:it\s+)?(?:private|confidential|secret))\b",
        r"\b(?:want\s+to\s+remain\s+anonymous|stay\s+anonymous|privacy\s+(?:matters|please)|not\s+comfortable)\b",
        r"\b(?:don'?t\s+(?:ask|record|save)|no\s+name|without\s+(?:my\s+)?name)\b"
    ]
    if any(re.search(p, lower) for p in en_patterns):
        return True

    roman_patterns = [
        r"\b(?:nahi|nahin|na)\s+(?:batana|bataunga|bataungi|batao|share|dena|dungi|dunga)\b",
        r"\b(?:share\s+(?:nahi|nahin|na)|nahi\s+share|nahin\s+share)\b",
        r"\b(?:info|information|details|kuch|naam|name)\s+(?:.*?\s+)?(?:nahi|nahin|na)\s+(?:deni|dena|batana|share|karna|karni)\b",
        r"\b(?:private\s+hai|personal\s+hai|raaz\s+rakhna|secret\s+hai)\b",
        r"\b(?:rehne\s+dein|rehne\s+do|choro|chhoro|zaroorat\s+nahi)\b"
    ]
    if any(re.search(p, lower) for p in roman_patterns):
        return True

    ur_patterns = [
        r"(نہیں بتانا|نہیں بتاؤں گا|نہیں بتاؤں گی|معلومات نہیں دینی|معلومات نہیں دوں گا|معلومات نہیں دوں گی)",
        r"(شیئر نہیں کرنا|کوائف نہیں دینے|کوئی معلومات نہیں|کچھ نہیں بتانا|نجی معاملہ ہے|پرائیویسی ہے)",
        r"(راز رکھنا ہے|نہیں چاہتا|نہیں چاہتی|نہیں کہنا|چھوڑیں|رہنے دیں|نام نہیں بتانا|نام نہیں دینا)",
        r"(نام نہیں بتانا چاہتا|نام نہیں بتانا چاہتی|کوئی تفصیل نہیں دینی|ضرورت نہیں)"
    ]
    if any(re.search(p, text) for p in ur_patterns):
        return True

    return False

def extract_preferred_call_name(text: str) -> str | None:
    """
    Extracts what the user wants to be called (nickname, pseudonym, or alias)
    when they decline full personal details or when prompted for a preferred call name.
    """
    if not text:
        return None
    clean = text.strip(" .،,!?:-\"'")
    if is_declining_info_sharing(clean) or clean.lower() in ["nothing", "no name", "none", "کچھ نہیں", "کوئی نہیں", "کوئی نام نہیں"]:
        return None

    patterns_en = [
        r"(?:you can call me|just call me|please call me|call me|address me as|my friends call me|my nickname is)\s+([A-Za-z0-9\u0600-\u06FF]+(?:\s+[A-Za-z0-9\u0600-\u06FF]+){0,2})",
        r"(?:call me)\s+([A-Za-z0-9\u0600-\u06FF]+)"
    ]
    for p in patterns_en:
        m = re.search(p, clean, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    patterns_roman = [
        r"(?:aap mujhe|ap mujhe|mujhe|bas mujhe|bus mujhe)\s+([A-Za-z0-9\u0600-\u06FF]+(?:\s+[A-Za-z0-9\u0600-\u06FF]+){0,2})\s*(?:keh lein|keh lain|keh sakti hain|keh sakte hain|keh kar pukarein|bulayein|kahain)",
        r"(?:sirf)\s+([A-Za-z0-9\u0600-\u06FF]+)\s*(?:keh lein|keh dain)"
    ]
    for p in patterns_roman:
        m = re.search(p, clean, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    patterns_ur = [
        r"(?:آپ مجھے|مجھے|بس مجھے|صرف)\s+([\u0600-\u06FF0-9A-Za-z]+(?:\s+[\u0600-\u06FF0-9A-Za-z]+){0,2})\s*(?:کہہ لیں|کہہ کر پکاریں|کہہ کر بلائیں|کہہ سکتی ہیں|کہہ سکتے ہیں|پکاریں|کہیے|کہیں)",
        r"(?:بس|صرف)\s+([\u0600-\u06FF0-9A-Za-z]+)\s*(?:کہیے|کہہ دیں|کہہ لیں)"
    ]
    for p in patterns_ur:
        m = re.search(p, clean)
        if m:
            return m.group(1).strip()

    words = clean.split()
    if 1 <= len(words) <= 3:
        stopwords = r"\b(hi|hello|hey|salam|adaab|adab|bye|help|yes|no|naam|name|call|me|kuch|nahi|nahin|karo|batao)\b"
        if not re.search(stopwords, clean, re.IGNORECASE):
            return clean

    return None

def is_save_details_command(text: str) -> bool:
    """Checks if the user explicitly commanded the AI to save their details."""
    if not text:
        return False
    lower = text.lower().strip()
    patterns = [
        r"\b(save my details|save my info|remember me|record my details|save this)\b",
        r"(میری تفصیلات محفوظ|میرے کوائف محفوظ|میری معلومات محفوظ|مجھے یاد رکھنا|محفوظ کر لو|سیو کر لو)"
    ]
    return any(re.search(p, lower) for p in patterns)

def is_who_did_you_talk_to_query(text: str) -> bool:
    """Detects if user is asking who the AI has talked to or about past people."""
    if not text:
        return False
    lower = text.lower().strip()
    patterns = [
        r"\b(who did you talk to|who have you talked to|who else did you talk to|who did you speak with|people you met)\b",
        r"(کن کن لوگوں سے بات|کس کس سے بات|کن لوگوں سے گفتگو|کس سے بات کی|جن سے بات ہوئی|کوئی اور بھی تھا|کس سے ملاقات ہوئی|پہلے کس سے بات کی)"
    ]
    return any(re.search(p, lower) for p in patterns)

# ─────────────────────────────────────────────────────────────────────────────
# Key-Points Extraction & Personality Summarization Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def extract_key_points_and_summarize_personality(messages: list, current_profile: dict = None) -> dict:
    """
    Intelligent conversation analysis algorithm:
    1. Extracts topics discussed (e.g. poetry, technology, philosophical life questions)
    2. Analyzes communication tone & demeanor
    3. Synthesizes a warm, B1-B2 level Urdu personality summary
    4. Generates an evocative impression of what that person seems like.
    """
    user_texts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_texts.append(msg.get("content", ""))
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0]:
            user_texts.append(msg[0])

    combined_text = " ".join(user_texts).lower()

    # Topic indicators
    topics = []
    if any(w in combined_text for w in ["ghalib", "غالب", "desire", "فلسفہ"]):
        topics.append("مرزا غالب کی فلسفیانہ شاعری")
    if any(w in combined_text for w in ["iqbal", "اقبال", "khudi", "عزم", "شاہین"]):
        topics.append("علامہ اقبال کا فلسفۂ خودی")
    if any(w in combined_text for w in ["faiz", "فیض", "barish", "rain", "بہار", "بارش", "گل"]):
        topics.append("فیض احمد فیض، بارش اور رومانوی مناظر")
    if any(w in combined_text for w in ["code", "software", "ai", "کمپیوٹر", "ٹیکنالوجی", "پروگرامنگ"]):
        topics.append("ٹیکنالوجی، مصنوعی ذہانت اور جدید علوم")
    if any(w in combined_text for w in ["time", "date", "وقت", "تاریخ"]):
        topics.append("حاضر وقت اور نظام الاوقات")

    if not topics:
        topics.append("ادبی گفتگو اور شائستہ احوال پرسی")

    # Tone analysis
    tone_traits = []
    if any(w in combined_text for w in ["please", "شکریہ", "مہربانی", "جناب", "adaab", "آداب", "kindly"]):
        tone_traits.append("شائستہ اور بااخلاق")
    if any(w in combined_text for w in ["why", "how", "کیوں", "کیسے", "سمجھائیں", "explain"]):
        tone_traits.append("متجسس اور فکری گہرائی رکھنے والے")
    if any(w in combined_text for w in ["love", "ishq", "دل", "محبت", "احساس", "sad", "سوز"]):
        tone_traits.append("حساس اور لطیف جذبات کے حامل")
    if not tone_traits:
        tone_traits.append("سلجھے ہوئے اور پرخلوص")

    user_name = (current_profile.get("display_name") or current_profile.get("name")) if current_profile else "محترم شخصیت"
    consent = current_profile.get("info_sharing_consent") if current_profile else "unspecified"
    traits_str = "، ".join(tone_traits)
    topics_str = " اور ".join(topics)

    if consent in ("declined", "preferred_name_only"):
        personality_summary = (
            f"جناب {user_name} ایک {traits_str} اور خوددار انسان معلوم ہوتے ہیں۔ "
            f"انہوں نے اپنی ذاتی تفصیلات کو پردۂ راز میں رکھنے کی ترجیح دی ہے۔ "
            f"گفتگو کے دوران ان کی خصوصی دلچسپی {topics_str} میں نمایاں رہی۔ "
            f"ان کا اندازِ گفتگو نہایت باوقار اور متانت سے بھرپور ہے جس سے ان کے سلجھے ہوئے ذوق کا پتا چلتا ہے۔"
        )
    else:
        profession = current_profile.get("profession") if current_profile else "علم و ہنر کے دلدادہ"
        country = current_profile.get("country") if current_profile else ""
        education = current_profile.get("education") if current_profile else ""
        country_part = f"جن کا تعلق {country} سے ہے" if country else ""
        prof_part = f"اور وہ پیشے کے اعتبار سے {profession} ہیں" if profession else ""

        personality_summary = (
            f"جناب {user_name} ایک {traits_str} انسان معلوم ہوتے ہیں{('، ' + country_part) if country_part else ''}"
            f"{(' ' + prof_part) if prof_part else ''}۔ "
            f"گفتگو کے دوران ان کی خصوصی دلچسپی {topics_str} میں نمایاں رہی۔ "
            f"ان کا اندازِ گفتگو نہایت باوقار اور متانت سے بھرپور ہے جس سے ان کے سلجھے ہوئے ذوق کا پتا چلتا ہے۔"
        )

    # Generate Impression of what this person would seem like
    appearance_impression = (
        f"ایک باوقار اور پرسکون انداز رکھنے والی شخصیت، جن کی گفتگو میں شائستگی اور دھیما پن ہے۔ "
        f"ان کا مزاج دوستانہ اور فکری ہے، اور وہ سنجیدہ گفتگو کو خوش دلی اور توجہ سے سنتے اور سراہتے ہیں۔"
    )

    key_points = " • " + "\n • ".join(topics)

    return {
        "topics": ", ".join(topics),
        "tone_traits": ", ".join(tone_traits),
        "personality_summary": personality_summary,
        "appearance_impression": appearance_impression,
        "key_points": key_points
    }

# ─────────────────────────────────────────────────────────────────────────────
# "Who did you talk to?" Response Generator in B1-B2 Urdu
# ─────────────────────────────────────────────────────────────────────────────

def format_who_did_you_talk_to_response(persona: str = "Tehzeeb", db_path: str = None) -> str:
    """
    Produces a natural, warm B1-B2 level Urdu response listing the people
    the AI has talked to, along with their professions, countries, and personality summaries,
    evaluating their privacy consent choices using both database SQL CASE and Python match/case.
    """
    profiles = get_profiles_with_sharing_switch(db_path)
    is_male = (persona and ("tabraiz" in persona.lower() or "male" in persona.lower()))
    verb_gender = "ہوا ہے" if is_male else "ہوئی ہے"
    self_name = "تبریز" if is_male else "تہذیب"
    verb_save = "کر سکتا ہوں" if is_male else "کر سکتی ہوں"

    if not profiles:
        return (
            f"آداب عرض ہے! اس وقت یادداشت بالکل صاف (خالی) ہے اور کوئی سابقہ ریکارڈ موجود نہیں ہے۔ "
            f"جیسے ہی ہم گفتگو فرمائیں گے، آپ کے کوائف اور شخصیت کا خلاصہ یہاں باادب محفوظ ہو جائے گا۔"
        )

    response_lines = [
        f"آداب عرض ہے! مجھے اب تک جن معزز احباب سے گفتگو کا شرف حاصل {verb_gender}، ان کے احوال اور شخصیت کا خلاصہ یہ ہے:\n"
    ]

    for idx, p in enumerate(profiles, 1):
        name = p.get("display_name") or p.get("name") or "معزز مہمان"
        consent_info = evaluate_user_sharing_consent(p.get("info_sharing_consent"))
        count = p.get("conversation_count", 1)
        summary = p.get("personality_summary") or "نہایت شائستہ اور باادب گفتگو رہی۔"
        impression = p.get("appearance_impression") or "ایک باوقار اور متین شخصیت۔"

        match consent_info["key"]:
            case "consented":
                prof = p.get("profession") or "پیشے کی معلومات درج نہیں"
                country = p.get("country") or "ملک کی معلومات درج نہیں"
                edu = p.get("education")
                edu_str = f" | تعلیم: {edu}" if edu else ""
                card = (
                    f"{idx}. 🌟 **{name}** ({country} — {prof}{edu_str})\n"
                    f"   • **رضامندی:** {consent_info['badge']}\n"
                    f"   • **شخصیت کا خلاصہ:** {summary}\n"
                    f"   • **شخصی جھلک و انداز:** {impression}\n"
                    f"   • **ملاقاتیں:** {count} بار گفتگو ہو چکی ہے۔\n"
                )
            case "preferred_name_only":
                card = (
                    f"{idx}. 🏷️ **{name}** (صرف پسندیدہ نامِ تخاطب)\n"
                    f"   • **رضامندی:** {consent_info['badge']}\n"
                    f"   • **کوائف کا تحفظ:** صارف کی ترجیح کے مطابق ذاتی تفصیلات (پیشہ و ملک) مخفی رکھی گئی ہیں۔\n"
                    f"   • **شخصیت کا خلاصہ:** {summary}\n"
                    f"   • **ملاقاتیں:** {count} بار گفتگو ہو چکی ہے۔\n"
                )
            case "declined":
                card = (
                    f"{idx}. 🔒 **{name}** (پرائیویسی ترجیح / معلومات مخفی)\n"
                    f"   • **رضامندی:** {consent_info['badge']}\n"
                    f"   • **کوائف کا تحفظ:** صارف نے کوائف کا اشتراک نہ کرنے کا انتخاب کیا، جس کا پورا احترام کیا گیا۔\n"
                    f"   • **ملاقاتیں:** {count} بار گفتگو ہو چکی ہے۔\n"
                )
            case _:
                prof = p.get("profession") or ""
                country = p.get("country") or ""
                extra = f" ({country} — {prof})" if (country or prof) else ""
                card = (
                    f"{idx}. 👤 **{name}**{extra}\n"
                    f"   • **شخصیت کا خلاصہ:** {summary}\n"
                    f"   • **ملاقاتیں:** {count} بار گفتگو ہو چکی ہے۔\n"
                )
        response_lines.append(card)

    response_lines.append(f"یہ تمام احباب ادب دوست اور باوقار ہیں۔ فرمائیے، کیا آپ ان میں سے کسی کے بارے میں مزید کچھ جاننا چاہتے ہیں؟")
    return "\n".join(response_lines)

def record_dialogue_turn(
    user_query: str,
    bot_response: str,
    detected_topic: str = "",
    key_entities: list = None,
    session_id: str = "default_session",
    db_path: str = None
) -> int:
    """
    Records a completed dialogue turn (what is being talked about) into SQLite.
    Updates the active conversation context for anaphora / follow-up resolution.
    """
    entities_str = json.dumps(key_entities or [], ensure_ascii=False) if isinstance(key_entities, list) else str(key_entities or "")
    clean_topic = (detected_topic or "").strip()
    clean_query = (user_query or "").strip()
    clean_resp = (bot_response or "").strip()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO conversation_turns (session_id, user_query, bot_response, detected_topic, key_entities)
        VALUES (?, ?, ?, ?, ?)
        """, (session_id, clean_query, clean_resp, clean_topic, entities_str))
        turn_id = cursor.lastrowid

        # Upsert into conversation_context
        cursor.execute("""
        INSERT INTO conversation_context (session_id, active_topic, last_entities, last_query, last_response, turn_count, updated_at)
        VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET
            active_topic = CASE WHEN ? != '' THEN ? ELSE conversation_context.active_topic END,
            last_entities = ?,
            last_query = ?,
            last_response = ?,
            turn_count = conversation_context.turn_count + 1,
            updated_at = CURRENT_TIMESTAMP
        """, (session_id, clean_topic, entities_str, clean_query, clean_resp,
              clean_topic, clean_topic, entities_str, clean_query, clean_resp))

        conn.commit()
        return turn_id
    finally:
        conn.close()

def get_recent_conversation_context(
    limit: int = 5,
    session_id: str = "default_session",
    db_path: str = None
) -> list:
    """
    Retrieves the most recent dialogue turns for the specified session from SQLite.
    Returns list of dicts: [{'user_query': ..., 'bot_response': ..., 'detected_topic': ..., 'timestamp': ...}]
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT id, session_id, user_query, bot_response, detected_topic, key_entities, timestamp
        FROM conversation_turns
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        results = [dict(r) for r in reversed(rows)]
        return results
    finally:
        conn.close()

def get_active_discussion_topic(
    session_id: str = "default_session",
    db_path: str = None
) -> dict:
    """
    Retrieves the current active topic and context from SQLite.
    Returns dict: {'active_topic': ..., 'last_query': ..., 'last_response': ..., 'last_entities': ..., 'turn_count': ...}
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT active_topic, last_entities, last_query, last_response, turn_count, updated_at
        FROM conversation_context
        WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {
            "active_topic": "",
            "last_entities": "[]",
            "last_query": "",
            "last_response": "",
            "turn_count": 0,
            "updated_at": None
        }
    finally:
        conn.close()

def clear_conversation_memory(session_id: str = None, db_path: str = None) -> bool:
    """
    Clears topic and turn memory from SQLite. If session_id is None, clears all sessions.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        if session_id:
            cursor.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM conversation_context WHERE session_id = ?", (session_id,))
        else:
            cursor.execute("DELETE FROM conversation_turns;")
            cursor.execute("DELETE FROM conversation_context;")
        conn.commit()
        return True
    except Exception as e:
        print(f"[MemoryEngine] Error clearing conversation memory: {e}")
        return False
    finally:
        conn.close()

def clear_user_memory(db_path: str = None) -> bool:
    """
    Clears all recorded user profiles and conversation history from SQLite database.
    Ensures fresh clean onboarding for future conversations.
    """
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_conversations;")
        cursor.execute("DELETE FROM user_profiles;")
        cursor.execute("DELETE FROM conversation_turns;")
        cursor.execute("DELETE FROM conversation_context;")
        conn.commit()
        conn.close()
        print("[MemoryEngine] Successfully cleared all user profiles and conversation records.")
        return True
    except Exception as e:
        print(f"[MemoryEngine] Error clearing memory: {e}")
def log_telemetry_event(
    event_type: str,
    details: str = "",
    vram_allocated_mb: float = 0.0,
    vram_reserved_mb: float = 0.0,
    error_msg: str = "",
    db_path: str = None
):
    """Logs system telemetry or crash events into SQLite."""
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            details TEXT,
            vram_allocated_mb REAL DEFAULT 0.0,
            vram_reserved_mb REAL DEFAULT 0.0,
            error_msg TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cursor.execute("""
        INSERT INTO telemetry_logs (event_type, details, vram_allocated_mb, vram_reserved_mb, error_msg)
        VALUES (?, ?, ?, ?, ?)
        """, (event_type, details, vram_allocated_mb, vram_reserved_mb, error_msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[MemoryEngine] Telemetry log failed: {e}")

if __name__ == "__main__":
    print("Initializing memory engine...")
    init_memory_tables()
    # Demo save profile
    p1 = save_or_update_user_profile(
        name="عماد رضا",
        profession="سافٹ ویئر انجینئر",
        country="پاکستان",
        education="کمپیوٹر سائنس",
        personality_summary="ایک ذہین اور پرجوش انجینئر جنہیں اردو شاعری اور جدید ٹیکنالوجی کا امتزاج پسند ہے۔",
        appearance_impression="تیز نظر اور متین چہرہ، جو مسائل کو مسکراہٹ کے ساتھ حل کرنے کا حوصلہ رکھتے ہیں۔"
    )
    print("Saved Profile 1:", p1["name"], "-", p1["profession"])
    print("\n'Who did you talk to?' Output:")
    print(format_who_did_you_talk_to_response("Tehzeeb"))
