"""
=============================================================================
Adaab Studio: Temporal Awareness & Local Clock Module (time_context.py)
Provides accurate local PC time and date awareness in both English and
natural B1-B2 level cultured Urdu.
=============================================================================
"""

import os
import sys
import re
from datetime import datetime

# Day of week translations in clean B1-B2 Urdu
URDU_DAYS = {
    0: "پیر (Monday)",
    1: "منگل (Tuesday)",
    2: "بدھ (Wednesday)",
    3: "جمعرات (Thursday)",
    4: "جمعہ (Friday)",
    5: "ہفتہ (Saturday)",
    6: "اتوار (Sunday)"
}

URDU_DAYS_PLAIN = {
    0: "پیر",
    1: "منگل",
    2: "بدھ",
    3: "جمعرات",
    4: "جمعہ",
    5: "ہفتہ",
    6: "اتوار"
}

# Month names in standard Urdu
URDU_MONTHS = {
    1: "جنوری",
    2: "فروری",
    3: "مارچ",
    4: "اپریل",
    5: "مئی",
    6: "جون",
    7: "جولائی",
    8: "اگست",
    9: "ستمبر",
    10: "اکتوبر",
    11: "نومبر",
    12: "دسمبر"
}

def get_time_period_urdu(hour: int) -> tuple[str, str]:
    """
    Returns the time period description and greeting in B1-B2 level Urdu.
    (period_name, appropriate_greeting)
    """
    if 4 <= hour < 12:
        return "صبح", "صبح بخیر"
    elif 12 <= hour < 16:
        return "دوپہر", "دوپہر بخیر"
    elif 16 <= hour < 20:
        return "شام", "شام بخیر"
    else:
        return "رات", "شب بخیر"

def get_local_time_context() -> dict:
    """
    Reads the exact local PC system clock and generates bilingual context.
    """
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    hour_12 = hour % 12 or 12
    am_pm = "AM" if hour < 12 else "PM"

    period_urdu, greeting_urdu = get_time_period_urdu(hour)
    day_name = URDU_DAYS_PLAIN.get(now.weekday(), "")
    month_name = URDU_MONTHS.get(now.month, "")

    # Human-readable English
    english_str = now.strftime("%A, %B %d, %Y - %I:%M %p")

    # Natural B1-B2 Urdu datetime string
    urdu_time_str = f"{period_urdu} کے {hour_12} بج کر {minute:02d} منٹ"
    urdu_date_str = f"{day_name}، {now.day} {month_name} {now.year}"
    urdu_full_str = f"{urdu_date_str} — {urdu_time_str}"

    return {
        "datetime_obj": now,
        "iso": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": hour,
        "minute": minute,
        "weekday": now.weekday(),
        "day_name_urdu": day_name,
        "month_name_urdu": month_name,
        "period_urdu": period_urdu,
        "greeting_urdu": greeting_urdu,
        "english_str": english_str,
        "urdu_time_str": urdu_time_str,
        "urdu_date_str": urdu_date_str,
        "urdu_full_str": urdu_full_str
    }

def is_time_or_date_query(text: str) -> bool:
    """Detects if user is asking about current time, date, or day."""
    if not text:
        return False
    lower = text.lower().strip()
    patterns = [
        r"\b(time|date|day|clock|today)\b",
        r"(کیا وقت|کتنے بجے|وقت کیا|کیا تاریخ|کونسا دن|آج کیا دن|آج کیا تاریخ|وقت بتا|ٹائم کیا)",
        r"\b(what time|current time|what is the date|what day is it)\b"
    ]
    return any(re.search(p, lower) for p in patterns)

def get_time_date_response(query: str = "", persona: str = "Tehzeeb") -> str:
    """
    Generates a natural, warm B1-B2 level Urdu response answering the time/date query.
    """
    ctx = get_local_time_context()
    lower = query.lower() if query else ""

    asking_only_time = any(w in lower for w in ["time", "کتنے بجے", "وقت کیا", "ٹائم"]) and not any(w in lower for w in ["date", "تاریخ", "day", "دن"])
    asking_only_date = any(w in lower for w in ["date", "تاریخ", "day", "دن"]) and not any(w in lower for w in ["time", "وقت", "بجے", "ٹائم"])

    verb_gender = "کر سکتی ہوں" if (not persona or "tehzeeb" in persona.lower() or "female" in persona.lower()) else "کر سکتا ہوں"

    if asking_only_time:
        return (
            f"جی جناب! اس وقت آپ کے کمپیوٹر کی گھڑی کے مطابق {ctx['urdu_time_str']} ہوئے ہیں۔ "
            f"فرمائیے، میں آپ کی اور کیا مدد {verb_gender}؟"
        )
    elif asking_only_date:
        return (
            f"جی! آج {ctx['urdu_date_str']} ہے۔ "
            f"فرمائیے، میں آپ کی کیا مدد {verb_gender}؟"
        )
    else:
        return (
            f"جی محترم! اس وقت کی تاریخ اور وقت یہ ہے:\n"
            f"📅 تاریخ: {ctx['urdu_date_str']}\n"
            f"⏰ وقت: {ctx['urdu_time_str']} ({ctx['english_str']})\n"
            f"{ctx['greeting_urdu']}! فرمائیے، آج کا کیا ارادہ ہے؟"
        )

if __name__ == "__main__":
    c = get_local_time_context()
    print("English:", c["english_str"])
    print("Urdu:", c["urdu_full_str"])
    print("Greeting:", c["greeting_urdu"])
    print("\nSample Response (Time):", get_time_date_response("ابھی کیا وقت ہے؟"))
    print("\nSample Response (Date):", get_time_date_response("آج کیا تاریخ ہے؟"))
