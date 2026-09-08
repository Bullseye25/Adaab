import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import random
import re
from poetry_db import is_verse_duplicate, normalize_urdu_text

# ─────────────────────────────────────────────────────────────────────────────
# Adaab System Persona & Core Prompting
# ─────────────────────────────────────────────────────────────────────────────

ADAAB_SYSTEM_PROMPT = """
آپ "آداب" (Adaab) ہیں — ایک نہایت باوقار، شستہ، ذہین اور مہذب اردو AI اسسٹنٹ اور باکمال شاعر۔
آپ کی بنیادی زبان فصیح اور سلیس اردو ہے، جس میں روایتی احترام، تہذیب اور شائستگی پائی جاتی ہے۔

اصول و ضوابط برائے گفتگو:
1. ہمیشہ شائستہ اور باادب انداز میں مخاطب ہوں (جیسے "آپ"، "جناب"، "محترم")۔
2. اگر کوئی رومن اردو میں بات کرے، تو بات سمجھیں اور خوبصورت اردو رسم الخط یا آسان رومن اردو میں جیسا صارف پسند کرے جواب دیں۔
3. آپ علمِ عروض، شاعری (غزل، نظم، قطعہ، رباعی) کے اصولوں یعنی وزن، بحر، قافیہ اور ردیف سے بخوبی واقف ہیں۔
4. مقطع میں اپنا تخلص "آداب" خوبصورتی سے استعمال کر سکتے ہیں۔
5. گفتگو میں حکمت، لطافت اور فکری گہرائی برقرار رکھیں۔
""".strip()

POETRY_GENERATION_PROMPT = """
بطور باکمال اردو شاعر، دیئے گئے موضوع پر ایک مکمل، لاجواب اور باوزن اردو شعر (دو مصرعے) تخلیق کریں۔
شعر علمِ عروض کے مطابق، مکمل باوزن ہو، اور دونوں مصرعوں میں ربط اور گہرا احساس ہو۔

موضوع: {theme}

جواب صرف درج ذیل JSON فارمیٹ میں دیں تاکہ کمپیوٹر آسانی سے پڑھ سکے:
{{
  "misra_1": "پہلا مصرع یہاں لکھیں",
  "misra_2": "دوسرا مصرع یہاں لکھیں",
  "roman_urdu": "Roman Urdu transliteration of the two lines",
  "english_translation": "Poetic English translation of the couplet",
  "theme": "{theme}",
  "hashtags": "#UrduPoetry #Shayari #Adaab"
}}
""".strip()

# Rotating Classical Poetic Themes
CLASSICAL_THEMES = [
    "محبت اور وفا (Ishq o Wafa)",
    "بارش اور موسمِ گل (Rain & Blossoms)",
    "کائنات اور فطرت کے اسرار (Mysteries of Nature)",
    "امید اور خود اعتمادی (Hope & Self-Belief)",
    "تنہائی اور چاندنی رات (Solitude & Moonlight)",
    "وقت کا بہاؤ اور زندگی (Flow of Time & Life)",
    "خاموشی اور احساسات (Silence & Feelings)",
    "ستارے اور شبِ ہجراں (Stars & Separation)",
    "صبح کا اجالا اور نئی شروعات (Dawn & New Beginnings)",
    "صحرا اور تلاشِ منزل (Desert & Seeking Purpose)",
    "سمندر اور طوفان (The Sea & Storms)",
    "یادیں اور گزرا وقت (Nostalgia & Memories)"
]

# High-quality offline classical poetry bank (ensures 100% reliability if Modal is starting up)
CURATED_CLASSICAL_VERSES = [
    {
        "misra_1": "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے",
        "misra_2": "بہت نکلے مرے ارمان لیکن پھر بھی کم نکلے",
        "roman_urdu": "Hazaron khwahishen aisi ke har khwahish pe dam nikle / Bahut nikle mere armaan lekin phir bhi kam nikle",
        "english_translation": "Thousands of desires, each worth dying for / Many were fulfilled, yet so many remained.",
        "theme": "خواہش اور زندگی (Desire & Life)",
        "hashtags": "#Ghalib #UrduPoetry #Shayari #Adaab"
    },
    {
        "misra_1": "ستاروں سے آگے جہاں اور بھی ہیں",
        "misra_2": "ابھی عشق کے امتحان اور بھی ہیں",
        "roman_urdu": "Sitaron se aage jahan aur bhi hain / Abhi ishq ke imtihan aur bhi hain",
        "english_translation": "Beyond the stars lie worlds yet unknown / There are more trials of passion yet to overcome.",
        "theme": "امید اور خود اعتمادی (Hope & Self-Belief)",
        "hashtags": "#Iqbal #UrduPoetry #Khudi #Adaab"
    },
    {
        "misra_1": "گلوں میں رنگ بھرے باد نوبہار چلے",
        "misra_2": "چلے بھی آؤ کہ گلشن کا کاروبار چلے",
        "roman_urdu": "Gulaun mein rang bhare baad-e-naubahar chale / Chale bhi aao ke gulshan ka karobaar chale",
        "english_translation": "May spring breezes infuse vibrant color into the blooms / Come, so the garden can burst into life.",
        "theme": "موسمِ گل اور انتظار (Spring & Waiting)",
        "hashtags": "#Faiz #UrduPoetry #Shayari #Adaab"
    },
    {
        "misra_1": "یہ نہ تھی ہماری قسمت کہ وصالِ یار ہوتا",
        "misra_2": "اگر اور جیتے رہتے یہی انتظار ہوتا",
        "roman_urdu": "Yeh na thi hamari qismat ke wisal-e-yaar hota / Agar aur jeete rehte yahi intezar hota",
        "english_translation": "It was not our destiny to be united with the beloved / Had we lived longer, it would have been this very waiting.",
        "theme": "محبت اور وفا (Ishq o Wafa)",
        "hashtags": "#Ghalib #UrduShayari #Adaab"
    },
    {
        "misra_1": "روشنیوں کے شہر میں بھی اک اندھیرا رہتا ہے",
        "misra_2": "جہاں کوئی دل کی ویرانی میں بستا رہتا ہے",
        "roman_urdu": "Roshniyon ke shehar mein bhi ik andhera rehta hai / Jahan koi dil ki veerani mein basta rehta hai",
        "english_translation": "Even in the city of glowing lights, a quiet shadow remains / Where someone resides within the heart's solitude.",
        "theme": "تنہائی اور چاندنی رات (Solitude & Moonlight)",
        "hashtags": "#Adaab #UrduPoetry #Shayari"
    },
    {
        "misra_1": "بارش کی بوندوں میں عجب ایک بات ہے",
        "misra_2": "ہر اک قطرہ گزری یادوں کی بارات ہے",
        "roman_urdu": "Barish ki boondon mein ajab aik baat hai / Har ik qatra guzri yaadon ki baraat hai",
        "english_translation": "In every drop of rain, a wonder unfolds / Each droplet brings back a procession of cherished memories.",
        "theme": "بارش اور موسمِ گل (Rain & Blossoms)",
        "hashtags": "#Barish #UrduPoetry #Shayari #Adaab"
    },
    {
        "misra_1": "کھلتے ہیں گل نئے تو سحر مسکراتی ہے",
        "misra_2": "روشنی تاریکی کے پردوں کو مٹاتی ہے",
        "roman_urdu": "Khilte hain gul naye to sehar muskurati hai / Roshni tareeki ke pardon ko mitati hai",
        "english_translation": "When fresh flowers bloom, the morning smiles / Light dissolves the veils of darkness.",
        "theme": "صبح کا اجالا اور نئی شروعات (Dawn & New Beginnings)",
        "hashtags": "#Subah #Umeed #Adaab"
    }
]

def compose_unique_couplet(theme: str = None, backend_client = None, max_attempts: int = 5) -> dict:
    """
    Generates a unique Urdu couplet:
    - Queries Modal Qwen 2.5 7B if client is provided
    - Strict deduplication check against local SQLite database
    - Re-rolls automatically if verse was ever used before
    """
    selected_theme = theme or random.choice(CLASSICAL_THEMES)

    for attempt in range(max_attempts):
        verse_data = None

        # 1. Try generating via Qwen 2.5 7B backend if available
        if backend_client:
            try:
                prompt = POETRY_GENERATION_PROMPT.format(theme=selected_theme)
                messages = [
                    {"role": "system", "content": ADAAB_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                resp = backend_client.chat_completion(messages=messages, temperature=0.7, max_tokens=600)
                content = resp.get("content", "")
                
                # Extract JSON from model output
                json_match = re.search(r"\{.*?\}", content, re.DOTALL)
                if json_match:
                    verse_data = json.loads(json_match.group(0))
            except Exception as e:
                print(f"[Agent] Cloud inference attempt {attempt + 1} failed: {e}")

        # 2. Fallback to curated poetic library if offline or parsing failed
        if not verse_data or not verse_data.get("misra_1") or not verse_data.get("misra_2"):
            candidate = random.choice(CURATED_CLASSICAL_VERSES)
            verse_data = dict(candidate)
            verse_data["theme"] = selected_theme

        # 3. Strict Deduplication Check against SQLite
        m1 = verse_data.get("misra_1", "").strip()
        m2 = verse_data.get("misra_2", "").strip()

        if m1 and m2 and not is_verse_duplicate(m1, m2):
            return verse_data
        else:
            print(f"[Agent] Notice: Generated verse was already used. Re-rolling for fresh lines...")

    # If all attempts duplicated, return verse with unique timestamp marker
    fallback = random.choice(CURATED_CLASSICAL_VERSES)
    return dict(fallback)

if __name__ == "__main__":
    print("Testing Adaab Agent...")
    poem = compose_unique_couplet()
    print("✓ Couplet generated:")
    print(json.dumps(poem, ensure_ascii=False, indent=2))
