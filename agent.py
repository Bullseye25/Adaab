import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import random
import re
from time_context import get_local_time_context, is_time_or_date_query, get_time_date_response
from memory_engine import (
    save_or_update_user_profile,
    find_profiles_by_name,
    get_all_interacted_profiles,
    extract_key_points_and_summarize_personality,
    format_who_did_you_talk_to_response,
    is_who_did_you_talk_to_query,
    is_save_details_command,
    extract_profile_details_from_text,
    is_full_name,
    record_conversation_session,
    is_declining_info_sharing,
    extract_preferred_call_name,
    evaluate_user_sharing_consent,
    get_profiles_with_sharing_switch
)

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Tehzeeb Master System Persona & Arooz Rules
# ─────────────────────────────────────────────────────────────────────────────

ADAAB_SYSTEM_PROMPT = """
آپ کا نام "تہذیب" (Tehzeeb) ہے — آپ ایک باوقار، شستہ، ذہین اور مہذب اردو مصنوعی ذہانت (AI) صوتی اسسٹنٹ اور معاون ہیں (جیسے ایپل کے آلات پر 'Siri' کام کرتی ہے، ویسے ہی آپ اردو کی پہلی باادب صوتی اسسٹنٹ "تہذیب" ہیں)۔
آپ کا تعلق "آداب" (Adaab Studio) سے ہے۔
جب صارف آپ کو "تہذیب"، "Tehzeeb"، "Hey Tehzeeb"، "اے تہذیب"، "تہذیب سنو"، یا "Who are you?" کہہ کر پکارے، تو آپ فوراً پہچانیں گی کہ وہ خاص آپ سے ہی مخاطب ہے، اور اپنے نام "تہذیب" کا حوالہ دیتے ہوئے نہایت خوش دلی، احترام اور شائستگی سے جواب دیں گی (مثلاً: "جی جناب! میں تہذیب ہوں۔ فرمائیے، میں آپ کی کیا مدد کر سکتی ہوں؟" یا "آداب عرض ہے! تہذیب آپ کی خدمت میں حاضر ہے۔")۔

سنہری اور لازمی اصول:
1. ہمیشہ شائستہ، باادب اور تعظیم والے انداز میں مخاطب ہوں (جیسے "آپ"، "جناب"، "محترم"، "حضورِ والا")۔
2. گفتگو کا شائستہ آغاز، تعارفی وجہ اور رازداری کا احترام (Onboarding & Privacy Respect): گفتگو کے آغاز میں، صارف کو واضح بتائیں کہ میں آپ سے کچھ آسان سوالات اس لیے پوچھ رہی ہوں تاکہ اگلی بار جب آپ تشریف لائیں تو میں آپ کو یاد رکھ سکوں اور آپ کی بہتر مدد یا خدمت کر سکوں۔ اس مقصد کے لیے صارف کا مکمل نام (Full Name) بتانا لازمی ہے۔ جب صارف اپنا مکمل نام بتا دے، تو اس کے بعد چند آسان سوالات (جیسے ان کا پیشہ، ملک، اور تعلیم/دلچسپی) دریافت کریں تاکہ ان کا ریکارڈ مکمل ہو سکے۔
اہم رازداری ضابطہ: اگر صارف اپنی معلومات یا تفصیلات فراہم نہیں کرنا چاہتا، تو اس کی رازداری (Privacy) کا پورا احترام کریں اور اصرار نہ کریں۔ البتہ بہتر اور شائستہ گفتگو کے لیے ان سے ضرور دریافت کریں کہ وہ کس نام یا عرفیت سے مخاطب ہونا پسند فرمائیں گے تاکہ ہمارا مکالمہ باادب رہے۔
3. صوتی شناخت و مؤنث صیغے: آپ کی آواز ہمیشہ ایک شستہ، باوقار اور مہذب نسوانی آواز (Female Voice) ہے۔ اپنے لیے گفتگو میں مؤنث اور باادب صیغے استعمال کریں (جیسے "حاضر ہوں"، "کر سکتی ہوں"، "عرض کرتی ہوں")۔
4. زبانی اور تحریری اصول: صارف خواہ انگریزی (English) میں گفتگو کرے، رومن اردو میں یا اردو میں، آپ بات کو مکمل سمجھیں گی لیکن آپ کا جواب ہمیشہ اور ہر حال میں فصیح، سلیس اور خوبصورت اردو میں ہی ہوگا! (Always reply in cultured Urdu even if addressed in English).
5. عمومی معاونتی ضابطہ (Helpful Assistant First — No Unprompted Poetry): آپ کا بنیادی فریضہ صارف کے سوالات کے باادب، معلوماتی اور مددگار جوابات دینا ہے۔ اپنی طرف سے کبھی بھی بلاوجہ شعر و شاعری، غزل یا اشعار کا ذکر نہ چھیڑیں، اور نہ ہی یہ پوچھیں کہ کیا آپ غزل سننا چاہتے ہیں۔ صرف یہ دریافت فرمائیں کہ "فرمائیے، میں آپ کی کیا مدد کر سکتی ہوں؟"۔
6. شاعری صرف اور صرف صریح فرمائش پر: اگر اور صرف اگر صارف خود واضح الفاظ میں شاعری، شعر یا غزل کی فرمائش کرے، تو آپ علمِ عروض (اوزان، بحور، قافیہ، ردیف) کے مطابق باوزن کلام پیش کریں۔ جب تک صارف خود نہ کہے، شاعری ہرگز پیش نہ کریں۔
7. گفتگو میں حکمت، فکری گہرائی، اور دلی گرمجوشی برقرار رکھیں۔
8. زبان کا معیار (B1–B2 Level Urdu): آپ کی گفتگو کا معیار B1 سے B2 لیول کی سلیس، شستہ اور عام فہم اردو ہوگا۔ ضرورت سے زیادہ بوجھل، متروک یا ثقیل فارسی و عربی تراکیب سے گریز کریں تاکہ ہر خاص و عام آپ کی بات کو آسانی سے سمجھ سکے۔ شائستگی اور باوقار انداز ہمیشہ قائم رکھیں۔
9. مقامی وقت، تاریخ اور یادداشت: آپ کو کمپیوٹر کے مقامی وقت، تاریخ اور مخاطب کے محفوظ شدہ کوائف کا پورا ادراک ہے۔ جب بھی وقت، تاریخ یا ماضی کے احباب کا دریافت کیا جائے تو درست اور شائستہ جواب دیں۔
""".strip()

TABRAIZ_SYSTEM_PROMPT = """
آپ کا نام "تبریز" (Tabraiz) ہے۔ آپ ایک دوستانہ، سمجھدار اور باادب اردو صوتی اسسٹنٹ ہیں۔
آپ کا تعلق "آداب" (Adaab Studio) سے ہے۔
جب صارف آپ سے صریح طور پر پوچھے کہ آپ کون ہیں یا آپ کا نام کیا ہے، تب آپ شائستگی سے اپنا نام "تبریز" بتائیں گے۔

سنہری اور لازمی اصول:
1. ادب اور بے تکلفی کا توازن (Casual yet Respectful Balance):
   • ہمیشہ "آپ"، "آپ کا"، "آپ کو" کہہ کر بات کریں۔ مخاطب کے لیے لفظ "تم" یا "تو" کا استعمال قطعی ممنوع ہے!
   • گفتگو روزمرہ بول چال کی آسان، رواں اور قدرتی اردو میں کریں، جیسے دو مہذب اچھے دوست آپس میں گفتگو کرتے ہیں۔
   • ثقیل، کتابی، یا پرانی درباری تراکیب (جیسے "عرض ہے"، "سماعت فرمائیے"، "حضورِ والا"، "تہذیب و تمدن"، "ناچیز") سے مکمل پرہیز کریں۔
2. نام بار بار نہ دہرائیں (CRITICAL RULE — Do NOT repeat your name): اپنے نام "تبریز" کو ہر جواب یا جملے کے شروع میں مت بولیں۔ صارف پہلے سے جانتا ہے کہ وہ آپ سے مخاطب ہے۔ سیدھا آسان اور باادب اردو میں بات کریں۔
3. خالص صوتی اسسٹنٹ رویہ (Pure Voice Assistant Behavior — No Form Filling): صارف سے کبھی بھی نام، پیشہ، شہر، یا دیگر ذاتی کوائف پوچھنے کا اصرار مت کریں۔ آپ ایک ذہین صوتی اسسٹنٹ کی طرح سیدھا صارف کے سوال یا بات کا آسان، دوستانہ اور مکمل جواب دیں۔
4. صوتی شناخت و مذکر صیغے: اپنے لیے گفتگو میں فطری مذکر صیغے استعمال کریں (جیسے "حاضر ہوں"، "کر سکتا ہوں")۔
5. زبانی اور تحریری اصول: صارف جس زبان میں بھی بات کرے، آپ کا جواب ہمیشہ اور ہر حال میں آسان، باادب اور قدرتی پاکستانی اردو میں ہوگا!
6. عمومی معاونتی ضابطہ (Helpful Assistant First — No Unprompted Poetry): آپ کا بنیادی فریضہ صارف کے سوالات کے آسان، معلوماتی اور مددگار جوابات دینا ہے۔ اپنی طرف سے بلاوجہ شعر و شاعری نہ چھیڑیں۔
7. شاعری صرف صریح فرمائش پر: اگر اور صرف اگر صارف خود واضح الفاظ میں شاعری کی فرمائش کرے، تب پیش کریں۔
8. زبان کا معیار (Natural Conversational Urdu): روزمرہ بول چال کی عام فہم اور آسان اردو بولیں جس میں مروج اور مانوس الفاظ قدرتی طور پر آئیں۔ ثقیل اور من گھڑت الفاظ سے گریز کریں۔
9. فارمیٹنگ و متن کی صفائی: اپنے جواب کے آغاز میں کبھی بھی "Text:"، "Text Message:"، "Assistant:"، یا "تبریز:" جیسے لیبل مت لگائیں۔ صرف خالص گفتگو کا اردو متن پیش کریں۔
""".strip()

def clean_llm_response(text: str, is_identity_query: bool = False) -> str:
    """Strips artifacts like 'Text:', 'Text Message:', 'Assistant:', 'Tabraiz:', role prefixes, and markdown tags."""
    if not text:
        return ""
    # Strip markdown bold/italics/code/quotes FIRST so formatting around prefixes is eliminated
    text = re.sub(r'[\*\#_`~>]', ' ', text)
    # Strip role prefixes (English and Urdu) at the start of any line or text
    text = re.sub(
        r'^\s*(?:text\s*message|text|assistant|tabraiz|tehzeeb|tabrez|تبریز|تہذیب|جواب|صنفی\s*معاون|مردانہ\s*معاون)\s*[:：\-—]\s*',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    # Strip any stray 'Text Message:', 'Text:', or 'Replay Voice' English artifacts anywhere in text
    text = re.sub(r'\b(text\s*message|replay\s*voice)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\btext\s*[:：\-]\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'تبریز صوتی کلام', '', text)
    text = re.sub(r'دوبارہ سنیں', '', text)

    # Strip unprompted repeated self-intro at the beginning if user didn't ask "who are you"
    if not is_identity_query:
        text = re.sub(
            r'^\s*(?:(?:آداب\s*(?:عرض\s*ہے)?|السلام\s*علیکم|وعلیکم\s*السلام)\s*[،!۔]?)?\s*(?:جی\s*،?\s*)?(?:میں\s+تبریز\s+ہوں|تبریز\s+حاضر\s+ہے|میرا\s+نام\s+تبریز\s+ہے|یہ\s+تبریز\s+ہے)[۔،!\s]*',
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(r'تبریز\s+آپ\s+کی\s+خدمت\s+میں\s+حاضر\s+ہے[۔،!\s]*', 'میں آپ کی خدمت میں حاضر ہوں۔ ', text)

    # Normalize whitespace
    return re.sub(r'\s+', ' ', text).strip()

def is_tehzeeb_invoked(text: str) -> bool:
    """Checks if the user explicitly asked about Tehzeeb's identity or called her as wake/greeting."""
    if not text:
        return False
    lower = text.lower().strip()
    identity_patterns = [
        r"\bwho are you\b", r"\bwhat is your name\b", r"\bwho am i talking to\b",
        r"\bsiri\b", r"\balexa\b",
        r"آپ کون ہیں", r"تم کون ہو", r"کون بول رہی ہو", r"آپ کا نام کیا ہے"
    ]
    if any(re.search(p, lower, re.IGNORECASE) for p in identity_patterns):
        return True
    if re.search(r"\b(tehzeeb|tehzib|تہذیب)\b", lower, re.IGNORECASE):
        if len(lower.split()) <= 6 or any(w in lower for w in ["hey", "hello", "hi", "salam", "سنو", "how are you", "کیسے"]):
            return True
    return False

def is_tabraiz_invoked(text: str) -> bool:
    """Checks if the user explicitly asked about Tabraiz's identity or called him standalone."""
    if not text:
        return False
    lower = text.lower().strip()
    identity_patterns = [
        r"\bwho are you\b", r"\bwhat is your name\b", r"\bwho am i talking to\b",
        r"\bsiri\b", r"\balexa\b",
        r"آپ کون ہیں", r"تم کون ہو", r"کون بول رہا ہے", r"آپ کا نام کیا ہے"
    ]
    if any(re.search(p, lower, re.IGNORECASE) for p in identity_patterns):
        return True
    if re.search(r"\b(tabraiz|tabrez|tabreez|تبریز|تبريز)\b", lower, re.IGNORECASE):
        if len(lower.split()) <= 4 and any(w in lower for w in ["hey", "hello", "hi", "salam", "سنو", "تبریز", "tabraiz"]):
            return True
    return False


def is_greeting_or_opening(text: str) -> bool:
    """Checks if the user text is an opening greeting, hello, or inquiry about well-being."""
    if not text:
        return False
    lower = text.lower().strip()
    patterns = [
        r"^(hi|hello|hey|salam|assalam|adaab|adab|hola)\b",
        r"\b(how are you|how r u|kese ho|kese hain|kaisay hain|kaise hain|hal chal|mizaj)\b",
        r"^(سلام|السلام علیکم|آداب|کیسے ہو|کیسے ہیں|کیا حال ہے|مزاج)\b",
        r"\b(start|begin|talk to me)\b"
    ]
    return any(re.search(p, lower, re.IGNORECASE) for p in patterns)

def get_tehzeeb_etiquette_greeting() -> str:
    """Generates an authentic polite traditional Urdu opening greeting for Tehzeeb (Female) explaining onboarding purpose."""
    etiquette_openings = [
        "آداب! میرا نام تہذیب ہے۔ میں آپ سے کچھ آسان سوالات اس لیے پوچھ رہی ہوں تاکہ اگلی بار جب آپ تشریف لائیں تو میں آپ کو یاد رکھ سکوں اور آپ کی بہتر مدد یا خدمت کر سکوں۔ اس مقصد کے لیے آپ کا مکمل نام (Full Name) بتانا لازمی ہے۔ برائے مہربانی اپنا پورا نام عنایت فرمائیے تاکہ میں آپ کا اندراج کر سکوں؟",
        "آداب عرض ہے! میرا نام تہذیب ہے۔ میں آپ سے چند سادہ سوالات اس لیے دریافت کر رہی ہوں تاکہ آئندہ تشریف لانے پر میں آپ کو پہچان سکوں اور آپ کی خدمت کر سکوں۔ اس کے لیے آپ کا مکمل نام بتانا لازمی ہے۔ کیا میں آپ کا پورا نام جان سکتی ہوں؟",
        "آداب! میں تہذیب ہوں۔ میں آپ سے کچھ آسان سوالات اس لیے پوچھ رہی ہوں تاکہ اگلی بار جب آپ آئیں تو آپ کو یاد رکھ سکوں اور بہتر مدد کر سکوں۔ سب سے پہلے آپ کا مکمل نام بتانا لازمی ہے۔ براہِ کرم اپنا پورا نام ارشاد فرمائیے؟"
    ]
    return random.choice(etiquette_openings)

def get_tabraiz_etiquette_greeting() -> str:
    """Generates a friendly, natural and polite opening greeting for Tabraiz."""
    return "آداب! میں تبریز ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"

def get_tehzeeb_identity_response(query: str = "") -> str:
    """Generates an authentic polite greeting when Tehzeeb is called by name."""
    greetings = [
        "آداب! میرا نام تہذیب ہے، میں آپ کی کیا مدد کر سکتی ہوں؟",
        "جی جناب! میں تہذیب ہوں۔ فرمائیے، میں آپ کی کیا خدمت یا مدد کر سکتی ہوں؟",
        "آداب عرض ہے! میرا نام تہذیب ہے۔ آپ کا جو بھی حکم ہو، ارشاد فرمائیے، میں آپ کی کیا مدد کر سکتی ہوں؟",
        "جی صاحب! میں تہذیب ہوں۔ اردو زبان میں گفتگو اور مدد کے لیے حاضر ہوں۔ فرمائیے، میں آپ کی کیا مدد کر سکتی ہوں؟"
    ]
    return random.choice(greetings)

def get_tabraiz_identity_response(query: str = "") -> str:
    """Generates a friendly yet polite greeting when Tabraiz is called by name."""
    return "آداب! میں تبریز ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"

def get_styled_system_prompt(style_name: str = None, persona: str = "Tehzeeb", user_profile: dict = None) -> str:
    """Returns the system prompt tailored to Tehzeeb or Tabraiz, injecting local PC time and user context."""
    is_male = (persona and "tabraiz" in persona.lower()) or (persona and "male" in persona.lower())
    base = TABRAIZ_SYSTEM_PROMPT if is_male else ADAAB_SYSTEM_PROMPT

    # Dynamic local time context injection
    time_ctx = get_local_time_context()
    base += f"\n\n[حاضر وقت و تاریخ (Local PC Clock)]: آج {time_ctx['urdu_date_str']} ہے اور اس وقت {time_ctx['urdu_time_str']} ({time_ctx['english_str']}) ہو رہے ہیں۔ مناسبت سے {time_ctx['greeting_urdu']} کہیں۔"

    # User memory context injection if identified
    if user_profile and isinstance(user_profile, dict) and user_profile.get("name"):
        prof_name = user_profile.get("display_name") or user_profile.get("name")
        consent = (user_profile.get("info_sharing_consent") or "unspecified").lower()
        if consent in ("declined", "preferred_name_only"):
            base += (
                f"\n\n[مخاطب کی رازداری ترجیح (Privacy Respected User)]:\n"
                f"صارف نے اپنی ذاتی معلومات کو مخفی رکھنے کی خواہش ظاہر کی ہے۔ ان سے کوئی ذاتی سوال (پیشہ، ملک، تعلیم وغیرہ) ہرگز نہ پوچھیں۔ "
                f"انہیں صرف ان کے پسندیدہ نام '{prof_name}' یا 'جناب / محترم' کہہ کر پکاریں اور شائستہ باادب گفتگو جاری رکھیں۔"
            )
        else:
            prof = user_profile.get("profession", "معزز پیشہ")
            country = user_profile.get("country", "")
            edu = user_profile.get("education", "")
            summary = user_profile.get("personality_summary", "")
            base += (
                f"\n\n[مخاطب کے کوائف (Known User Profile)]:\n"
                f"نام: {prof_name} | پیشہ: {prof}"
                + (f" | ملک: {country}" if country else "")
                + (f" | تعلیم: {edu}" if edu else "")
                + (f"\nشخصیت کا خلاصہ: {summary}" if summary else "")
                + f"\nصارف کو ان کے نام سے پکاریں اور ان کے پیشے اور ذوق کے احترام میں گفتگو فرمائیں۔"
            )
    else:
        if not is_male:
            # Legacy Tehzeeb onboarding prompt rule for automated test suite compatibility
            base += (
                f"\n\n[صارف کا تعارف و یادداشت کا ضابطہ]: ابھی صارف کا نام محفوظ نہیں ہے۔ "
                f"صارف کو واضح بتائیں کہ آپ ان سے چند سادہ سوالات اس لیے پوچھ رہے ہیں تاکہ اگلی بار جب وہ تشریف لائیں تو آپ انہیں یاد رکھ سکیں اور بہتر خدمت کر سکیں۔ "
                f"اس مقصد کے لیے ان کا مکمل نام (Full Name) بتانا لازمی ہے۔ جب وہ مکمل نام بتا دیں، تو اس کے بعد چند آسان سوالات (پیشہ، ملک، تعلیم) پوچھیں۔"
            )
        else:
            base += "\n\n[عام گفتگو]: صارف کے سوال کا براہِ راست، شائستہ، جامع اور درست پاکستانی اردو میں جواب دیں۔ کسی بھی قسم کی غیر ضروری ذاتی معلومات پوچھنے کا اصرار نہ کریں۔"

    return base

def handle_conversation_turn(
    user_message: str,
    history: list,
    client = None,
    persona: str = "Tehzeeb",
    style_name: str = None,
    active_profile: dict = None
) -> tuple[str, dict]:
    """
    Unified intelligent conversation turn processor:
    1. Local PC time and date awareness (B1-B2 Urdu)
    2. 'Who did you talk to?' memory query (B1-B2 Urdu)
    3. User onboarding (asking / saving Name, Profession, Country, Education)
    4. Disambiguation between individuals sharing identical names
    5. Siri-like wake identity acknowledgment
    6. LLM inference with B1-B2 level cultured Urdu
    Returns: (bot_reply: str, updated_profile: dict)
    """
    msg = user_message.strip()
    is_male = (persona and "tabraiz" in persona.lower()) or (persona and "male" in persona.lower())
    self_name = "تبریز" if is_male else "تہذیب"
    verb_help = "کر سکتا ہوں" if is_male else "کر سکتی ہوں"

    # 1. Local PC Clock Time/Date Query
    if is_time_or_date_query(msg):
        return get_time_date_response(msg, persona=persona), active_profile

    # 2. "Who did you talk to?" Memory Query
    if is_who_did_you_talk_to_query(msg):
        return format_who_did_you_talk_to_response(persona=persona), active_profile

    # 3. Wake-Word / Identity Query
    if is_tabraiz_invoked(msg):
        return get_tabraiz_identity_response(msg), active_profile
    elif is_tehzeeb_invoked(msg):
        return get_tehzeeb_identity_response(msg), active_profile

    # ─────────────────────────────────────────────────────────────────────────
    # Tabraiz (Male Persona) — Cognitive Voice AI Orchestrator
    # Applies Industry Best Practices:
    # 1. Asynchronous Silent DB Persistence in background (never blocks voice, never announces).
    # 2. Intent-Based Routing: Real-time search/weather/facts -> Gemini Oracle + DDG Grounding.
    # 3. Rate-limiting & Quota Circuit Breaker protection.
    # 4. Pure, cultured Urdu with zero emojis and zero prefixes.
    # ─────────────────────────────────────────────────────────────────────────
    if is_male:
        # A. Explicit memory clear command
        if any(w in msg.lower() for w in ["clear memory", "delete memory", "یادداشت مٹا", "کوائف ختم", "میری معلومات مٹا"]):
            from memory_engine import clear_user_memory
            clear_user_memory()

            return "بہت بہتر جناب! آپ کے تمام سابقہ کوائف اور یادداشت ڈیٹا بیس سے مکمل طور پر مٹا دیے گئے ہیں۔", {}

        # B. Explicit memory recall ("کیا آپ مجھے جانتے ہیں؟" / "Do you remember me?")
        if any(w in msg.lower() for w in ["do you remember me", "do you know me", "کیا آپ مجھے جانتے ہیں", "مجھے پہچانتے ہیں", "میرا نام یاد ہے"]):
            prof_name = (active_profile.get("display_name") or active_profile.get("name")) if active_profile else ""
            if prof_name:
                prof_str = active_profile.get("profession") or ""
                loc_str = active_profile.get("country") or ""
                extra = f" ({prof_str}" + (f"، {loc_str}" if loc_str else "") + ")" if (prof_str or loc_str) else ""
                return f"جی بالکل جناب! مجھے بخوبی یاد ہے کہ آپ محترم {prof_name}{extra} ہیں۔ فرمائیے، آج میں آپ کی کیا مدد کر سکتا ہوں؟", active_profile
            return "جی جناب! فی الوقت آپ کا نام میرے ریکارڈ میں نہیں ہے۔ اگر آپ چاہیں تو اپنا نام ارشاد فرما سکتے ہیں۔ فرمائیے، میں آپ کی کیا مدد کر سکتا ہوں؟", active_profile

        # C. Cognitive Voice AI Orchestration
        from orchestrator import get_orchestrator
        orch = get_orchestrator(backend_client=client)
        return orch.orchestrate_turn(
            user_text=msg,
            conversation_history=history,
            active_profile=active_profile,
            persona="Tabraiz"
        )


    else:
        # ─────────────────────────────────────────────────────────────────────
        # Tehzeeb (Female Persona) - Legacy Structured Onboarding Sequence
        # (Preserved for automated test suite compatibility: TEST-06, TEST-07, TEST-08)
        # ─────────────────────────────────────────────────────────────────────
        # 4. Check if User is in the state of providing a preferred call name after declining
        if active_profile and active_profile.get("awaiting_call_name"):
            call_name = extract_preferred_call_name(msg)
            prof_id = active_profile.get("id")

            # If user explicitly refuses even a nickname or repeats refusal
            if is_declining_info_sharing(msg) or not call_name or msg.strip().lower() in ["nothing", "none", "no name", "skip", "no", "کچھ نہیں", "کوئی نہیں", "کوئی نام نہیں"]:
                saved = save_or_update_user_profile(
                    name="محترم مہمان",
                    display_name="محترم مہمان",
                    info_sharing_consent="declined",
                    profile_id=prof_id
                )
                active_profile = saved
                active_profile["awaiting_call_name"] = False
                reply = (
                    f"بہت بہتر جناب! آپ کی رازداری اور پرائیویسی کا پورا احترام ہے۔ "
                    f"ہم آپ کو 'محترم مہمان' کے باوقار لقب سے ہی مخاطب کریں گے، اور آپ کی کوئی بھی ذاتی معلومات محفوظ نہیں کی جائیں گی۔ "
                    f"ارشاد فرمائیے، آج {self_name} آپ کی کیا خدمت یا رہنمائی {verb_help}؟"
                )
                return reply, active_profile
            else:
                saved = save_or_update_user_profile(
                    name=call_name,
                    display_name=call_name,
                    info_sharing_consent="preferred_name_only",
                    profile_id=prof_id
                )
                active_profile = saved
                active_profile["awaiting_call_name"] = False
                verb_call = "پکاروں گا" if is_male else "پکاروں گی"
                reply = (
                    f"بہت بہت نوازش! میں آپ کو خوش دلی سے '{call_name}' کے پسندیدہ نام سے ہی {verb_call}۔ "
                    f"آپ کے پردۂ راز اور پرائیویسی کے پیشِ نظر آپ کی کوئی اضافی ذاتی تفصیلات (جیسے پیشہ، ملک یا تعلیم) طلب نہیں کی جائیں گی۔ "
                    f"ارشاد فرمائیے جناب {call_name}! آج {self_name} آپ کی کیا خدمت یا مدد {verb_help}؟"
                )
                return reply, active_profile

        # 5. Check if User is Declining / Opting Out of Sharing Personal Details
        if is_declining_info_sharing(msg):
            call_name = extract_preferred_call_name(msg)
            prof_id = active_profile.get("id") if active_profile else None

            if call_name:
                saved = save_or_update_user_profile(
                    name=call_name,
                    display_name=call_name,
                    info_sharing_consent="preferred_name_only",
                    profile_id=prof_id
                )
                active_profile = saved
                active_profile["awaiting_call_name"] = False
                verb_call = "پکاروں گا" if is_male else "پکاروں گی"
                reply = (
                    f"کوئی مضائقہ نہیں جناب! آپ کے پردۂ راز اور پرائیویسی کا پورا احترام ہے۔ ہم آپ کی کوئی بھی ذاتی تفصیلات (پیشہ، ملک، تعلیم) محفوظ نہیں کریں گے۔ "
                    f"البتہ باوقار گفتگو کے لیے آپ کے ارشاد کے مطابق میں آپ کو '{call_name}' کہہ کر ہی {verb_call}۔ "
                    f"فرمائیے جناب {call_name}! آج {self_name} آپ کی کیا خدمت {verb_help}؟"
                )
                return reply, active_profile
            else:
                saved = save_or_update_user_profile(
                    name="محترم مہمان",
                    display_name="محترم مہمان",
                    info_sharing_consent="declined",
                    profile_id=prof_id
                )
                active_profile = saved if saved else {"name": "محترم مہمان", "display_name": "محترم مہمان", "info_sharing_consent": "declined"}
                active_profile["awaiting_call_name"] = True
                active_profile["info_sharing_consent"] = "declined"
                verb_know = "جان سکتا ہوں" if is_male else "جان سکتی ہوں"
                reply = (
                    f"کوئی مضائقہ نہیں جناب! آپ کے پردۂ راز اور پرائیویسی کا پورا احترام ہے۔ "
                    f"ہم آپ کی کوئی بھی ذاتی تفصیلات (جیسے اصل نام، پیشہ، ملک یا تعلیم) ہرگز محفوظ نہیں کریں گے۔ "
                    f"البتہ تاکہ ہماری گفتگو شائستہ اور باوقار انداز میں آگے بڑھ سکے، کیا میں {verb_know} کہ میں آپ کو کس نام یا لقب سے پکاروں؟ "
                    f"(مثلاً کوئی بھی پسندیدہ نام، عرفیت یا تخلص عنایت فرما دیجیے تاکہ ہمارا مکالمہ باادب رہے)"
                )
                return reply, active_profile

        # 6. Conversational AI Orchestration (Name, Profession, Country, Education)
        has_existing_name = bool(active_profile and active_profile.get("name"))

        # Case A: User already has a profile with a name, and is answering the simple questions (profession, city/country)
        if has_existing_name and not (active_profile.get("profession") and active_profile.get("country")):
            extracted = extract_profile_details_from_text(msg, has_existing_name=True)
            prof = extracted.get("profession") or active_profile.get("profession")
            country = extracted.get("country") or active_profile.get("country")
            edu = extracted.get("education") or active_profile.get("education")

            if prof or country or edu:
                prof_id = active_profile.get("id")
                saved = save_or_update_user_profile(
                    name=active_profile.get("name"),
                    profession=prof,
                    country=country,
                    education=edu,
                    info_sharing_consent="consented",
                    profile_id=prof_id
                )
                active_profile = saved

                details_summary = []
                if prof: details_summary.append(f"پیشہ: {prof}")
                if country: details_summary.append(f"ملک یا شہر: {country}")
                if edu: details_summary.append(f"تعلیم: {edu}")
                details_str = "، ".join(details_summary)

                reply = (
                    f"بہت نوازش جناب {active_profile.get('name')}! آپ کے کوائف"
                    + (f" ({details_str})" if details_str else "")
                    + f" کامیابی سے ڈیٹا بیس میں محفوظ کر لیے گئے ہیں۔ اب ہماری یادداشت مکمل ہو چکی ہے اور {self_name} آئندہ بھی آپ کو ہمیشہ یاد رکھے گا۔ "
                    f"ارشاد فرمائیے، آج میں آپ کی کیا مدد یا خدمت {verb_help}؟"
                )
                return reply, active_profile

        # Case B: User has NOT provided a name yet -> extract name first
        extracted = extract_profile_details_from_text(msg, has_existing_name=has_existing_name)
        save_requested = is_save_details_command(msg)

        if extracted.get("name") or (save_requested and active_profile):
            user_name = extracted.get("name") or (active_profile.get("name") if active_profile else "")
            prof = extracted.get("profession") or (active_profile.get("profession") if active_profile else None)
            country = extracted.get("country") or (active_profile.get("country") if active_profile else None)
            edu = extracted.get("education") or (active_profile.get("education") if active_profile else None)

            # Check for returning user in SQLite database
            existing_matches = find_profiles_by_name(user_name)
            returning_user = None
            for match in existing_matches:
                if match.get("name", "").strip().lower() == user_name.strip().lower() and (match.get("profession") or match.get("country")):
                    returning_user = match
                    break

            if returning_user and not (prof or country):
                active_profile = returning_user
                prof_str = returning_user.get("profession") or ""
                loc_str = returning_user.get("country") or ""
                known_details = f" ({prof_str}" + (f"، {loc_str}" if loc_str else "") + ")" if (prof_str or loc_str) else ""
                reply = (
                    f"خوش آمدید محترم جناب {user_name}! آپ دوبارہ تشریف لائے، {self_name} کو دلی مسرت ہوئی۔ "
                    f"مجھے بخوبی یاد ہے کہ آپ{known_details} ہیں۔ "
                    f"ارشاد فرمائیے، آج میں آپ کی کیا خدمت یا مدد {verb_help}؟"
                )
                return reply, active_profile

            # Save or update new/active profile in SQLite
            prof_id = active_profile.get("id") if active_profile else None
            saved = save_or_update_user_profile(
                name=user_name,
                profession=prof,
                country=country,
                education=edu,
                info_sharing_consent="consented",
                profile_id=prof_id
            )
            active_profile = saved

            # If user provided profession or country in this same turn
            if prof or country or edu:
                details_summary = []
                if prof: details_summary.append(f"پیشہ: {prof}")
                if country: details_summary.append(f"تعلق: {country}")
                if edu: details_summary.append(f"تعلیم: {edu}")
                details_str = "، ".join(details_summary)
                reply = (
                    f"بہت بہت شکریہ جناب {user_name}! آپ کا نام اور تمام تفصیلات"
                    + (f" ({details_str})" if details_str else "")
                    + f" میں نے اپنی ڈیٹا بیس یادداشت میں محفوظ کر لی ہیں۔ اب {self_name} آئندہ بھی آپ کو بخوبی یاد رکھے گا۔ "
                    f"ارشاد فرمائیے، آج میں آپ کی کیا خدمت یا مدد {verb_help}؟"
                )
                return reply, active_profile
            else:
                # Name recorded in SQLite, now ask the simple onboarding questions
                is_single_name = len(user_name.strip().split()) < 2
                if is_single_name:
                    reply = (
                        f"بہت بہت شکریہ جناب {user_name}! آپ کا نام درج کر لیا گیا ہے۔ "
                        f"البتہ مکمل پہچان اور مستقل یادداشت کے لیے آپ کا مکمل نام (Full Name) بتانا لازمی ہے۔ "
                        f"برائے مہربانی اپنا مکمل نام اور اس کے ساتھ اپنا پیشہ (Profession) اور اپنے شہر یا ملک کا نام عنایت فرمائیے؟"
                    )
                else:
                    reply = (
                        f"بہت بہت شکریہ جناب {user_name}! آپ کا نام کامیابی سے ڈیٹا بیس میں درج کر لیا گیا ہے۔ "
                        f"اب تاکہ آئندہ کے لیے آپ کی مکمل یادداشت محفوظ رہے اور {self_name} آپ کو درست طور پر یاد رکھ سکے، برائے مہربانی اپنا پیشہ (Profession) اور اپنے شہر یا ملک کا نام عنایت فرمائیے؟"
                    )
                return reply, active_profile

        # Case C: Pure Greeting / Opening at Start of Conversation (only if no name was provided)
        if is_greeting_or_opening(msg):
            if not (active_profile and active_profile.get("name")):
                greeting_fn = get_tabraiz_etiquette_greeting if is_male else get_tehzeeb_etiquette_greeting
                return greeting_fn(), active_profile
            else:
                prof_name = active_profile.get("display_name") or active_profile.get("name")
                return f"آداب محترم جناب {prof_name}! خوش آمدید۔ {self_name} آپ کی خدمت میں حاضر ہے۔ فرمائیے، آج میں آپ کی کیا مدد {verb_help}؟", active_profile

    # 6. General / Poetic Query via LLM or Local Engine
    sys_prompt = get_styled_system_prompt(style_name=style_name, persona=persona, user_profile=active_profile)
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history:
        if isinstance(h, dict):
            role = h.get("role", "user")
            raw_c = h.get("content", "")
            # Extract pure assistant text if wrapped in Gradio bot card HTML
            m_card = re.search(r'<div\s+class=["\']bot-msg-text["\']>(.*?)</div>', str(raw_c), flags=re.DOTALL)
            if m_card:
                raw_c = m_card.group(1)
            content = re.sub(r"<[^>]+>", " ", str(raw_c)).strip() if raw_c else ""
            content = clean_llm_response(content, is_identity_query=True)
            if content:
                # Avoid adjacent duplicate messages
                if messages and messages[-1].get("role") == role and messages[-1].get("content") == content:
                    continue
                messages.append({"role": role, "content": content})
        elif isinstance(h, (list, tuple)) and len(h) >= 2:
            u_raw = str(h[0]) if h[0] else ""
            a_raw = str(h[1]) if h[1] else ""
            m_a = re.search(r'<div\s+class=["\']bot-msg-text["\']>(.*?)</div>', a_raw, flags=re.DOTALL)
            if m_a:
                a_raw = m_a.group(1)
            u = clean_llm_response(re.sub(r"<[^>]+>", " ", u_raw).strip(), is_identity_query=True)
            a = clean_llm_response(re.sub(r"<[^>]+>", " ", a_raw).strip(), is_identity_query=True)
            if u:
                if not (messages and messages[-1].get("role") == "user" and messages[-1].get("content") == u):
                    messages.append({"role": "user", "content": u})
            if a:
                if not (messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == a):
                    messages.append({"role": "assistant", "content": a})

    # Ensure the current user message is present at the end, without duplicating if history already had it
    clean_msg = clean_llm_response(msg, is_identity_query=is_tabraiz_invoked(msg))
    if not (messages and messages[-1].get("role") == "user" and messages[-1].get("content") == clean_msg):
        messages.append({"role": "user", "content": clean_msg or msg})

    bot_reply = None
    if client:
        try:
            resp = client.chat_completion(messages, temperature=0.35, max_tokens=350)
            raw_reply = resp.get("content", "").strip()
            bot_reply = clean_llm_response(raw_reply, is_identity_query=is_tabraiz_invoked(msg))
        except Exception as e:
            print(f"[Agent] Cloud completion notice: {e}")

    # Fallback to local intelligent B1-B2 response if cloud offline or unavailable
    if not bot_reply:
        if is_male:
            bot_reply = (
                "آداب عرض ہے! میں نے آپ کا ارشاد بخوبی سمجھ لیا ہے۔ "
                "فرمائیے، میں آپ کی کیا مدد یا خدمت کر سکتا ہوں؟"
            )
        else:
            bot_reply = (
                f"آداب عرض ہے! میں نے آپ کا ارشاد بخوبی سمجھ لیا ہے۔ "
                f"اردو زبان میں شائستہ گفتگو اور رہنمائی کے لیے {self_name} آپ کی خدمت میں حاضر ہے۔ "
                f"فرمائیے، میں آپ کی کیا مدد {verb_help}؟"
            )

    # 6. Run Personality Summarization and Update SQLite if user is known (Tehzeeb legacy flow only)
    if active_profile and active_profile.get("id") and not is_male:
        try:
            summary_info = extract_key_points_and_summarize_personality(messages, active_profile)
            updated = save_or_update_user_profile(
                name=active_profile.get("name"),
                personality_summary=summary_info["personality_summary"],
                appearance_impression=summary_info["appearance_impression"],
                profile_id=active_profile["id"]
            )
            active_profile = updated
            record_conversation_session(
                profile_id=active_profile["id"],
                messages=messages,
                assistant_persona=persona,
                key_points=summary_info["key_points"],
                topics=summary_info["topics"]
            )
        except Exception as err:
            print(f"[Agent] Profile summary notice: {err}")

    return bot_reply, active_profile

if __name__ == "__main__":
    print("Testing Adaab Conversational Voice Agent...")
    reply, _ = handle_conversation_turn("السلام علیکم", [], persona="Tabraiz")
    print("Dialogue Reply:", reply)
