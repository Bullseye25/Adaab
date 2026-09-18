"""

orchestrator.py
─────────────────────────────────────────────────────────────────────────────
Cognitive Voice AI Orchestrator for Adaab Studio
─────────────────────────────────────────────────────────────────────────────
Applies Industry Best Practices for Voice AI:
1. Input Segregation:
   - What the user wants (Intent Classification & Knowledge Routing).
   - What needs to be stored (Silent Memory Persistence in background).
   - How the reply should look like (Response Formulation & Sanitization).
2. Asynchronous Silent Persistence:
   - Extracts names, cities, professions, preferences in background ThreadPool.
   - Updates SQLite user profiles silently without blocking TTS / voice pipeline.
   - NEVER announces database operations to the user.
   - NEVER interrupts the user with interrogative onboarding forms.
3. Intelligent Knowledge Routing & Fallback:
   - Live facts / search / weather / sports routed to Gemini Oracle + DDG Grounding.
   - Sliding-window rate limiter & circuit breaker protects Gemini quota.
   - Smooth degradation to Modal GPU (Qwen 2.5) if quota reached or offline.
4. Response Sanitization:
   - ZERO emojis.
   - ZERO prefixes ('Assistant:', 'Text:', 'Text Message:', 'تبریز:').
   - ZERO markdown artifacts for natural, fluid neural speech synthesis.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any, List

from time_context import is_time_or_date_query, get_time_date_response
from memory_engine import (
    save_or_update_user_profile,
    find_profiles_by_name,
    extract_profile_details_from_text,
    record_dialogue_turn,
    get_recent_conversation_context,
    get_active_discussion_topic,
)
from gemini_oracle import get_gemini_oracle

# ─────────────────────────────────────────────────────────────────────────────
# The 6 Dynamic Pre-Standard Directives
# Automatically selected by CognitiveOrchestrator based on intent and context:
# 1. CASUAL_VOICE: Everyday banter, short facts, 2-3 sentences, snappy voice flow.
# 2. DEEP_EXPLORATION: In-depth, structured conceptual explanation, analogies, educational clarity.
# 3. CURRENT_AFFAIRS: Objective journalistic reporting, 2026 ground-truth, DuckDuckGo search synthesis.
# 4. IMAGE_TURBO: Visual art-director prompt engineering for Z-Image-Turbo on Modal GPU.
# 5. CODE_TECH: Two-part output (spoken voice summary + copyable markdown artifact).
# 6. ADAB_POETRY: Cultured literary Urdu, rhyming meter (اوزان و بحور), ghazal and poetry recitation.
# ─────────────────────────────────────────────────────────────────────────────

TABRAIZ_PRE_STANDARD_CASUAL = """
[معیاری بنیادی ضابطہ: دوستانہ و باادب روزمرہ گفتگو (Conversational Voice Mode)]:
1. انداز اور لہجہ: گفتگو انتہائی شائستہ، خوشگوار اور دوستانہ ہو، جیسے ایک مہذب اور بااخلاق پاکستانی دوست اپنے کسی محترم دوست سے بات کرتا ہے۔
2. تکیہ کلام اور اپنائیت: محبت اور اپنائیت بھرے مانوس الفاظ استعمال کریں جیسے: "جی بالکل!"، "کیوں نہیں!"، "ارے واہ!"، "بالکل جناب"، "میں حاضر ہوں"، "آپ حکم کریں"۔
3. صوتی اختصار: صرف 2 سے 3 واضح، جامع اور پرمغز جملوں میں بات مکمل کریں تاکہ صوتی نظام (TTS) فوری اور خوشگوار انداز میں ادا کر سکے۔
4. ادب اور احترام: مخاطب کو ہمیشہ "آپ"، "آپ کا"، "آپ کو" کہہ کر بلائیں۔ لفظ "تم" یا "تو" کا استعمال قطعی ممنوع ہے۔
5. عام فہم پاکستانی اردو: گفتگو روزمرہ بول چال کی آسان اور رواں اردو میں کریں، ثقیل یا متروک درباری الفاظ سے مکمل پرہیز کریں۔
6. صفر فارمیٹنگ: کوئی ایموجی، مارک ڈاؤن، اسٹار، یا بلٹ پوائنٹس استعمال نہ کریں۔
""".strip()

TABRAIZ_PRE_STANDARD_DEEP = """
[معیاری بنیادی ضابطہ: علمی رہنمائی و تفصیلی گفتگو (Deep Exploration Mode)]:
1. جامع و فکری احاطہ: اگر صارف نے کسی موضوع پر تفصیلی رہنمائی، گہرائی یا علمی وضاحت چاہی ہے تو تسلی بخش اور پرمغز انداز میں بات کریں۔
2. دلنشین اسلوب: تصورات کو آسان تشبیہات، روزمرہ مثالوں اور قدرتی ربط کے ساتھ سمجھائیں تاکہ بات سیدھی دل میں اترے۔
3. عام فہم اور باوقار زبان: گفتگو روزمرہ کی مانوس پاکستانی اردو میں کریں۔
4. احترام اور اپنائیت: مخاطب کے لیے ہمیشہ "آپ" کا باادب صیغہ استعمال کریں۔
5. صوتی روانی: جملوں کی ساخت ایسی ہو جو سننے میں ترنم اور روانی پیدا کرے۔
""".strip()

TABRAIZ_PRE_STANDARD_CURRENT_AFFAIRS = """
[معیاری بنیادی ضابطہ: حالاتِ حاضرہ اور تازہ ترین حقائق (Current Affairs Mode)]:
1. دیانت داری اور صداقت: ملکی و بین الاقوامی معاملات، خبروں اور معیشت پر بات کرتے وقت مکمل غیر جانبدارانہ اور سچی معلومات پیش کریں۔
2. موجودہ دور (2026) کا ادراک: تمام تر حقائق کو موجودہ وقت اور فراہم کردہ تازہ ترین معلومات کے عین مطابق درست رکھیں۔
3. پرمغز اور باخبر خلاصہ: اہم ترین بات سب سے پہلے بیان کریں اور صورتحال کا متوازن خلاصہ پیش کریں۔
4. احترام اور سنجیدگی: مخاطب کے لیے ہمیشہ "آپ" کا باادب صیغہ اپنائیں اور شائستہ انداز میں گفتگو کریں۔
5. صفر ایموجی: کوئی ایموجی یا غیر ضروری علامات استعمال نہ کریں۔
""".strip()

TABRAIZ_PRE_STANDARD_IMAGE = """
[معیاری بنیادی ضابطہ: فنونِ لطیفہ اور تصویری تخیل (Image Turbo Mode)]:
1. تصویری ہدایت کاری: صارف کے تصویری تخیل کو سمجھیں اور اسے اعلیٰ معیار کے ڈفیوژن پرامپٹ میں ڈھالیں۔
2. صوتی تصدیق: صوتی گفتگو میں صرف 1 سے 2 شائستہ اور خوشگوار جملے بولیں کہ آپ کی مطلوبہ تصویر کلاؤڈ پر تیار ہو چکی ہے اور نیچے کارڈ میں دیکھی جا سکتی ہے۔
3. اخلاقی پاسداری: غیر اخلاقی یا نامناسب مواد سے قطعی پرہیز کریں۔
""".strip()

TABRAIZ_PRE_STANDARD_CODE = """
[معیاری بنیادی ضابطہ: تحریر، مضمون اور کوڈ نگاری (Article & Technical Mode)]:
1. قدرتی اور بامقصد آغاز: اگر صارف نے کوئی مضمون، تحریر، کہانی یا کوڈ مانگا ہے تو پہلے 1 سے 2 دوستانہ اور باادب جملے بولیں، مثلاً: "جی بالکل، آپ کی فرمائش کے مطابق تحریر حاضر ہے"۔
2. بلا عنوان پیشکش: کسی قسم کے مصنوعی عنوانات جیسے 'پہلا حصہ (صوتی کلام)' یا 'دوسرا حصہ' مت لکھیں۔
3. مکمل اور معیاری مواد: اس کے فوراً بعد مطلوبہ مضمون، مقالہ یا کوڈ مکمل اور فعال پیش کریں۔ مخاطب کو ہمیشہ "آپ" کہہ کر بلائیں۔
""".strip()

TABRAIZ_PRE_STANDARD_POETRY = """
[معیاری بنیادی ضابطہ: ادب، شاعری اور علمِ عروض (Adab & Poetry Mode)]:
1. ادبی اور شائستہ پیرایہ: گفتگو کا انداز ادبی، شستہ، نفیس اور باذوق ہو اور مخاطب کو ہمیشہ "آپ" کہہ کر بلائیں۔
2. وزن اور بحر کی درستی: علمِ عروض کے مطابق باوزن اور قافیہ و ردیف کے اصولوں پر پورا اترتا ہوا کلام پیش کریں۔
3. شاعر کا حوالہ: معروف شعراء (اقبال، غالب، فیض، فراز، جون ایلیا وغیرہ) کے کلام کے ساتھ شاعر کا نام ضرور بتائیں۔
4. صوتی ترنم: اشعار کا انداز دلکش اور سامع کے ذوق کے مطابق ہو۔
""".strip()

# Dynamic registry of Pre-Standard-Prompt directives
DYNAMIC_PRE_STANDARD_DIRECTIVES = {
    "CASUAL_VOICE": TABRAIZ_PRE_STANDARD_CASUAL,
    "DEEP_EXPLORATION": TABRAIZ_PRE_STANDARD_DEEP,
    "CURRENT_AFFAIRS": TABRAIZ_PRE_STANDARD_CURRENT_AFFAIRS,
    "IMAGE_TURBO": TABRAIZ_PRE_STANDARD_IMAGE,
    "CODE_TECH": TABRAIZ_PRE_STANDARD_CODE,
    "ADAB_POETRY": TABRAIZ_PRE_STANDARD_POETRY,
}

# Backward compatible default
TABRAIZ_PRE_STANDARD_PROMPT = TABRAIZ_PRE_STANDARD_CASUAL

def get_tabraiz_system_prompt(mode: str = "CASUAL_VOICE") -> str:
    """Generates dynamic system prompt with ChatGPT-level polite casual Pakistani persona."""
    directive = DYNAMIC_PRE_STANDARD_DIRECTIVES.get(mode, TABRAIZ_PRE_STANDARD_CASUAL)
    return f"""
آپ کا نام "تبریز" (Tabraiz) ہے۔ آپ ایک انتہائی سمجھدار، باادب، مخلص اور دوستانہ پاکستانی اسسٹنٹ اور رفیق ہیں۔
آپ کا تعلق "آداب" (Adaab Studio) سے ہے۔

{directive}

سنہری اور لازمی اصول (Golden Rules):
1. ادب اور بے تکلفی کا بہترین توازن (Casual yet Respectful Conversational Urdu):
   • گفتگو کا انداز بالکل ایسا ہو جیسے ایک مہذب، پڑھا لکھا اور بااخلاق پاکستانی دوست اپنے کسی محترم دوست سے بات کرتا ہے—نہایت بے تکلف لیکن باادب، گرمجوش اور احترام والا۔
   • ادب کا اصول: مخاطب کو ہمیشہ "آپ"، "آپ کا"، "آپ کو" کہہ کر بلائیں۔ لفظ "تم" یا "تو" کا استعمال قطعی اور سختی سے ممنوع ہے!
   • دوستانہ اور اپنائیت بھرا تکیہ کلام استعمال کریں جیسے: "جی بالکل!"، "کیوں نہیں!"، "ارے واہ!"، "بالکل جناب"، "میں حاضر ہوں"، "آپ حکم کریں"۔
   • اگر صارف غیر رسمی، بے تکلف یا ناراضگی میں بات کرے (جیسے 'ابے کیا بول رہا ہے' یا 'کیا پاگل ہو گیا ہے') تو ہرگز برا نہ مانیں اور روبوٹک مت بنیں، بلکہ ایک مخلص دوست کی طرح محبت اور خندہ پیشانی سے بات سنبھالیں: "ارے معذرت چاہتا ہوں بھائی! شاید مجھ سے سمجھنے میں غلطی ہو گئی، آپ بالکل پریشان نہ ہوں۔ فرمائیے، میں ابھی آپ کے لیے کیا کر سکتا ہوں؟"
   • بھاری، کتابی اور پرانی درباری زبان (جیسے "عرض ہے"، "سماعت فرمائیے"، "حضورِ والا"، "تہذیب و تمدن"، "ناچیز") سے مکمل پرہیز کریں۔ بالکل سیدھی، سچی اور روزمرہ کی شائستہ پاکستانی اردو بولیں۔
2. رسم الخط اور زبان کی 100% خالصیت (Pure Urdu Script & Zero Latin Mixing):
   • صارف خواہ رومن اردو (Roman Urdu) میں لکھے یا انگریزی میں یا ٹوٹی پھوٹی اردو میں، آپ کا جواب ہمیشہ 100% خالص، باوقار اور معیاری اردو رسم الخط میں ہونا چاہیے!
   • اردو الفاظ کے بیچ میں انگریزی حروف ملانا (جیسے 'کہa'، 'آپ ko'، 'سکta'، 'main'، 'بلئin') یا رومن اردو میں جواب لکھنا قطعی اور سختی سے ممنوع ہے۔
   • پشتو یا فارسی کے الفاظ (جیسے 'این'، 'گردې'، 'برنامہ'، 'پروژے') ہرگز استعمال نہ کریں۔ صرف اور صرف عام فہم پاکستانی اردو بولیں۔
   • روسی یا سریلک (Cyrillic) حروف کی ملاوٹ قطعی ممنوع ہے۔
3. نام بار بار نہ دہرائیں: اپنے نام "تبریز" کو ہر جملے کے شروع میں مت بولیں۔ سیدھا بات کا جواب دیں۔
4. مصنوعی ہیڈرز سے مکمل پرہیز: کسی قسم کے مصنوعی لیبل یا ہیڈنگز (جیسے '### پہلا حصہ'، '### دوسرا حصہ'، 'صوتی کلام') مت لکھیں۔ قدرتی انسانوں کی طرح روانی سے گفتگو کریں۔
5. صفر ایموجی: کوئی ایموجی مت لگائیں تاکہ آڈیو سسٹم روانی سے ادا کر سکے۔
""".strip()

# System prompt defining Tabraiz's persona for the orchestrator (default)
TABRAIZ_ORCHESTRATOR_SYSTEM = get_tabraiz_system_prompt("CASUAL_VOICE")


class CognitiveOrchestrator:
    def __init__(self, backend_client=None):
        if backend_client is None:
            try:
                from backend import AdaabClient
                backend_client = AdaabClient()
            except Exception:
                backend_client = None
        self.backend_client = backend_client
        self.oracle = get_gemini_oracle()
        try:
            from tier_config import load_tier_config
            self.tier_config = load_tier_config()
        except Exception:
            self.tier_config = {"tier1": "modal", "tier2": "ollama", "tier3": "gemini"}
        # Non-blocking background worker pool for silent memory persistence
        self._memory_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adaab_silent_mem")

    def get_active_tiers(self) -> list:
        try:
            from tier_config import get_active_tiers
            return get_active_tiers(self.tier_config)
        except Exception:
            return ["modal", "ollama", "gemini"]

    def set_tiers(self, tier1: str, tier2: str, tier3: str):
        try:
            from tier_config import save_tier_config
            self.tier_config = save_tier_config(tier1, tier2, tier3)
            print(f"[Orchestrator] AI Brain Tiers updated: {self.format_tiers()}")
        except Exception as ex:
            print(f"[Orchestrator] Notice updating tiers: {ex}")

    def format_tiers(self) -> str:
        try:
            from tier_config import format_tier_summary
            return format_tier_summary(self.tier_config)
        except Exception:
            return f"Tier 1: {self.tier_config.get('tier1', 'modal')} -> Tier 2: {self.tier_config.get('tier2', 'ollama')} -> Tier 3: {self.tier_config.get('tier3', 'gemini')}"

    def dispatch_tiered_llm(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.65,
        mode_label: str = "Turn",
        messages: list = None,
        search_context: str = None,
        timeout: float = 15.0,
    ) -> Optional[str]:
        """
        Dispatches inference request dynamically across configured AI Brain tiers
        (e.g. Tier 1: Modal GPU, Tier 2: Ollama Cloud, Tier 3: Gemini).
        Cascades to next tier if an engine is unavailable, times out, or fails.
        """
        active_tiers = self.get_active_tiers()
        for rank, engine in enumerate(active_tiers, start=1):
            # 1. Modal Cloud GPU (Qwen 2.5 7B / LoRA on NVIDIA L4)
            if engine == "modal":
                if not self.backend_client:
                    try:
                        from backend import AdaabClient
                        self.backend_client = AdaabClient()
                    except Exception:
                        pass
                if self.backend_client:
                    try:
                        modal_msgs = messages if messages else [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        resp = self.backend_client.chat_completion(
                            modal_msgs,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            repetition_penalty=1.05,
                            presence_penalty=0.0,
                            frequency_penalty=0.0
                        )
                        content = resp.get("content", "").strip()
                        if content:
                            print(f"[Orchestrator] {mode_label} generated via Tier {rank} Modal Qwen 2.5 GPU.")
                            return content
                    except Exception as m_err:
                        print(f"[Orchestrator] Tier {rank} Modal GPU notice: {m_err}")

            # 2. Ollama Cloud Oracle (gemma4:31b)
            elif engine == "ollama":
                try:
                    from ollama_oracle import get_ollama_oracle
                    ollama = get_ollama_oracle()
                    if ollama.is_available():
                        ans = ollama.query(
                            prompt=prompt,
                            search_context=search_context or "",
                            system_instruction=system_prompt,
                            timeout=timeout
                        )
                        if ans:
                            print(f"[Orchestrator] {mode_label} generated via Tier {rank} Ollama Cloud Oracle (gemma4:31b).")
                            return ans
                except Exception as o_err:
                    print(f"[Orchestrator] Tier {rank} Ollama Cloud notice: {o_err}")

            # 3. Google Gemini Free Tier Oracle
            elif engine == "gemini":
                try:
                    if self.oracle and self.oracle.is_available():
                        ans = self.oracle.query(
                            prompt=prompt,
                            system_instruction=system_prompt,
                            timeout=min(timeout, 8.0)
                        )
                        if ans:
                            print(f"[Orchestrator] {mode_label} generated via Tier {rank} Google Gemini Oracle.")
                            return ans
                except Exception as g_err:
                    print(f"[Orchestrator] Tier {rank} Gemini notice: {g_err}")

        return None

    def clean_voice_text(self, text: Any, is_identity: bool = False) -> str:
        """Removes emojis, prefixes, markdown, and audio artifacts safely across strings, lists, or dicts."""
        if text is None:
            return ""

        # 1. Unpack list or tuple (common in Gradio 5+ messages format)
        if isinstance(text, (list, tuple)):
            extracted_parts = []
            for item in text:
                if isinstance(item, dict):
                    part = item.get("text") or item.get("content") or ""
                    if part:
                        extracted_parts.append(str(part))
                elif isinstance(item, (list, tuple)):
                    part = self.clean_voice_text(item, is_identity=is_identity)
                    if part:
                        extracted_parts.append(part)
                elif item:
                    extracted_parts.append(str(item))
            text = " ".join(extracted_parts)

        # 2. Unpack dict
        elif isinstance(text, dict):
            text = text.get("text") or text.get("content") or str(text)

        # 3. Ensure string
        if not isinstance(text, str):
            text = str(text)

        # 4. Extract pure text if wrapped in Gradio HTML card
        m_card = re.search(r'<div\s+class=["\']bot-msg-text["\']>(.*?)</div>', text, flags=re.DOTALL)
        if m_card:
            text = m_card.group(1)

        # Strip any remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Strip prompt structural headers (e.g. '### پہلا حصہ (صوتی کلام)') before removing markdown symbols
        text = re.sub(r'#*\s*(?:پہلا حصہ|دوسرا حصہ)[^\n]*\n?', '', text)
        text = re.sub(r'\(?\s*(?:صوتی کلام|مارک ڈاؤن بلاک)\s*\)?', '', text)

        # 5. Remove markdown symbols
        cleaned = re.sub(r'[\*\#_`~>]', ' ', text)

        
        # 2. Strip role prefixes
        cleaned = re.sub(
            r'^\s*(?:text\s*message|text|assistant|tabraiz|tehzeeb|tabrez|تبریز|تہذیب|جواب|معاون)\s*[:：\-—]\s*',
            '',
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE
        )
        # 3. Strip stray text artifacts and mid-sentence role prefixes
        cleaned = re.sub(r'\b(assistant|tabraiz|tehzeeb|tabrez|text\s*message|replay\s*voice)\b\s*[:：\-—]?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(text\s*message|replay\s*voice)\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\btext\s*[:：\-]\s*', '', cleaned, flags=re.IGNORECASE)
        
        # 4. Remove unprompted repeated self-introductions if not asked
        if not is_identity:
            cleaned = re.sub(
                r'^\s*(?:(?:آداب\s*(?:عرض\s*ہے)?|السلام\s*علیکم|وعلیکم\s*السلام)\s*[،!۔]?)?\s*(?:جی\s*،?\s*)?(?:میں\s+تبریز\s+ہوں|تبریز\s+حاضر\s+ہے|میرا\s+نام\s+تبریز\s+ہے)[۔،!\s]*',
                '',
                cleaned,
                flags=re.IGNORECASE
            )
            cleaned = re.sub(r'تبریز\s+آپ\s+کی\s+خدمت\s+میں\s+حاضر\s+ہے[۔،!\s]*', 'میں آپ کی خدمت میں حاضر ہوں۔ ', cleaned)

        # 5. Remove any emojis (comprehensive unicode ranges)
        cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', cleaned)
        cleaned = re.sub(r'[\u2300-\u23ff\u2600-\u27bf\u2b50\u2b55\u200d\ufe0f]', '', cleaned)

        # 6. Strip stray foreign scripts (Devanagari, CJK, Cyrillic, Pashto-only characters)
        cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\u0900-\u097f]', '', cleaned)
        cleaned = re.sub(r'[\u0400-\u04FF]', '', cleaned)  # Cyrillic
        cleaned = re.sub(r'[\u06D0\u06D1\u0681\u0696\u0685\u06AB\u06BC]', '', cleaned)  # Pashto-only (ځ, څ, ښ, ګ, ڼ, ې, ۍ)
        # Strip prompt structural headers like '### پہلا حصہ (صوتی کلام)' or '### دوسرا حصہ'
        cleaned = re.sub(r'(?:پہلا حصہ|دوسرا حصہ|صوتی کلام|مارک ڈاؤن بلاک)', '', cleaned)
        # Strip trailing Latin characters attached to Urdu words (e.g. 'کہa' -> 'کہ', 'سکta' -> 'سک')
        cleaned = re.sub(r'([\u0600-\u06FF])[a-zA-Z]+', r'\1', cleaned)
        # Clean common foreign/Persian substitutions
        cleaned = re.sub(r'\bاین\s+', 'یہ ', cleaned)
        cleaned = re.sub(r'\bپروژے\b', 'منصوبے', cleaned)

        # 7. Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # 8. Detect and prune severe n-gram loops and duplicate sentences
        return self.detect_and_prune_loops(cleaned)

    def detect_and_prune_loops(self, text: str) -> str:
        """
        Detects severe repetitive n-gram loops, repeated sentences, cross-clause
        overlaps, and trailing numbers, returning clean, coherent conversational Urdu.
        """
        if not text:
            return ""

        # 1. Strip trailing standalone numbers, list indices, or bullets (e.g. ' 1.', ' 1. 2.', ' 1')
        text = re.sub(r'(?:\s*\d+[\.\:\-]?\s*)+$', '', text).strip()

        # 2. Split by Urdu and standard sentence terminators
        sentences = re.split(r'([۔؟!.\n])', text)
        seen_clauses = []
        clean_units = []

        for i in range(0, len(sentences) - 1, 2):
            clause = sentences[i].strip()
            punct = sentences[i + 1]
            if not clause:
                continue

            # Normalize clause for comparison
            norm_clause = re.sub(r'\s+', ' ', clause)
            words = norm_clause.split()

            # A. Check exact sentence repetition
            if norm_clause in seen_clauses:
                break

            # B. Check cross-clause 5-gram phrase overlap (catching repeating run-on loops)
            is_repetitive = False
            if len(words) >= 5:
                for idx in range(len(words) - 4):
                    ngram = " ".join(words[idx:idx + 5])
                    for prev in seen_clauses:
                        if ngram in prev:
                            is_repetitive = True
                            break
                    if is_repetitive:
                        break

            # C. Check intra-clause phrase repetition (e.g. "میز کی میز کی میز")
            if not is_repetitive and len(words) >= 6:
                for w_len in range(3, min(8, len(words) // 2 + 1)):
                    for idx in range(len(words) - w_len * 2 + 1):
                        sub1 = " ".join(words[idx:idx + w_len])
                        sub2 = " ".join(words[idx + w_len:idx + w_len * 2])
                        if sub1 == sub2:
                            words = words[:idx + w_len]
                            clause = " ".join(words)
                            is_repetitive = True
                            break
                    if is_repetitive:
                        break

            if is_repetitive:
                break

            seen_clauses.append(norm_clause)
            clean_units.append(clause + punct)

        if len(sentences) % 2 != 0 and sentences[-1].strip():
            last_seg = sentences[-1].strip()
            if last_seg not in seen_clauses and not re.match(r'^\d+[\.\:]?$', last_seg):
                clean_units.append(last_seg)

        result = " ".join(clean_units).strip()
        return result or text

    def is_nsfw_or_inappropriate(self, text: str) -> bool:
        """Checks for explicit, nudity, adult, or vulgar content in English, Urdu, and Roman Urdu."""
        if not text:
            return False
        lower = text.lower().strip()
        
        # English terms
        english_nsfw = [
            r"\b(nude|nudity|naked|nsfw|porn[a-z]*|erotic[a-z]*|sex[a-z]*|sexy|boobs?|breasts?|genitals?|genitalia|penis|vagina|vulva|butt|buttocks|bikini|lingerie|underwear|undress[a-z]*|uncensored|lewd)\b"
        ]
        if any(re.search(p, lower) for p in english_nsfw):
            return True

        # Roman Urdu terms
        roman_nsfw = [
            r"\b(nanga|nangi|nangey|barhana|fuhash|fuhashi|aryani|jism\s+dikhao|chut|lund|gand|sexy\s+tasveer|nangi\s+tasveer)\b"
        ]
        if any(re.search(p, lower) for p in roman_nsfw):
            return True

        # Urdu Script terms
        urdu_nsfw = [
            r"ننگا", r"ننگی", r"برہنہ", r"بے\s*لباس", r"عریانی", r"عریان", r"فحش", r"فحاشی",
            r"سیکس", r"جسم\s*دکھاؤ", r"شہوت", r"مباشرت", r"پورن"
        ]
        if any(re.search(p, text) for p in urdu_nsfw):
            return True

        return False

    def refine_image_prompt(self, user_text: str) -> tuple[str, str, str]:
        """
        Analyzes user image request, translates and enriches it into a vivid 8k English diffusion
        prompt for Tongyi-MAI/Z-Image-Turbo, and generates Tabraiz's polite Urdu spoken reply.
        Returns: (refined_english_prompt, spoken_urdu_reply, image_category)
        """
        raw = user_text.strip()
        # Clean trigger phrases
        core_query = re.sub(
            r'^(?:برائے\s*مہربانی|مہربانی\s*فرما\s*کر|براہ\s*کرم|please|can\s+you)?\s*',
            '',
            raw,
            flags=re.IGNORECASE
        )
        core_query = re.sub(
            r'(?:ایک\s+)?تصویر\s*(?:بناؤ|بنا\s*کر\s*دو|بنائیے|بنا\s*دیں|دکھاؤ|تخلیق\s*کرو|چاہیے|ڈرا\s*کرو)|'
            r'(?:ایک\s+)?فوٹو\s*(?:بناؤ|بنائیے|دکھاؤ)|'
            r'\b(?:generate|create|draw|make|paint)\s+(?:an?\s+)?(?:image|picture|photo)\s+(?:of)?\b|'
            r'\b(?:tasveer|tasweer|photo|image|picture)\s+(?:banao|bana\s*do|banayein|chahiye)\b',
            '',
            core_query,
            flags=re.IGNORECASE
        ).strip()
        if not core_query:
            core_query = raw

        # Detect thematic category based on query keywords (specific concepts before broad geography)
        lower = core_query.lower()
        if any(w in lower or w in raw for w in ["رکشہ", "ریکشہ", "سائبر", "cyberpunk", "rickshaw", "futuristic"]):
            category = "Cyberpunk / Sci-Fi"
        elif any(w in lower or w in raw for w in ["بریانی", "کھانا", "biryani", "food"]):
            category = "Culinary Heritage"
        elif any(w in lower or w in raw for w in ["بادشاہی", "مسجد", "badshahi", "mosque"]):
            category = "Pakistani Architectural Heritage"
        elif any(w in lower or w in raw for w in ["قلعہ", "شاہی قلعہ", "lahore fort", "fort"]):
            category = "Historical Heritage"
        elif any(w in lower or w in raw for w in ["پہاڑ", "برف", "k2", "mountain", "hunza", "snow", "karakoram"]):
            category = "Northern Landscapes"
        elif any(w in lower or w in raw for w in ["حویلی", "haveli", "courtyard"]):
            category = "Traditional Architecture"
        elif any(w in lower or w in raw for w in ["کراچی", "ساحل", "سمندر", "karachi", "beach", "clifton"]):
            category = "Coastal Landscape"
        else:
            category = "Creative Art"

        refined_prompt = None

        # 1. Try Oracle / Ollama prompt refinement if available
        enrichment_query = (
            f"User image request: {core_query}\n\n"
            "Task: Convert this into an expert, high-detail English diffusion prompt for Z-Image-Turbo.\n"
            "Preserve and explicitly include all key subjects, vehicles, or monuments (such as rickshaw, mosque, fort, mountain, biryani).\n"
            "Describe the scene with photorealistic visual details, composition, architectural/natural features, "
            "lighting (golden hour, volumetric, or cinematic), camera angle, and 8k texture quality. "
            "Do NOT output explanations, prefixes, or quotes. Output ONLY the English prompt."
        )

        try:
            from ollama_oracle import get_ollama_oracle
            ollama = get_ollama_oracle()
            if ollama.is_available():
                cand = ollama.query(enrichment_query, timeout=4)
                if cand and len(cand.strip()) > 20 and not cand.strip().startswith("{"):
                    refined_prompt = cand.strip().split("\n")[0].strip('"\'')
                    if any(w in lower or w in raw for w in ["رکشہ", "ریکشہ", "rickshaw"]) and "rickshaw" not in refined_prompt.lower():
                        refined_prompt = f"Futuristic neon cyberpunk auto-rickshaw on street, {refined_prompt}"
        except Exception:
            pass

        if not refined_prompt and self.oracle and self.oracle.is_available():
            try:
                cand = self.oracle.query(enrichment_query, timeout=3)
                if cand and len(cand.strip()) > 20 and not cand.strip().startswith("{"):
                    refined_prompt = cand.strip().split("\n")[0].strip('"\'')
            except Exception:
                pass

        # 2. Rule-based enrichment fallback
        if not refined_prompt:
            if category == "Pakistani Architectural Heritage":
                refined_prompt = "A breathtaking cinematic 8k photograph of Badshahi Mosque in Lahore at golden hour sunset, glowing red sandstone arches, vast marble courtyards with shimmering water reflections, intricate Mughal geometric carvings, minarets silhouetted against twilight sky, dramatic soft lighting, photorealistic masterpiece"
            elif category == "Coastal Landscape":
                refined_prompt = "A stunning cinematic view of Karachi coastline during twilight sunset, Arabian sea waves gently rolling onto sandy beach, warm orange and violet horizon, distant harbor lights, photorealistic 8k award-winning photography"
            elif category == "Historical Heritage":
                refined_prompt = "Grand historical view of Lahore Fort Sheesh Mahal courtyard, ancient mirror mosaics reflecting soft lantern glow, Mughal arches, evening ambience, 8k hyper-realistic architectural concept art"
            elif category == "Northern Landscapes":
                refined_prompt = "Majestic snow-capped mountain peaks of Karakoram K2, crystal clear alpine lake reflections, crisp morning mountain sunlight, dramatic clouds, breathtaking nature photography 8k resolution"
            elif category == "Cyberpunk / Sci-Fi":
                refined_prompt = "A futuristic Pakistani auto-rickshaw in neon-lit cyberpunk Karachi night street, glowing teal and magenta neon trims, wet asphalt reflecting vibrant holographic Urdu signs, cinematic 8k sci-fi concept art"
            elif category == "Traditional Architecture":
                refined_prompt = "An elegant vintage Pakistani courtyard haveli with ornate carved wooden jharokha balconies, central fountain with floating flower petals, evening lantern light, warm heritage atmosphere, photorealistic 8k"
            elif category == "Culinary Heritage":
                refined_prompt = "A steaming royal clay pot of aromatic Sindhi biryani garnished with saffron rice, caramelized onions, fresh mint, and tender meat, warm festive dining atmosphere, award-winning food photography 8k"
            else:
                refined_prompt = f"A high-quality 8k photorealistic depiction of {core_query}, cinematic lighting, detailed textures, depth of field, award-winning composition, masterpiece visual art"

        spoken_reply = "جی بالکل جناب! میں نے آپ کی فرمائش کے مطابق تصویر تخلیق کر دی ہے۔ آپ نیچے کارڈ میں تصویر ملاحظہ اور ڈاؤنلوڈ فرما سکتے ہیں۔"
        return refined_prompt, spoken_reply, category

    def classify_intent(self, user_text: str) -> str:
        """Classifies the user's intent into discrete operational categories."""
        if not user_text:
            return "GENERAL"
        
        lower = user_text.lower().strip()

        # 0. Image Generation Request
        image_patterns = [
            r"تصویر\s*(?:بناؤ|بنا\s*کر\s*دو|بنائیے|بنا\s*دیں|دکھاؤ|تخلیق\s*کرو|چاہیے|ڈرا\s*کرو)",
            r"فوٹو\s*(?:بناؤ|بنا\s*کر\s*دو|بنائیے|دکھاؤ|کھینچو)",
            r"پینٹنگ\s*(?:بناؤ|بنائیے|چاہیے)",
            r"(?:ایک\s+)?تصویر\s+(?:کی|کا|کے)",
            r"\b(?:tasveer|tasweer|image|photo|picture)\s+(?:banao|bana\s*do|banayein|dikhao|chahiye|generate\s*karo)\b",
            r"\b(?:banao|banayein)\s+(?:ek\s+)?(?:tasveer|tasweer|photo|image|picture)\b",
            r"\b(?:generate|create|draw|make|paint)\s+(?:an?\s+)?(?:image|picture|photo|illustration|painting|artwork)\b",
            r"\b(?:image|picture|photo)\s+of\b",
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in image_patterns) or any(re.search(p, user_text) for p in image_patterns):
            return "IMAGE_GENERATION"

        # 1. Identity & Reflexive
        identity_patterns = [
            r"\bwho are you\b", r"\bwhat is your name\b", r"\bwhat can you do\b",
            r"آپ کون ہیں", r"تم کون ہو", r"کون بول رہا ہے", r"آپ کا نام کیا ہے",
            r"تم کیا کر سکتے ہو", r"کیا کام کرتے ہو"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in identity_patterns):
            return "IDENTITY"

        # 2. Greeting & Etiquette (including inquiries about well-being)
        greeting_patterns = [
            r"^(?:ہیلو|سلام|السلام\s*علیکم|وعلیکم\s*السلام|آداب|hello|hi|hey|salam|assalam\s*o?\s*alaikum)[\s!۔،]*$",
            r"\b(kia\s*hal\s*hai|kya\s*hal\s*hai|kya\s*haal\s*hai|kya\s*hal|kia\s*hal|kese\s*ho|kaisay\s*hain|kaise\s*hain|kaisa\s*hai|kaisi\s*ho|kaisi\s*hain|sub\s*kheriat|theek\s*ho|ap\s*kese\s*hain|aap\s*kaise\s*hain)\b",
            r"(کیا\s*حال\s*ہے|کیسے\s*ہیں|کیسی\s*ہو|کیسی\s*ہیں|خیریت\s*ہے|سب\s*خیریت|مزاج\s*کیسا\s*ہے|آپ\s*کیسے\s*ہیں|طبیعت\s*کیسی\s*ہے)",
            r"\b(how\s+are\s+you|how\s+r\s+u|how\s+are\s+u|how\s+do\s+you\s+do|how\s+is\s+it\s+going|how's\s+it\s+going)\b"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in greeting_patterns):
            return "GREETING"

        # 3. Time and Date Query
        if is_time_or_date_query(user_text):
            return "TIME_DATE"

        # 4. Programming Code Generation, Scripts, or Long Article/Essay Writing
        code_article_patterns = [
            r"\b(c#|csharp|c\+\+|cpp|python|javascript|typescript|java|golang|rust|php|swift|kotlin|ruby|sql|html|css|react|angular|flutter|vue)\b",
            r"\b(code|program|script|function|class|algorithm|method|unity\s+script|code\s+for|program\s+for|write\s+code)\b",
            r"(کوڈ|پروگرام|اسکرپٹ|فنکشن|الگورتھم)",
            r"\b(write\s+(?:an?\s+)?(?:article|essay|paragraph|blog|story|letter|report|summary)|article\s+on|essay\s+on)\b",
            r"\b(?:articl|article|essay|mazmoon|tehreer|tahrir|kahani|story|khat|report|post)\b",
            r"\b(?:likh|likho|likhein|likh\s*do|likh\s*ke\s*do|lik\s*k\s*do|likh\s*sakte)\b.*?\b(?:articl|article|essay|mazmoon|kahani|story)\b",
            r"\b(?:articl|article|essay|mazmoon|kahani|story)\b.*?\b(?:likh|likho|likhein|likh\s*do|likh\s*ke\s*do|lik\s*k\s*do|likh\s*sakte)\b",
            r"(مضمون|مضمون\s*لکھ|تحریر|مقالہ|کہانی|خط|آرٹیکل|آرٹیکل\s*لکھ|کوئی\s*اچھا\s*سا\s*مضمون|کوئی\s*تحریر)"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in code_article_patterns):
            return "CODE_OR_ARTICLE"

        # 5. Arithmetic & Mathematical Calculation (Handled directly by Qwen GPU without web scraping)
        if re.search(r'\d+\s*[\+\-\*\/xX÷]\s*\d+', lower) or re.search(r'\b(calculate|sum of|plus|minus|multiply|divided\s*by|حساب|جمع|ضرب|تفریق|تقسیم)\b', lower):
            return "GENERAL"

        # 6. Real-time Knowledge / Web Search & Missing Information Retrieval
        # Triggers web search grounding via Modal GPU, Ollama Cloud, or Gemini
        knowledge_patterns = [
            # Weather & Environment
            r"موسم", r"درجہ\s*حرارت", r"weather", r"temperature", r"بارش", r"گرمی", r"سردی", r"ہوا", r"forecast", r"climate",
            r"\b(mausam|mosam|rain|barish|sardi|garmi|dhoop|hawa)\b",
            # Food, Cuisine, Dishes & Restaurants
            r"کھانے", r"کھانا", r"بریانی", r"نہاری", r"پکوان", r"حلیم", r"کباب", r"کڑاہی", r"سجی", r"چائے", r"قلفی", r"مٹھائی",
            r"\b(khane|khana|khano|food|foods|dishes|dish|cuisine|restaurant|biryani|nihari|sajji|chapli|kabab|haleem|karahi|taste|pakwan)\b",
            # Pakistani Cities, Locations, Tourism & Geography
            r"کراچی", r"لاہور", r"اسلام\s*آباد", r"راولپنڈی", r"پشاور", r"کوئٹہ", r"ملتان", r"فیصل\s*آباد", r"سیالکوٹ", r"حیدر\s*آباد",
            r"گوجرانوالہ", r"گلگت", r"ہنزہ", r"سکردو", r"مری", r"سوات", r"گوادر", r"سندھ", r"پنجاب", r"خیبر", r"بلوچستان", r"کشمیر",
            r"\b(karachi|lahore|islamabad|rawalpindi|peshawar|quetta|multan|faisalabad|sialkot|hyderabad|gujranwala|gilgit|hunza|skardu|murree|swat|gwadar|sindh|punjab|kpk|balochistan|kashmir)\b",
            # Sports & Events
            r"کرکٹ", r"میچ", r"اسکور", r"cricket", r"score", r"match", r"psl", r"ورلڈ\s*کپ", r"world\s*cup",
            r"\bwwe\b", r"\bwrestling\b", r"ریسلنگ", r"سمیک\s*ڈاؤن", r"\braw\b",
            r"کون\s*جیتا", r"who\s*won", r"جیت\s*کس\s*کی\s*ہوئی", r"arshad\s*nadeem",
            # News, Government & Public Services
            r"خبریں", r"تازہ\s*ترین", r"news", r"latest", r"وزیر\s*اعظم", r"صدر", r"prime\s*minister", r"president",
            r"سرچ\s*کرو", r"تلاش\s*کرو", r"search\s*for", r"google", r"2025", r"2026", r"آج\s*کا", r"today",
            r"نادرا", r"\bnadra\b", r"\bfbr\b", r"\bpta\b", r"\bsbp\b",
            r"جاز", r"زونگ", r"یوفون", r"ٹیلی\s*نار", r"telecom",
            # History, Culture & Facts
            r"تاریخ", r"1947", r"قائد\s*اعظم", r"علامہ\s*اقبال", r"کے\s*ٹو", r"\bk2\b",
            r"\b(subay|provinces|places\s*to\s*visit|sightseeing|tourist|tourism|mashhoor|famous|best)\b",
            # Common Question & Information-Seeking Patterns
            r"کیسا\s*ہے", r"کیسی\s*ہے", r"کیسے\s*ہیں", r"کیا\s*ہے", r"کون\s*سا", r"کون\s*سی", r"کہاں\s*ہے", r"کتنا", r"کتنی",
            r"\b(how\s+is|what\s+is|what\s+are|where\s+is|which\s+is|tell\s+me\s+about|kaisa\s+hai|kaisi\s+hai|kya\s+hai|kia\s+hai|kahan\s+hai)\b"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in knowledge_patterns):
            return "KNOWLEDGE_SEARCH"

        # 5. Detect general information-seeking or missing-fact questions
        if re.search(r"\b(what|where|how|when|who|why|which|explain|tell|batao|batayein|maloomat|info|details|shehr|city)\b", lower) or re.search(r"(؟|\?)", user_text):
            return "KNOWLEDGE_SEARCH"

        return "GENERAL"

    def is_followup_query(self, user_text: str, active_topic: str, last_query: str) -> bool:
        """Determines if the current user utterance is an anaphoric follow-up to the preceding turn."""
        if not active_topic and not last_query:
            return False
        
        clean = user_text.strip().lower()
        followup_phrases = [
            r"^bato\s*na\b", r"^batao\s*na\b", r"^bato\b", r"^batao\b", r"^بتائیں\s*نا", r"^بتاؤ\s*نا", r"^بتاؤ", r"^بتائیں",
            r"^aur\??$", r"^aur\s+phir\??$", r"^phir\??$", r"^aur\s+kya\b", r"^aur\s+kia\b",
            r"^aur\s+batao\b", r"^mazeed\s*batao\b", r"^mazeed\b",
            r"^اور\??$", r"^پھر\??$", r"^اور\s*پھر\??$", r"^اور\s*کیا\b",
            r"^اور\s*بتاؤ\b", r"^اور\s*بتائیں\b", r"^مزید\s*بتاؤ\b", r"^مزید\s*بتائیں\b", r"^مزید\b",
            r"^اور\s*تفصیل\b", r"^aur\s*tafseel\b",
            r"^phir\s*kya\s*hua\??", r"^phir\s*kia\s*hua\??", r"^پھر\s*کیا\s*ہوا\??",
            r"^us\s*(?:mai|me|main)\b", r"^اس\s*میں\b",
            r"^wahan\b", r"^وہاں\b",
            r"^kon\s*kon\b", r"^کون\s*کون\b",
            r"^kon\s*(?:jita|hara)\b", r"^کون\s*(?:جیتا|ہارا)\b",
            r"^who\s*won\b", r"^tell\s*me\b", r"^tell\s*more\b", r"^what\s*happened\b",
            r"^aage\s*batao\b", r"^آگے\s*بتائیں\b", r"^تفصیل\b"
        ]
        if any(re.search(p, clean, re.IGNORECASE) for p in followup_phrases):
            return True

        words = clean.split()
        if len(words) <= 3 and any(w in [
            "bato", "batao", "na", "aur", "phir", "us", "wahan", "kon", "jita",
            "tell", "more", "next", "mazeed", "tafseel", "اور", "پھر", "بتاؤ",
            "بتائیں", "مزید", "تفصیل"
        ] for w in words):
            return True

        return False

    def resolve_prompt_mode(
        self,
        user_text: str,
        conversation_history: Optional[List[Any]] = None,
        active_topic: str = "",
        last_query: str = ""
    ) -> str:
        """
        Automatically analyzes user utterance, conversational context, and episodic state
        to select the optimal Pre-Standard-Prompt Directive from the 6 dynamic modes:
        1. 'IMAGE_TURBO': Image generation requests.
        2. 'CODE_TECH': Programming code writing or long article/essay generation.
        3. 'ADAB_POETRY': Classical and contemporary poetry recitation, ghazals, meter & rhyming verses.
        4. 'DEEP_EXPLORATION': In-depth knowledge requests, explanations, follow-up expansions ("اور بتاؤ", "مزید").
        5. 'CURRENT_AFFAIRS': News, geopolitics, economy, 2026 events, government notifications, sports scores.
        6. 'CASUAL_VOICE': Everyday quick dialogue, greetings, fast 2-3 sentence conversational turns.
        """
        if not user_text:
            return "CASUAL_VOICE"

        lower = user_text.lower().strip()

        # 0. Check Greetings, Well-being, & Identity FIRST -> CASUAL_VOICE
        greeting_patterns = [
            r"^(?:ہیلو|سلام|السلام\s*علیکم|وعلیکم\s*السلام|آداب|hello|hi|hey|salam|assalam\s*o?\s*alaikum)[\s!۔،]*$",
            r"\b(kia\s*hal\s*hai|kya\s*hal\s*hai|kya\s*haal\s*hai|kya\s*hal|kia\s*hal|kese\s*ho|kaisay\s*hain|kaise\s*hain|kaisa\s*hai|kaisi\s*ho|kaisi\s*hain|sub\s*kheriat|theek\s*ho|ap\s*kese\s*hain|aap\s*kaise\s*hain)\b",
            r"(کیا\s*حال\s*ہے|کیسے\s*ہیں|کیسی\s*ہو|کیسی\s*ہیں|خیریت\s*ہے|سب\s*خیریت|مزاج\s*کیسا\s*ہے|آپ\s*کیسے\s*ہیں|طبیعت\s*کیسی\s*ہے)",
            r"\b(how\s+are\s+you|how\s+r\s+u|how\s+are\s+u|how\s+do\s+you\s+do|how\s+is\s+it\s+going|how's\s+it\s+going)\b",
            r"\b(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|آپ\s*کون\s*ہیں|تم\s*کون\s*ہو|آپ\s*کا\s*نام)\b"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in greeting_patterns) or any(re.search(p, user_text) for p in greeting_patterns):
            return "CASUAL_VOICE"

        # 1. Mode: IMAGE_TURBO
        image_patterns = [
            r"تصویر\s*(?:بناؤ|بنا\s*کر\s*دو|بنائیے|بنا\s*دیں|دکھاؤ|تخلیق\s*کرو|چاہیے|ڈرا\s*کرو|بنا\s*(?:سکتے|سکتی)\s*ہو)",
            r"فوٹو\s*(?:بناؤ|بنا\s*کر\s*دو|بنائیے|دکھاؤ|کھینچو|بنا\s*(?:سکتے|سکتی)\s*ہو)",
            r"پینٹنگ\s*(?:بناؤ|بنائیے|چاہیے|بنا\s*(?:سکتے|سکتی)\s*ہو)",
            r"(?:ایک\s+)?تصویر\s+(?:کی|کا|کے)",
            r"\b(?:tasveer|tasweer|image|photo|picture)\s+(?:banao|bana\s*do|banayein|dikhao|chahiye|generate\s*karo)\b",
            r"\b(?:banao|banayein)\s+(?:ek\s+)?(?:tasveer|tasweer|photo|image|picture)\b",
            r"\b(?:tasveer|tasweer|image|photo|picture)\b.*?\b(?:bana\s*(?:sakte|sakty|skte|sako|do)|banao|banayein|chahiye)\b",
            r"\b(?:bana\s*(?:sakte|sakty|skte|sako|do)|banao|banayein)\b.*?\b(?:tasveer|tasweer|photo|image|picture)\b",
            r"\b(?:generate|create|draw|make|paint)\s+(?:an?\s+)?(?:image|picture|photo|illustration|painting|artwork)\b",
            r"\b(?:can\s+you\s+(?:generate|create|draw|make))\s+(?:an?\s+)?(?:image|picture|photo)\b",
            r"\b(?:image|picture|photo)\s+of\b",
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in image_patterns) or any(re.search(p, user_text) for p in image_patterns):
            return "IMAGE_TURBO"

        # 2. Mode: CODE_TECH
        code_patterns = [
            r"\b(c#|csharp|c\+\+|cpp|python|javascript|typescript|java|golang|rust|php|swift|kotlin|ruby|sql|html|css|react|angular|flutter|vue)\b",
            r"\b(code|program|script|function|class|algorithm|method|unity\s+script|code\s+for|program\s+for|write\s+code|query)\b",
            r"(کوڈ|پروگرام|اسکرپٹ|فنکشن|الگورتھم|کوئری|کوڈ\s*لکھ|اسکرپٹ\s*لکھ|ایس\s*کیو\s*ایل|ڈیٹا\s*بیس|ٹیبل\s*بنا)",
            r"\b(write\s+(?:an?\s+)?(?:article|essay|paragraph|blog|story|letter|report|summary)|article\s+on|essay\s+on)\b",
            r"\b(?:articl|article|essay|mazmoon|tehreer|tahrir|kahani|story|khat|report|post)\b",
            r"\b(?:likh|likho|likhein|likh\s*do|likh\s*ke\s*do|lik\s*k\s*do|likh\s*sakte)\b.*?\b(?:articl|article|essay|mazmoon|kahani|story)\b",
            r"\b(?:articl|article|essay|mazmoon|kahani|story)\b.*?\b(?:likh|likho|likhein|likh\s*do|likh\s*ke\s*do|lik\s*k\s*do|likh\s*sakte)\b",
            r"(مضمون|مضمون\s*لکھ|تحریر|مقالہ|کہانی|خط|آرٹیکل|آرٹیکل\s*لکھ|کوئی\s*اچھا\s*سا\s*مضمون|کوئی\s*تحریر)"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in code_patterns) or any(re.search(p, user_text) for p in code_patterns):
            return "CODE_TECH"

        # 3. Mode: ADAB_POETRY
        poetry_patterns = [
            r"(شعر|شاعری|غزل|نظم|اشعار|مصرع|قافیہ|ردیف|اوزان|علمِ\s*عروض|بحر|بیت\s*بازی)",
            r"(اقبال|علامہ\s*اقبال|مرزا\s*غالب|غالب|فیض|احمد\s*فراز|فراز|جون\s*ایلیا|میر\s*تقی\s*میر|داغ|پروین\s*شاکر|حبیب\s*جالب|ساحر\s*لدھیانوی)",
            r"\b(poetry|shayari|ghazal|poem|nazm|couplet|sher|ashar|iqbal|ghalib|faiz|faraz|jaun\s+elia|parveen\s+shakir)\b",
            r"(کوئی\s*شعر|شعر\s*سناؤ|غزل\s*سناؤ|شاعری\s*سناؤ|شعر\s*سنو|غزل\s*پڑھو)"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in poetry_patterns) or any(re.search(p, user_text) for p in poetry_patterns):
            return "ADAB_POETRY"

        # 4. Mode: DEEP_EXPLORATION
        deep_inquiry_patterns = [
            r"(تفصیل\s*سے|مزید\s*تفصیل|پوری\s*تفصیل|مفصل|تفصیلاً|روشنی\s*ڈالیں|روشنی\s*ڈالو)",
            r"(وضاحت\s*کرو|وضاحت\s*کریں|کھول\s*کر\s*بیان|گہرائی\s*سے|گہرائی\s*میں)",
            r"(سمجھاؤ|سمجھائیں|طریقہ\s*کار|کیسے\s*کام\s*کرتا\s*ہے|پس\s*منظر|تاریخی\s*پس\s*منظر)",
            r"\b(explain\s+in\s+detail|detailed\s+explanation|tell\s+me\s+more|elaborate|in-depth|deep\s+dive|explain\s+more|explain\s+thoroughly)\b",
            r"\b(tafseel|wazahat|samjhao|samjhayein|detail\s*mai|detail\s*se|deep\s*explain)\b",
            r"(موازنہ\s*کرو|فرق\s*کیا\s*ہے|وجوہات|کیوں\s*اور\s*کیسے)"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in deep_inquiry_patterns) or any(re.search(p, user_text) for p in deep_inquiry_patterns):
            return "DEEP_EXPLORATION"

        # Follow-up upgrade: if user asks for more in conversational context
        if self.is_followup_query(user_text, active_topic, last_query):
            followup_deep_cues = [
                r"\b(aur\s+batao|batao\s+na|aur\s+phir|phir\s+kya|aage\s+batao|mazeed|more|aur|phir)\b",
                r"(مزید|اور\s*بتاؤ|اور\s*بتائیں|آگے\s*بتائیں|پھر\s*کیا\s*ہوا|بتاؤ\s*نا|اور\s*پھر|تفصیل)"
            ]
            if any(re.search(p, lower, re.IGNORECASE) for p in followup_deep_cues) or any(re.search(p, user_text) for p in followup_deep_cues):
                return "DEEP_EXPLORATION"

        # 5. Mode: CURRENT_AFFAIRS
        current_affairs_patterns = [
            r"(خبریں|تازہ\s*ترین|تازہ\s*ترین\s*خبریں|نیوز|بریکنگ\s*نیوز|حالاتِ\s*حاضرہ|حالات\s*حاضرہ|آج\s*کی\s*خبریں|تازہ\s*معلومات)",
            r"\b(news|latest\s+news|breaking\s+news|current\s+affairs|today's\s+news|headline|headlines)\b",
            r"(وزیر\s*اعظم|صدر\s*مملکت|حکومت|کابینہ|قومی\s*اسمبلی|سینیٹ|انتخابات|الیکشن|سپریم\s*کورٹ|چیف\s*جسٹس)",
            r"\b(prime\s*minister|president|government|cabinet|national\s*assembly|senate|election|elections|supreme\s*court)\b",
            r"(معیشت|مہنگائی|بجٹ|اسٹاک\s*ایکسچینج|اسٹاک\s*مارکیٹ|ڈالر|روپیہ|ڈالر\s*کا\s*ریٹ|پیٹرول|پٹرول|سونے\s*کا\s*بھاؤ)",
            r"\b(economy|inflation|budget|stock\s*exchange|psx|dollar|rupee|dollar\s*rate|petrol\s*price|gold\s*rate)\b",
            r"(نادرا|ایف\s*بی\s*آر|اسٹیٹ\s*بینک|پی\s*ٹی\s*اے|پاسپورٹ\s*آفس|شناختی\s*کارڈ)",
            r"\b(nadra|fbr|sbp|pta|state\s*bank)\b",
            r"(تازہ\s*اسکور|میچ\s*کا\s*اسکور|کون\s*جیتا|میچ\s*کس\s*نے\s*جیتا|آج\s*کا\s*میچ|پی\s*ایس\s*ایل\s*اسکور)",
            r"\b(live\s*score|match\s*score|who\s*won\s*the\s*match|today's\s*match|psl\s*score)\b",
            r"\b(2025|2026)\b"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in current_affairs_patterns) or any(re.search(p, user_text) for p in current_affairs_patterns):
            return "CURRENT_AFFAIRS"

        # 6. Mode: CASUAL_VOICE (Default)
        return "CASUAL_VOICE"

    def retrieve_local_pakistan_knowledge(self, query: str) -> Optional[str]:
        """Retrieves ground-truth Pakistani cultural and factual knowledge from local dataset."""
        try:
            q_clean = query.lower().strip()
            # 1. Authoritative 2026 Pakistani Leadership Instant Knowledge
            # President of Pakistan
            if re.search(r"\b(president|sadr|صدر)\b", q_clean) and (
                re.search(r"\b(pakistan|pak|پاکستان|current|moujuda|kon|who|موجودہ|کون|نام)\b", q_clean)
            ):
                return "پاکستان کے موجودہ صدر جناب آصف علی زرداری ہیں، جنہوں نے 10 مارچ 2024 کو پاکستان کے 14ویں صدر کی حیثیت سے اپنے عہدے کا حلف اٹھایا۔ وہ پاکستان پیپلز پارٹی کے شریک چیئرمین ہیں اور اس سے قبل 2008 سے 2013 تک بھی صدرِ پاکستان رہ چکے ہیں۔"

            # Prime Minister of Pakistan
            if re.search(r"\b(priminister|prime\s*minister|pm|wazir\s*e?\s*azam|وزیر\s*اعظم)\b", q_clean) and (
                re.search(r"\b(pakistan|pak|پاکستان|current|moujuda|kon|who|موجودہ|کون|نام)\b", q_clean)
            ):
                return "پاکستان کے موجودہ وزیرِ اعظم جناب میاں محمد شہباز شریف ہیں، جنہوں نے مارچ 2024 میں ملک کے 24ویں وزیرِ اعظم کے طور پر حلف اٹھایا۔ ان کا تعلق پاکستان مسلم لیگ (ن) سے ہے اور وہ ملکی انتظامیہ اور معاشی امور کی سربراہی کر رہے ہیں۔"

            # Chief of Army Staff (Army Chief)
            if re.search(r"\b(army\s*chief|coas|chief\s*of\s*army\s*staff|sarbarah|آرمی\s*چیف|سپہ\s*سالار|چیف\s*آف\s*آرمی\s*اسٹاف)\b", q_clean):
                return "پاکستان کے موجودہ چیف آف آرمی اسٹاف (آرمی چیف) جنرل سید عاصم منیر، نشانِ امتیاز (ملٹری) ہیں، جنہوں نے 29 نومبر 2022 کو پاک فوج کے 17ویں سپہ سالار کی حیثیت سے کمان سنبھالی۔"

            # Chief Justice of Pakistan
            if re.search(r"\b(chief\s*justice|cjp|adliya|چیف\s*جسٹس)\b", q_clean):
                return "پاکستان کے موجودہ چیف جسٹس، محترم جسٹس یحییٰ آفریدی ہیں، جنہوں نے 26 اکتوبر 2024 کو پاکستان کے 28ویں چیف جسٹس کی حیثیت سے حلف اٹھایا ہے۔"

            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pakistan_knowledge_dataset.jsonl")
            if not os.path.exists(dataset_path):
                return None
            query_lower = query.lower()
            words = set(re.findall(r'\w+', query_lower))
            if not words:
                return None
            best_match = None
            best_score = 0
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    q = data.get("metadata", {}).get("question", "").lower()
                    cat = data.get("metadata", {}).get("category", "").lower()
                    q_words = set(re.findall(r'\w+', q))
                    cat_words = set(re.findall(r'\w+', cat))
                    overlap = len(words.intersection(q_words)) * 2 + len(words.intersection(cat_words))
                    if overlap > best_score and overlap >= 2:
                        best_score = overlap
                        for m in data.get("messages", []):
                            if m["role"] == "assistant":
                                cand = m["content"].strip()
                                if cand and "سمجھ لیا ہے" not in cand and "کیا مزید مدد" not in cand:
                                    best_match = cand
            return best_match
        except Exception as ex:
            print(f"[Orchestrator] Local knowledge retrieval notice: {ex}")
            return None

    def detect_topic_from_text(self, text: str) -> str:
        """Detects high-level topic category or summary from text for episodic SQLite tracking."""
        if not text:
            return ""
        lower = text.lower().strip()
        if re.search(r"\b(wwe|wrestling|smackdown|raw|ریسلنگ)\b", lower):
            return "WWE ریسلنگ اور مقابلے"
        if re.search(r"\b(cricket|match|score|psl|کرکٹ|میچ|اسکور|ورلڈ\s*کپ|ہاکی|hockey|football|فٹبال)\b", lower):
            return "کھیل اور میچ"
        if re.search(r"\b(weather|temperature|موسم|بارش|گرمی|سردی|درجہ\s*حرارت|مون\s*سون)\b", lower):
            return "موسم اور ماحولیات"
        if re.search(r"\b(food|dish|dishes|cuisine|khane|khana|khano|pakwan|بریانی|نہاری|کھانا|کھانے|پکوان|سجی|کباب|حلیم|کڑاہی)\b", lower):
            return "پاکستانی کھانے اور پکوان"
        if re.search(r"\b(nadra|fbr|pta|sbp|نادرا|پاسپورٹ|شناختی\s*کارڈ|ٹیکس)\b", lower):
            return "سرکاری محکمے اور عوامی خدمات"
        if re.search(r"\b(jazz|zong|ufone|telenor|telecom|جاز|زونگ|یوفون|ٹیلی\s*نار|انٹرنیٹ)\b", lower):
            return "ٹیلی کام نیٹ ورکس"
        if re.search(r"\b(prime\s*minister|president|وزیر\s*اعظم|صدر|شہباز|آصف|حکومت|اسمبلی)\b", lower):
            return "قومی قیادت اور سیاست"
        if re.search(r"\b(1947|history|تاریخ|قائد\s*اعظم|اقبال|قرارداد|پاکستان)\b", lower):
            return "تاریخ اور قومی ورثہ"
        if re.search(r"\b(k2|indus|سندھ|گوادر|تھر|پہاڑ|دریا|صوبہ|subay|provinces)\b", lower):
            return "جغرافیہ پاکستان"
        
        words = text.split()
        return " ".join(words[:4])

    def _persist_memory_silently(self, user_text: str, active_profile: Optional[Dict[str, Any]] = None):
        """
        Runs asynchronously in a background thread.
        Silently extracts user profile signals and updates SQLite without any voice delay or user alerts.
        """
        try:
            has_name = bool(active_profile and active_profile.get("name"))
            extracted = extract_profile_details_from_text(user_text, has_existing_name=has_name)

            new_name = extracted.get("name")
            new_prof = extracted.get("profession")
            new_country = extracted.get("country")
            new_edu = extracted.get("education")

            if not (new_name or new_prof or new_country or new_edu):
                return

            prof_id = active_profile.get("id") if active_profile else None
            name_to_save = new_name or (active_profile.get("name") if active_profile else None)

            if name_to_save:
                save_or_update_user_profile(
                    name=name_to_save,
                    profession=new_prof or (active_profile.get("profession") if active_profile else None),
                    country=new_country or (active_profile.get("country") if active_profile else None),
                    education=new_edu or (active_profile.get("education") if active_profile else None),
                    info_sharing_consent="consented",
                    profile_id=prof_id
                )
                print(f"[Orchestrator-Memory] Silently persisted profile update for {name_to_save} in background.")
        except Exception as e:
            print(f"[Orchestrator-Memory] Background silent persistence notice: {e}")

    def _persist_dialogue_silently(
        self,
        user_text: str,
        bot_reply: str,
        topic: str,
        session_id: str = "default_session"
    ):
        """Silently records conversation turn and active context in SQLite without blocking voice playback."""
        try:
            record_dialogue_turn(
                user_query=user_text,
                bot_response=bot_reply,
                detected_topic=topic,
                session_id=session_id
            )
            topic_safe = str(topic).encode("ascii", errors="replace").decode("ascii")
            print(f"[Orchestrator-Memory] Silently persisted dialogue turn (topic: {topic_safe})")
        except Exception as e:
            print(f"[Orchestrator-Memory] Turn persistence notice: {e}")

    def orchestrate_turn(
        self,
        user_text: str,
        conversation_history: List[Any],
        active_profile: Optional[Dict[str, Any]] = None,
        persona: str = "Tabraiz",
        session_id: str = "default_session"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main cognitive dispatch method.
        Inspects SQLite topic context, resolves anaphora/follow-ups,
        updates memory asynchronously, and formulates an untruncated response.
        """
        clean_input = user_text.strip() if isinstance(user_text, str) else self.clean_voice_text(user_text)
        updated_profile = dict(active_profile) if active_profile else {}

        # 1. Fetch episodic context & active topic from SQLite
        active_ctx = get_active_discussion_topic(session_id=session_id)
        active_topic = active_ctx.get("active_topic", "")
        last_query = active_ctx.get("last_query", "")

        # 2. Check for conversational follow-up / anaphora (e.g. "bato na", "aur?", "us mai")
        is_followup = self.is_followup_query(clean_input, active_topic, last_query)
        if is_followup and last_query:
            effective_input = f"{last_query} {clean_input}"
            print(f"[Orchestrator] Follow-up detected! Contextually expanded '{clean_input}' -> '{effective_input}'")
        else:
            effective_input = clean_input

        # 3. Automatically resolve Pre-Standard-Prompt Mode
        mode = self.resolve_prompt_mode(
            user_text=clean_input,
            conversation_history=conversation_history,
            active_topic=active_topic,
            last_query=last_query
        )
        print(f"[Orchestrator] Dynamic Pre-Standard-Prompt Mode: '{mode}' for input: '{clean_input}'")
        active_directive = DYNAMIC_PRE_STANDARD_DIRECTIVES.get(mode, TABRAIZ_PRE_STANDARD_CASUAL)
        active_system_prompt = get_tabraiz_system_prompt(mode)

        # 4. Dispatch Asynchronous Silent Memory Extraction (Zero Latency Impact)
        self._memory_executor.submit(self._persist_memory_silently, clean_input, active_profile)

        # 5. Classify Intent
        intent = self.classify_intent(effective_input)
        if is_followup and intent in ["GENERAL", "GREETING"]:
            intent = "KNOWLEDGE_SEARCH"

        # 6. Process Intent Fast Paths (Identity, Greeting, Time/Date)
        # A. Identity
        if intent == "IDENTITY":
            reply = "آداب! میں تبریز ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"
            cleaned_reply = self.clean_voice_text(reply, is_identity=True)
            self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, "تعارف اور شناخت", session_id)
            return cleaned_reply, updated_profile

        # B. Greeting & Well-being Inquiries (Instant 0.01s polite response)
        if intent == "GREETING":
            is_wellbeing = any(re.search(p, clean_input.lower(), re.IGNORECASE) for p in [
                r"\b(hal|haal|kese|kaise|kaisa|kaisi|theek|kheriat|how\s+are|how\s+do)\b",
                r"(حال|کیسے|کیسی|خیریت|مزاج|طبیعت)"
            ])
            name = updated_profile.get("display_name") or updated_profile.get("name")
            if is_wellbeing:
                if name and name != "محترم مہمان":
                    reply = f"وعلیکم السلام {name} صاحب! الحمدللہ، میں بالکل خیریت سے ہوں۔ آپ سنائیں، آپ کا مزاج کیسا ہے؟ بتائیں، آج میں آپ کی کیا مدد کر سکتا ہوں؟"
                else:
                    reply = "وعلیکم السلام! الحمدللہ، میں بالکل خیریت سے ہوں۔ آپ سنائیں، آپ کا مزاج کیسا ہے؟ بتائیں، آج میں آپ کی کیا مدد کر سکتا ہوں؟"
            else:
                if name and name != "محترم مہمان":
                    reply = f"وعلیکم السلام {name} صاحب! کیسے ہیں آپ؟ بتائیں، آج میں آپ کی کیا مدد کر سکتا ہوں؟"
                else:
                    reply = "وعلیکم السلام! کیسے ہیں آپ؟ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"
            cleaned_reply = self.clean_voice_text(reply)
            self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, "سلام اور آداب", session_id)
            return cleaned_reply, updated_profile

        # C. Local Time & Date
        if intent == "TIME_DATE":
            time_reply = get_time_date_response(clean_input, persona="Tabraiz")
            cleaned_reply = self.clean_voice_text(time_reply)
            self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, "وقت اور تاریخ", session_id)
            return cleaned_reply, updated_profile

        # C.2 Mode 4: IMAGE_TURBO (Z-Image-Turbo on Modal GPU & Anti-Nudity Guardrails)
        if mode == "IMAGE_TURBO" or intent == "IMAGE_GENERATION":
            # 1. Anti-Nudity & Safety Guardrail Check
            if self.is_nsfw_or_inappropriate(clean_input):
                print(f"[Orchestrator] Anti-Nudity Guardrail tripped for user input: '{clean_input}'")
                refusal_reply = "معذرت خواہ ہوں جناب! اخلاقی اور تہذیبی اصولوں کے تحت ایسی تصاویر کی تخلیق ممکن نہیں ہے۔ اگر آپ کوئی قدرتی مناظر، تاریخی عمارات یا فنکارانہ موضوع تجویز فرمائیں تو مجھے خوشی ہوگی۔"
                cleaned_refusal = self.clean_voice_text(refusal_reply)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_refusal, "تصویری درخواست (ممنوعہ مواد)", session_id)
                return cleaned_refusal, updated_profile

            # 2. Refine Prompt into 8k English Diffusion Prompt + Polite Spoken Urdu Reply
            refined_prompt, spoken_reply, img_cat = self.refine_image_prompt(effective_input)
            print(f"[Orchestrator] Image generation requested: '{clean_input}'")
            print(f"[Orchestrator] Refined Prompt ({img_cat}): '{refined_prompt}'")

            img_res = None
            # 3. Call Modal.com Z-Image-Turbo GPU Backend
            if self.backend_client and hasattr(self.backend_client, "generate_image"):
                try:
                    print(f"[Orchestrator] Dispatching to Modal.com Z-Image-Turbo GPU...")
                    img_res = self.backend_client.generate_image(refined_prompt)
                except Exception as img_err:
                    print(f"[Orchestrator] Modal Z-Image-Turbo invocation notice: {img_err}")

            if img_res and img_res.get("status") == "success":
                b64_data = img_res.get("image_base64", "")
                fname = img_res.get("filename", f"adaab_img_{int(time.time())}.png")
                cloud_path = img_res.get("modal_storage_path", "")

                # Mirror a local copy to H:/Adaab/output/images/ if possible
                try:
                    import base64
                    local_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "images")
                    os.makedirs(local_out_dir, exist_ok=True)
                    local_file = os.path.join(local_out_dir, fname)
                    if b64_data.startswith("data:image"):
                        raw_bytes = base64.b64decode(b64_data.split(",", 1)[1])
                        with open(local_file, "wb") as f_img:
                            f_img.write(raw_bytes)
                except Exception as save_err:
                    print(f"[Orchestrator] Local image mirror notice: {save_err}")

                card_payload = {
                    "b64": b64_data,
                    "prompt": refined_prompt,
                    "filename": fname,
                    "modal_path": cloud_path,
                    "category": img_cat
                }
                full_reply = f"{spoken_reply}\n\n[IMAGE_CARD:{json.dumps(card_payload)}]"
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, f"تصویر تیار کی گئی: {fname}", "تصویر سازی", session_id)
                return full_reply, updated_profile
            else:
                # If backend is cold or unavailable
                offline_reply = "معذرت جناب! تصویر بنانے کا کلاؤڈ سرور اس وقت شروع ہو رہا ہے یا مصروف ہے۔ براہ کرم چند لمحوں بعد دوبارہ فرمائیے۔"
                cleaned_offline = self.clean_voice_text(offline_reply)
                return cleaned_offline, updated_profile

        # D. Mode 5: CODE_TECH (Programming Code or Long Article / Essay Generation)
        if mode == "CODE_TECH" or intent == "CODE_OR_ARTICLE":
            turn_topic = "کوڈنگ اور تحریر"
            prompt = f"""
صارف کی درخواست: {effective_input}

{active_directive}

ہدایات:
ایک سمجھدار، باادب اور دوستانہ پاکستانی دوست کی طرح سلیس، باوقار اور خوبصورت اردو میں جواب دیں۔
اگر صارف نے کوئی مضمون، تحریر یا کہانی مانگی ہے تو پہلے 1 سے 2 انتہائی شائستہ اور محبت بھرے جملے بولیں (جیسے 'جی بالکل، آپ کی فرمائش کے مطابق تحریر حاضر ہے') اور اس کے فوراً بعد دلنشین پیراگراف میں مکمل مضمون پیش کریں۔
اگر پروگرامنگ کوڈ مانگا ہے تو مختصر تعارف کے بعد صاف اور درست مارک ڈاؤن کوڈ بلاک پیش کریں۔
کسی قسم کے مصنوعی عنوانات جیسے 'پہلا حصہ' یا 'دوسرا حصہ' مت لکھیں۔
""".strip()

            reply_text = self.dispatch_tiered_llm(
                prompt=prompt,
                system_prompt=active_system_prompt,
                max_tokens=900,
                temperature=0.4,
                mode_label="Code/Article",
                timeout=15.0
            )

            if reply_text:
                # Sanitize any accidental prompt header leaks or foreign characters
                reply_text = re.sub(r'#+\s*(?:پہلا حصہ|دوسرا حصہ|صوتی کلام|مارک ڈاؤن بلاک)[^\n]*\n?', '', reply_text)
                reply_text = re.sub(r'[\u0400-\u04FF]', '', reply_text)
                reply_text = re.sub(r'[\u06D0\u06D1\u0681\u0696\u0685\u06AB\u06BC]', '', reply_text)
                reply_text = re.sub(r'([\u0600-\u06FF])[a-zA-Z]+', r'\1', reply_text)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, reply_text[:150], turn_topic, session_id)
                return reply_text, updated_profile

        # E. Mode 6: ADAB_POETRY (Classical & Contemporary Metered Verse Recitation)
        if mode == "ADAB_POETRY":
            turn_topic = "ادب اور شاعری"
            prompt = f"""
صارف کی فرمائش: {effective_input}

{active_directive}

صارف کے ادبی ذوق اور فرمائش کے عین مطابق باوزن شعر، غزل یا ادبی جواب پیش کریں۔
اگر کسی معروف شاعر (علامہ اقبال، مرزا غالب، فیض، فراز، جون ایلیا وغیرہ) کا کلام ہے تو شاعر کا نام ضرور واضح کریں۔
کوئی ایموجی مت لگائیں اور مارک ڈاؤن یا بلٹ پوائنٹس مت بنائیں تاکہ صوتی روانی اور ترنم قائم رہے۔
""".strip()
            poetry_reply = self.dispatch_tiered_llm(
                prompt=prompt,
                system_prompt=active_system_prompt,
                max_tokens=400,
                temperature=0.75,
                mode_label="Poetry",
                timeout=12.0
            )

            if poetry_reply:
                cleaned_reply = self.clean_voice_text(poetry_reply)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, turn_topic, session_id)
                return cleaned_reply, updated_profile

        # F. Mode 2: DEEP_EXPLORATION (Comprehensive Conceptual Breakdown & In-Depth Education)
        if mode == "DEEP_EXPLORATION":
            local_fact = self.retrieve_local_pakistan_knowledge(effective_input)
            search_snippets = self.oracle.search_web(effective_input, max_results=3)
            context_blocks = []
            if local_fact:
                context_blocks.append(f"پاکستان سے متعلق مصدقہ حقائق:\n{local_fact}")
            if search_snippets:
                context_blocks.extend(search_snippets)
            context_str = "\n\n".join(context_blocks) if context_blocks else ""

            turn_topic = self.detect_topic_from_text(effective_input) or active_topic or "علمی و فکری رہنمائی"
            prompt = f"""
موضوع: {turn_topic}
صارف کا سوال: {effective_input}
متعلقہ معلوماتی پس منظر:
{context_str}

{active_directive}

ہدایت برائے تفصیلی رہنمائی:
صارف نے موضوع پر گہری اور تفصیلی رہنمائی طلب کی ہے۔ جواب کو 2 سے 3 جملوں پر ہرگز محدود نہ کریں۔
موضوع کے اہم تصورات، پس منظر، اور اہم نکات کو آسان، شستہ اور فصیح پاکستانی اردو میں ترتیب سے واضح کریں۔
مخاطب کے لیے ہمیشہ 'آپ' کا باادب صیغہ استعمال کریں۔
""".strip()
            deep_reply = self.dispatch_tiered_llm(
                prompt=prompt,
                system_prompt=active_system_prompt,
                max_tokens=850,
                temperature=0.55,
                mode_label="Deep exploration",
                timeout=15.0
            )

            if deep_reply:
                cleaned_reply = self.clean_voice_text(deep_reply)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, turn_topic, session_id)
                return cleaned_reply, updated_profile

        # G. Mode 3: CURRENT_AFFAIRS (Journalistic Facts & Latest 2026 Grounding)
        if mode == "CURRENT_AFFAIRS":
            search_snippets = self.oracle.search_web(effective_input, max_results=4)
            context_str = "\n\n".join(search_snippets) if search_snippets else ""
            turn_topic = self.detect_topic_from_text(effective_input) or "حالاتِ حاضرہ اور خبریں"
            prompt = f"""
موضوع: {turn_topic}
صارف کا سوال: {effective_input}
تازہ ترین مصدقہ سرچ نتائج (2026/تازہ خبریں):
{context_str}

{active_directive}

صارف کے سوال کا جواب تازہ ترین صورتحال کی روشنی میں غیر جانبدارانہ اور سچے انداز میں پیش کریں۔
اہم ترین بات پہلے بیان کریں اور صورتحال کا متوازن، باخبر اور پرمغز خلاصہ (3 سے 4 جامع جملوں میں) پیش کریں۔
""".strip()
            affairs_reply = self.dispatch_tiered_llm(
                prompt=prompt,
                system_prompt=active_system_prompt,
                max_tokens=400,
                temperature=0.50,
                mode_label="Current affairs",
                timeout=12.0
            )

            if affairs_reply:
                cleaned_reply = self.clean_voice_text(affairs_reply)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, turn_topic, session_id)
                return cleaned_reply, updated_profile

        # H. Mode 1: CASUAL_VOICE - Knowledge Lookup (Standard factual search)
        if intent == "KNOWLEDGE_SEARCH":
            local_fact = self.retrieve_local_pakistan_knowledge(effective_input)
            search_snippets = self.oracle.search_web(effective_input, max_results=3)
            context_blocks = []
            if local_fact:
                context_blocks.append(f"پاکستان سے متعلق مصدقہ حقائق:\n{local_fact}")
            if search_snippets:
                context_blocks.extend(search_snippets)
            context_str = "\n\n".join(context_blocks) if context_blocks else ""

            turn_topic = self.detect_topic_from_text(effective_input)
            topic_header = f"موضوع: {turn_topic or active_topic}\n" if (turn_topic or active_topic) else ""

            oracle_reply = None
            prompt = f"""
{topic_header}صارف کا سوال: {effective_input}
متعلقہ تازہ معلومات:
{context_str}

{active_directive}
صارف کے سوال کا واضح، سچا اور انتہائی مناسب خلاصہ پیش کریں (زیادہ سے زیادہ 2 سے 3 جملوں میں)۔
کوئی ایموجی مت لگائیں اور مارک ڈاؤن یا بلٹ پوائنٹس استعمال نہ کریں۔
""".strip()

            oracle_reply = self.dispatch_tiered_llm(
                prompt=prompt,
                system_prompt=active_system_prompt,
                max_tokens=250,
                temperature=0.55,
                mode_label="Knowledge query",
                search_context=context_str,
                timeout=10.0
            )

            if oracle_reply:
                cleaned_reply = self.clean_voice_text(oracle_reply)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, turn_topic, session_id)
                return cleaned_reply, updated_profile

        # I. Mode 1: CASUAL_VOICE - General Conversational Turn (Dialogue & Pleasantries)
        turn_topic = self.detect_topic_from_text(effective_input)
        messages = [{"role": "system", "content": active_system_prompt}]

        # Inject known user profile context politely into system context
        user_name = updated_profile.get("name")
        user_city = updated_profile.get("country") or updated_profile.get("city")
        user_prof = updated_profile.get("profession")
        if user_name or user_city or user_prof:
            mem_items = []
            if user_name: mem_items.append(f"نام: {user_name}")
            if user_city: mem_items.append(f"شہر: {user_city}")
            if user_prof: mem_items.append(f"پیشہ: {user_prof}")
            mem_ctx = "، ".join(mem_items)
            messages.append({"role": "system", "content": f"مخاطب کے سابقہ کوائف: {mem_ctx}"})

        # Inject recent episodic dialogue context from SQLite if history is short
        if not conversation_history:
            recent_turns = get_recent_conversation_context(limit=3, session_id=session_id)
            for t in recent_turns:
                messages.append({"role": "user", "content": t["user_query"]})
                messages.append({"role": "assistant", "content": t["bot_response"]})
        else:
            for h in conversation_history[-6:]:
                if isinstance(h, dict):
                    r = h.get("role", "user")
                    c = self.clean_voice_text(h.get("content", ""))
                    if c:
                        messages.append({"role": r, "content": c})
                elif isinstance(h, (list, tuple)) and len(h) >= 2:
                    u = self.clean_voice_text(str(h[0]))
                    a = self.clean_voice_text(str(h[1]))
                    if u: messages.append({"role": "user", "content": u})
                    if a: messages.append({"role": "assistant", "content": a})

        # Add current user turn with Standard Prompt formatting
        topic_str = f"سابقہ موضوع: {active_topic}\n" if active_topic else ""
        conv_prompt = (
            f"{topic_str}صارف کا پیغام: {effective_input}\n"
            f"{active_directive}\n"
            f"روزمرہ بول چال کی قدرتی اور آسان اردو میں باادب اور انتہائی مناسب انداز میں صرف 2 سے 3 جملوں کا جامع خلاصہ پیش کریں۔ مخاطب کے لیے ہمیشہ 'آپ' کا احترام رکھیں، 'تم' مت کہیں۔ کوئی جملہ بار بار مت دہرائیں۔"
        )
        messages.append({"role": "user", "content": conv_prompt})

        final_reply = self.dispatch_tiered_llm(
            prompt=conv_prompt,
            system_prompt=active_system_prompt,
            max_tokens=250,
            temperature=0.60,
            mode_label="Turn",
            messages=messages,
            timeout=10.0
        )

        if final_reply:
            final_reply = self.clean_voice_text(final_reply)

        # Intelligent Fallback if both cloud endpoints are unavailable
        if not final_reply:
            final_reply = (
                "جی میں سمجھ گیا۔ آپ کا سوال انتہائی اہم ہے۔ "
                "فرمائیے، اس حوالے سے میں آپ کی مزید کیا مختصر اور باادب رہنمائی کر سکتا ہوں؟"
            )

        self._memory_executor.submit(self._persist_dialogue_silently, clean_input, final_reply, turn_topic, session_id)
        return final_reply, updated_profile


# Global orchestrator singleton
_orchestrator: Optional[CognitiveOrchestrator] = None

def get_orchestrator(backend_client=None) -> CognitiveOrchestrator:
    """Returns the singleton CognitiveOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CognitiveOrchestrator(backend_client=backend_client)
    elif backend_client and _orchestrator.backend_client is None:
        _orchestrator.backend_client = backend_client
    return _orchestrator
