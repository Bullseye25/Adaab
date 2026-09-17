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

# System prompt defining Tabraiz's persona for the orchestrator
TABRAIZ_ORCHESTRATOR_SYSTEM = """
آپ کا نام "تبریز" (Tabraiz) ہے۔ آپ ایک دوستانہ، سمجھدار اور باادب اردو صوتی اسسٹنٹ اور رفیق ہیں۔
آپ کا تعلق "آداب" (Adaab Studio) سے ہے۔

سنہری اور لازمی اصول:
1. ادب اور بے تکلفی کا بہترین توازن (Casual yet Respectful Conversational Urdu):
   • گفتگو کا انداز دوستانہ، قدرتی اور روزمرہ عام بول چال کا ہو، جیسے دو پڑھے لکھے اچھے دوست آپس میں بات کرتے ہیں۔
   • ادب کا اصول: مخاطب کو ہمیشہ "آپ"، "آپ کا"، "آپ کو" کہہ کر بلائیں۔ لفظ "تم" یا "تو" کا استعمال ہرگز نہ کریں!
   • بھاری، کتابی اور پرانی درباری زبان (جیسے "عرض ہے"، "سماعت فرمائیے"، "حضورِ والا"، "تہذیب و تمدن"، "ناچیز") سے مکمل پرہیز کریں۔
   • آسان، سیدھی، اور عام فہم پاکستانی اردو بولیں جس میں روزمرہ کے مانوس الفاظ فطری طور پر آئیں۔
2. نام بار بار نہ دہرائیں: اپنے نام "تبریز" کو ہر جملے کے شروع میں مت بولیں۔ صارف پہلے سے جانتا ہے کہ وہ آپ سے مخاطب ہے۔
3. خالص صوتی اسسٹنٹ رویہ: کبھی بھی ڈیٹا بیس، فارم بھرنے یا کوائف محفوظ کرنے کی بات نہ کریں۔ سیدھا صارف کی بات کا آسان، مددگار اور واضح جواب دیں۔
4. صوتی ساخت: جواب واضح، جامع اور بولنے میں آسان ہو (2 سے 3 جملے)، جو سننے والے کو بالکل قدرتی لگے۔
5. کوئی ایموجی مت لگائیں اور مارک ڈاؤن یا بلٹ پوائنٹس مت بنائیں تاکہ آڈیو روانی سے ادا ہو سکے۔

گفتگو کے مثالی نمونے (Few-Shot Conversational Examples):
صارف: پاکستان میں مشہور کھانے کون سے ہیں؟
تبریز: پاکستان کے روایتی کھانوں میں کراچی کی بریانی، لاہور کی نہاری، پشاور کے چپلی کباب اور بلوچستان کی سجی بے حد مقبول ہیں۔ ہر علاقے کا اپنا ایک خاص ذائقہ ہے جو دسترخوان کی رونق بڑھاتا ہے۔ بتائیے، آپ کو ان میں سے کون سا پکوان سب سے زیادہ پسند ہے؟

صارف: آج کا دن کیسا گزر رہا ہے؟
تبریز: جی الحمدللہ، سب خیریت ہے۔ آپ کا دن کیسا گزر رہا ہے؟ بتائیے، آج میں آپ کی کیا مدد کر سکتا ہوں؟
""".strip()


class CognitiveOrchestrator:
    def __init__(self, backend_client=None):
        self.backend_client = backend_client
        self.oracle = get_gemini_oracle()
        # Non-blocking background worker pool for silent memory persistence
        self._memory_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adaab_silent_mem")

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

        # 6. Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # 7. Detect and prune severe n-gram loops and duplicate sentences
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

    def classify_intent(self, user_text: str) -> str:
        """Classifies the user's intent into discrete operational categories."""
        if not user_text:
            return "GENERAL"
        
        lower = user_text.lower().strip()

        # 1. Identity & Reflexive
        identity_patterns = [
            r"\bwho are you\b", r"\bwhat is your name\b", r"\bwhat can you do\b",
            r"آپ کون ہیں", r"تم کون ہو", r"کون بول رہا ہے", r"آپ کا نام کیا ہے",
            r"تم کیا کر سکتے ہو", r"کیا کام کرتے ہو"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in identity_patterns):
            return "IDENTITY"

        # 2. Greeting & Etiquette
        greeting_patterns = [
            r"^(?:ہیلو|سلام|السلام\s*علیکم|آداب|hello|hi|hey|salam|assalam\s*o\s*alaikum)[\s!۔،]*$"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in greeting_patterns):
            return "GREETING"

        # 3. Time and Date Query
        if is_time_or_date_query(user_text):
            return "TIME_DATE"

        # 4. Real-time Knowledge / Web Search
        knowledge_patterns = [
            r"موسم", r"درجہ\s*حرارت", r"weather", r"temperature",
            r"کرکٹ", r"میچ", r"اسکور", r"cricket", r"score", r"match",
            r"خبریں", r"تازہ\s*ترین", r"news", r"latest",
            r"وزیر\s*اعظم", r"صدر", r"prime\s*minister", r"president",
            r"سرچ\s*کرو", r"تلاش\s*کرو", r"search\s*for", r"google",
            r"2025", r"2026", r"آج\s*کا", r"today",
            r"\bwwe\b", r"\bwrestling\b", r"ریسلنگ", r"سمیک\s*ڈاؤن", r"\braw\b",
            r"نادرا", r"\bnadra\b", r"\bfbr\b", r"\bpta\b", r"\bsbp\b",
            r"کون\s*جیتا", r"who\s*won", r"جیت\s*کس\s*کی\s*ہوئی",
            r"کھانے", r"بریانی", r"نہاری", r"پکوان", r"تاریخ", r"1947",
            r"قائد\s*اعظم", r"علامہ\s*اقبال", r"کے\s*ٹو", r"\bk2\b",
            r"جاز", r"زونگ", r"یوفون", r"ٹیلی\s*نار", r"telecom",
            r"\b(khane|khana|khano|food|dishes|cuisine|famous|mashhoor|pakwan|biryani|nihari|sajji|chapli|kabab|haleem|karahi)\b",
            r"\b(subay|provinces|arshad\s*nadeem|1992\s*world\s*cup)\b"
        ]
        if any(re.search(p, lower, re.IGNORECASE) for p in knowledge_patterns):
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
            r"^اور\??$", r"^پھر\??$", r"^اور\s*پھر\??$", r"^اور\s*کیا\b",
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
        if len(words) <= 3 and any(w in ["bato", "batao", "na", "aur", "phir", "us", "wahan", "kon", "jita", "tell", "more", "next", "اور", "پھر", "بتاؤ", "بتائیں"] for w in words):
            return True

        return False

    def retrieve_local_pakistan_knowledge(self, query: str) -> Optional[str]:
        """Retrieves ground-truth Pakistani cultural and factual knowledge from local dataset."""
        try:
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

        # 3. Dispatch Asynchronous Silent Memory Extraction (Zero Latency Impact)
        self._memory_executor.submit(self._persist_memory_silently, clean_input, active_profile)

        # 4. Classify Intent
        intent = self.classify_intent(effective_input)
        if is_followup and intent in ["GENERAL", "GREETING"]:
            intent = "KNOWLEDGE_SEARCH"

        # 5. Process Intent
        # A. Identity
        if intent == "IDENTITY":
            reply = "آداب! میں تبریز ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"
            cleaned_reply = self.clean_voice_text(reply, is_identity=True)
            self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, "تعارف اور شناخت", session_id)
            return cleaned_reply, updated_profile

        # B. Greeting (Zero onboarding interrogation)
        if intent == "GREETING":
            name = updated_profile.get("display_name") or updated_profile.get("name")
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

        # D. Real-Time Knowledge & Search
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
متعلقہ معلومات:
{context_str}

ہدایات:
آپ تبریز ہیں، ایک دوستانہ اور باادب پاکستانی اردو صوتی اسسٹنٹ۔
زبان کا توازن: گفتگو کا انداز روزمرہ بول چال کی عام اور آسان اردو ہو (نہ کہ کتابی یا ثقیل الفاظ)۔
ادب کا دائرہ: ہمیشہ "آپ" کا صیغہ استعمال کریں ("آپ کا"، "آپ کو"، "بتائیں")۔ لفظ "تم" یا "تو" کا استعمال قطعی نہ کریں۔
اگر سوال رومن اردو میں پوچھا گیا ہو تو بھی روزمرہ بول چال کی شائستہ اردو میں مکمل اور قدرتی جواب دیں۔
صارف کے سوال کا واضح، قدرتی اور مکمل جواب دیں تاکہ کوئی بات ادھوری نہ رہے۔
کسی بھی جملے یا لفظ کو بار بار مت دہرائیں، اور کوئی ایموجی یا مارک ڈاؤن بلٹ پوائنٹس مت بنائیں۔
""".strip()

            if self.oracle.is_available():
                oracle_reply = self.oracle.query(
                    prompt=prompt,
                    system_instruction=TABRAIZ_ORCHESTRATOR_SYSTEM,
                    timeout=14
                )
            else:
                try:
                    from chatgpt_browser_oracle import get_chatgpt_oracle
                    chatgpt = get_chatgpt_oracle()
                    if chatgpt.is_available():
                        print("[Orchestrator] Gemini paused. Using ChatGPT Web Oracle fallback...")
                        oracle_reply = chatgpt.query(
                            prompt=prompt,
                            system_instruction=TABRAIZ_ORCHESTRATOR_SYSTEM,
                            timeout=35
                        )
                except Exception as c_err:
                    print(f"[Orchestrator] ChatGPT fallback notice: {c_err}")

            if oracle_reply:
                cleaned_reply = self.clean_voice_text(oracle_reply)
                self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, turn_topic, session_id)
                return cleaned_reply, updated_profile

            # Fallback to Modal GPU (Qwen 2.5) with search context if Gemini unavailable or tripped
            if self.backend_client and context_str:
                try:
                    fallback_prompt = (
                        f"{topic_header}معلومات:\n{context_str}\n\n"
                        f"صارف کا سوال: {effective_input}\n"
                        f"اس معلومات کی روشنی میں روزمرہ اور عام فہم اردو میں مکمل جواب دیں۔ مخاطب کو ہمیشہ 'آپ' کہیں اور 'تم' سے پرہیز کریں۔ کوئی جملہ بار بار مت دہرائیں اور غیر ضروری تکرار سے گریز کریں۔"
                    )
                    messages = [
                        {"role": "system", "content": TABRAIZ_ORCHESTRATOR_SYSTEM},
                        {"role": "user", "content": fallback_prompt}
                    ]
                    resp = self.backend_client.chat_completion(
                        messages,
                        temperature=0.65,
                        max_tokens=800,
                        repetition_penalty=1.20,
                        presence_penalty=0.5,
                        frequency_penalty=0.5
                    )
                    bot_text = resp.get("content", "").strip()
                    if bot_text:
                        cleaned_reply = self.clean_voice_text(bot_text)
                        self._memory_executor.submit(self._persist_dialogue_silently, clean_input, cleaned_reply, turn_topic, session_id)
                        return cleaned_reply, updated_profile
                except Exception as ex:
                    print(f"[Orchestrator] Modal fallback notice: {ex}")

        # E. General Conversational Turn (Dialogue & Reasoning)
        turn_topic = self.detect_topic_from_text(effective_input)
        messages = [{"role": "system", "content": TABRAIZ_ORCHESTRATOR_SYSTEM}]

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

        # Add current user turn
        messages.append({"role": "user", "content": effective_input})

        # Generate Response: Prioritize Gemini Oracle for fluent native conversational Urdu,
        # falling back gracefully to Modal L4 GPU or ChatGPT Web Oracle
        final_reply = None

        topic_str = f"سابقہ موضوع: {active_topic}\n" if active_topic else ""
        conv_prompt = f"{topic_str}صارف کا پیغام: {effective_input}\nروزمرہ بول چال کی قدرتی اور آسان اردو میں باادب اور مکمل جواب دیں۔ مخاطب کے لیے ہمیشہ 'آپ' کا احترام رکھیں، 'تم' مت کہیں۔ کوئی جملہ بار بار مت دہرائیں۔"

        if self.oracle.is_available():
            final_reply = self.oracle.query(
                prompt=conv_prompt,
                system_instruction=TABRAIZ_ORCHESTRATOR_SYSTEM,
                timeout=14
            )

        if not final_reply and self.backend_client:
            try:
                resp = self.backend_client.chat_completion(
                    messages, 
                    temperature=0.65, 
                    max_tokens=800,
                    repetition_penalty=1.20,
                    presence_penalty=0.5,
                    frequency_penalty=0.5
                )
                final_reply = resp.get("content", "")
            except Exception as e:
                print(f"[Orchestrator] Backend completion notice: {e}")

        # If still no reply, try ChatGPT Web Oracle fallback
        if not final_reply:
            try:
                from chatgpt_browser_oracle import get_chatgpt_oracle
                chatgpt = get_chatgpt_oracle()
                if chatgpt.is_available():
                    final_reply = chatgpt.query(
                        prompt=conv_prompt,
                        system_instruction=TABRAIZ_ORCHESTRATOR_SYSTEM,
                        timeout=35
                    )
            except Exception:
                pass

        if final_reply:
            final_reply = self.clean_voice_text(final_reply)

        # Intelligent Fallback if both cloud endpoints are unavailable
        if not final_reply:
            final_reply = (
                "آداب عرض ہے! میں نے آپ کا ارشاد بخوبی سمجھ لیا ہے۔ "
                "فرمائیے، میں آپ کی کیا مزید مدد یا رہنمائی کر سکتا ہوں؟"
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
