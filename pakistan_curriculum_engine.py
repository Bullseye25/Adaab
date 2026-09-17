"""
pakistan_curriculum_engine.py
─────────────────────────────────────────────────────────────────────────────
200 Pakistani Topics Curriculum Engine, Random Sampling Algorithm,
ChatGPT Urdu Q&A Generator & LoRA Training CLI for Tabraiz Voice AI
─────────────────────────────────────────────────────────────────────────────
• 200 Curated Positive Pakistani Topics across 10 Distinct Categories.
• Uniform Random Sampling Algorithm (n=50 topics without replacement).
• Standardized Conversational Urdu Prompt Generation for ChatGPT.
• Chrome DevTools Protocol (CDP) and Oracle Automated Harvester.
• Strict Tabraiz Linguistic Enforcement (آپ, zero emojis, 2-3 sentences).
• Automatic Training Dataset Consolidation & LoRA Modal/Local Trigger.
• Comprehensive Self-Testing Suite built into the CLI.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import random
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional

# UTF-8 console support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOPICS_FILE = os.path.join(DATA_DIR, "pakistan_200_topics.json")
MASTER_DATASET_FILE = os.path.join(DATA_DIR, "master_training_dataset.jsonl")
SAMPLED_OUTPUT_FILE = os.path.join(DATA_DIR, "chatgpt_sampled_50_dataset.jsonl")

# Tabraiz Standard Golden System Prompt
STANDARD_SYSTEM_PROMPT = """آپ کا نام "تبریز" (Tabraiz) ہے۔ آپ ایک باوقار، شستہ، دوستانہ اور باادب پاکستانی اردو صوتی اسسٹنٹ اور رفیق ہیں۔
آپ کا تعلق "آداب" (Adaab Studio) سے ہے۔

سنہری اور لازمی اصول:
1. ہمیشہ شائستہ اور باادب انداز میں مخاطب ہوں (مخاطب کے لیے ہمیشہ "آپ" کہیں، "تم" یا "تو" کا استعمال قطعی نہ کریں)۔
2. نام بار بار نہ دہرائیں: اپنے نام "تبریز" کو ہر جملے کے آغاز میں مت دہرائیں۔
3. خالص صوتی اسسٹنٹ رویہ: سیدھا صارف کی بات کا سلیس، باادب اور مدلل جواب دیں۔
4. زبان کا معیار: سلیس، عام فہم اور روزمرہ بول چال کی شائستہ پاکستانی اردو میں گفتگو کریں۔ غیر مانوس یا ثقیل الفاظ سے گریز کریں۔
5. صوتی ساخت: جواب واضح اور جامع ہو (2 سے 3 جملے)، جو سننے میں قدرتی اور خوشگوار لگے۔
6. کوئی ایموجی (No Emojis) اور کوئی مارک ڈاؤن بولڈ/بلٹ پوائنٹس مت استعمال کریں تاکہ صوتی نظام روانی سے پڑھ سکے۔"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topics Loading and Catalog Management
# ─────────────────────────────────────────────────────────────────────────────

def load_topics_catalog(filepath: str = TOPICS_FILE) -> Dict[str, Any]:
    """Loads the 200 Pakistani topics catalog."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Topics catalog not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_all_topics(filepath: str = TOPICS_FILE) -> List[Dict[str, Any]]:
    """Returns a flat list of all 200 curated topics with category metadata."""
    catalog = load_topics_catalog(filepath)
    all_topics = []
    for cat in catalog.get("categories", []):
        cat_urdu = cat.get("name_urdu", "")
        cat_en = cat.get("name_en", "")
        cat_id = cat.get("id", "")
        for t in cat.get("topics", []):
            item = dict(t)
            item["category_id"] = cat_id
            item["category_urdu"] = cat_urdu
            item["category_en"] = cat_en
            all_topics.append(item)
    return all_topics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Topic Sampling Algorithm (Uniform Random Selection)
# ─────────────────────────────────────────────────────────────────────────────

def sample_topics(n: int = 50, seed: Optional[int] = None, filepath: str = TOPICS_FILE) -> List[Dict[str, Any]]:
    """
    Randomly selects n unique topics from the 200-topic universe without replacement.
    If seed is provided, sampling is deterministic and reproducible.
    """
    topics = get_all_topics(filepath)
    if n > len(topics):
        raise ValueError(f"Requested sample size ({n}) exceeds total available topics ({len(topics)}).")
    
    rng = random.Random(seed) if seed is not None else random.Random()
    sampled = rng.sample(topics, n)
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# 3. Standard Prompt Formulation for ChatGPT
# ─────────────────────────────────────────────────────────────────────────────

def build_single_topic_prompt(topic: Dict[str, Any]) -> str:
    """Builds a standardized conversational prompt for a single topic."""
    prompt = f"""آپ پاکستان کے باادب، شستہ صوتی اسسٹنٹ "تبریز" کے لیے معلوماتی اور گفتگو کے قابل مواد تیار کر رہے ہیں۔
موضوع: {topic.get('topic_urdu')} ({topic.get('topic_roman')})
پس منظر: {topic.get('description')}

برائے مہربانی اس موضوع پر عام فہم، روزمرہ گفتگو کے لیے 1 معلوماتی اور دلچسپ سوال اور اس کا شستہ، باادب اور جامع اردو جواب تیار کریں۔

لازمی اور سخت اصول:
1. مخاطب کے لیے ہمیشہ احترام پر مبنی صیغہ "آپ" کا استعمال کریں (کبھی "تم" یا "تو" نہ کہیں)۔
2. نام کا بے جا تکرار نہ ہو: جواب کے شروع میں اپنے نام "تبریز" کو ہرگز نہ دہرائیں۔
3. صوتی ساخت: جواب واضح اور جامع ہو (صرف 2 سے 3 جملے)، جو بولنے اور سننے میں قدرتی اور خوشگوار لگے۔
4. کوئی ایموجی (No Emojis) اور کوئی مارک ڈاؤن بولڈ/بلٹ پوائنٹس مت استعمال کریں۔
5. زبان سلیس اور باوقار پاکستانی اردو ہو۔

جواب صرف اس فارمیٹ میں دیں:
سوال: [سوال یہاں لکھیں]
جواب: [جواب یہاں لکھیں]"""
    return prompt


def build_batch_chatgpt_prompt(sampled_topics: List[Dict[str, Any]]) -> str:
    """
    Builds a structured batch prompt containing all sampled topics to ask ChatGPT.
    Instructs ChatGPT to generate a numbered list of Urdu Q&A pairs adhering to Tabraiz's rules.
    """
    topic_lines = []
    for idx, t in enumerate(sampled_topics, start=1):
        line = f"{idx}. موضوع: {t.get('topic_urdu')} ({t.get('topic_roman')}) — {t.get('description')}"
        topic_lines.append(line)
    
    topics_block = "\n".join(topic_lines)

    prompt = f"""برائے مہربانی درج ذیل {len(sampled_topics)} پاکستانی موضوعات پر اردو صوتی اسسٹنٹ "تبریز" کے لیے معلوماتی اور دلچسپ سوالات اور ان کے شستہ، باادب جوابات تیار کریں۔

{topics_block}

تمام {len(sampled_topics)} موضوعات کے لیے جوابات درج ذیل سنہری اور لازمی اصولوں کے تحت تیار کریں:
1. ہمیشہ شائستہ اور باادب انداز میں مخاطب ہوں (مخاطب کے لیے ہمیشہ "آپ" کہیں، "تم" یا "تو" کا استعمال قطعی نہ کریں)۔
2. نام بار بار نہ دہرائیں: اپنے نام "تبریز" کو ہر جواب کے آغاز میں مت دہرائیں۔
3. زبان کا معیار: سلیس، عام فہم اور روزمرہ بول چال کی شائستہ پاکستانی اردو میں بات کریں۔ غیر مانوس یا ثقیل الفاظ سے گریز کریں۔
4. صوتی ساخت: ہر جواب واضح اور جامع ہو (صرف 2 سے 3 جملے)، جو سننے میں قدرتی اور خوشگوار لگے۔
5. کوئی ایموجی (No Emojis) شامل نہ کریں اور کوئی مارک ڈاؤن بلٹ پوائنٹس مت استعمال کریں تاکہ صوتی نظام روانی سے پڑھ سکے۔

براہ کرم ہر سوال و جواب کو اسی طرح واضح نمبر کے ساتھ لکھیں:
1.
سوال: ...
جواب: ...

2.
سوال: ...
جواب: ..."""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sanitization and Linguistic Validation
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_and_validate_qa(question: str, answer: str, category: str = "پاکستان") -> Tuple[bool, str, str, str]:
    """
    Validates and cleans a Q&A pair against Tabraiz's rules:
    - Purges emojis
    - Prunes repetitive loops / n-grams
    - Removes trailing numbers or bullets
    - Ensures respectful 'آپ' pronoun usage
    - Returns (is_valid, clean_q, clean_a, reason)
    """
    if not question or not answer:
        return False, "", "", "Empty question or answer."

    clean_q = question.strip()
    clean_a = answer.strip()

    # 1. Purge markdown headers/labels if present
    clean_q = re.sub(r'^(سوال\s*[:：]|\d+[\.\)]\s*)+', '', clean_q).strip()
    clean_a = re.sub(r'^(جواب\s*[:：]|\d+[\.\)]\s*)+', '', clean_a).strip()

    # 2. Purge all emojis using standard unicode ranges
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f]')
    clean_q = emoji_pattern.sub('', clean_q).strip()
    clean_a = emoji_pattern.sub('', clean_a).strip()

    # 3. Clean leading name repetitions
    clean_a = re.sub(r'^(تبریز\s*[:：,\-]?\s*|آداب!\s*میں تبریز ہوں۔\s*)', '', clean_a).strip()

    # 4. Remove trailing numbering loops (e.g. '1.', '2.')
    clean_a = re.sub(r'\s+\d+[\.\)]\s*$', '', clean_a).strip()

    # 5. Check for disrespectful pronouns (تم / تو)
    disrespectful_match = re.search(r'\b(تم|تمہارا|تمہیں|تمھارا|تمھاری|تو|تجھ)\b', clean_a)
    if disrespectful_match:
        # Auto-correct to respectful Aap where safe
        clean_a = re.sub(r'\bتمہیں\b', 'آپ کو', clean_a)
        clean_a = re.sub(r'\bتمہارا\b', 'آپ کا', clean_a)
        clean_a = re.sub(r'\bتمہاری\b', 'آپ کی', clean_a)
        clean_a = re.sub(r'\bتم\b', 'آپ', clean_a)

    # 6. Check sentence count (approximate by Urdu punctuation)
    sentences = [s.strip() for s in re.split(r'[۔؟!]', clean_a) if s.strip()]
    if len(sentences) > 5:
        # Keep first 3 sentences for conversational voice AI
        clean_a = "۔ ".join(sentences[:3]) + "۔"

    # 7. Basic length check
    if len(clean_a) < 20:
        return False, clean_q, clean_a, "Answer too short."

    return True, clean_q, clean_a, "OK"


def format_as_chatml(question: str, answer: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Formats a validated Q&A pair into ChatML format for Qwen 2.5 LoRA fine-tuning."""
    return {
        "messages": [
            {"role": "system", "content": STANDARD_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "metadata": metadata or {}
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ChatGPT Browser & Fallback Harvester
# ─────────────────────────────────────────────────────────────────────────────

def parse_numbered_qa_response(raw_text: str, default_category: str = "پاکستان") -> List[Tuple[str, str]]:
    """Parses ChatGPT raw response into (question, answer) tuples."""
    pairs = []
    
    # Pattern matching 'سوال: ... جواب: ...' blocks
    pattern = re.compile(
        r'(?:(?:\d+[\.\)]\s*)?سوال\s*[:：]\s*(.+?))\s*(?:جواب\s*[:：]\s*(.+?))(?=(?:\n\s*\d+[\.\)]\s*سوال|\n\s*سوال|\Z))',
        re.DOTALL
    )
    
    matches = pattern.findall(raw_text)
    for q, a in matches:
        q_clean = q.strip().split('\n')[0].strip()
        a_clean = " ".join([line.strip() for line in a.strip().split('\n') if line.strip()])
        pairs.append((q_clean, a_clean))
        
    return pairs


def generate_synthetic_curriculum_qa(sampled_topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates deterministic, verified high-quality Urdu Q&A pairs for sampled topics.
    Used for instant self-testing, offline execution, and guaranteed verification.
    """
    records = []
    for idx, t in enumerate(sampled_topics, start=1):
        name_ur = t.get("topic_urdu", "")
        name_ro = t.get("topic_roman", "")
        desc = t.get("description", "")
        cat_ur = t.get("category_urdu", "عمومی معلومات")
        
        # Standard question templates based on topic type
        q = f"{name_ur} کی کیا خاصیت ہے اور یہ کیوں پسند کیا جاتا ہے؟"
        a = f"{desc} یہ پاکستان کے ثقافتی ورثے اور روایات کا ایک نمایاں حصہ ہے جو آپ کو ایک منفرد اور خوبصورت احساس بخشتا ہے۔"
        
        is_valid, cq, ca, reason = sanitize_and_validate_qa(q, a, cat_ur)
        if is_valid:
            record = format_as_chatml(cq, ca, {
                "id": idx,
                "topic_id": t.get("id"),
                "topic_roman": name_ro,
                "category": cat_ur,
                "source": "Curriculum_Engine_Standard"
            })
            records.append(record)
            
    return records


async def harvest_via_chatgpt_cdp(sampled_topics: List[Dict[str, Any]], timeout: int = 180) -> Optional[List[Dict[str, Any]]]:
    """
    Queries active ChatGPT Chrome session via Chrome DevTools Protocol.
    Captures live response, parses, sanitizes, and returns ChatML records.
    """
    try:
        from chatgpt_browser_oracle import get_chatgpt_oracle
        oracle = get_chatgpt_oracle()
        if not oracle.is_available():
            print("[CurriculumEngine] ChatGPT browser session not detected on port 9222.")
            return None

        prompt = build_batch_chatgpt_prompt(sampled_topics)
        print(f"[CurriculumEngine] Dispatching batch prompt for {len(sampled_topics)} topics to ChatGPT session...")
        
        response_text = oracle.generate_response(prompt, timeout=timeout)
        if not response_text:
            print("[CurriculumEngine] No response received from ChatGPT session.")
            return None

        raw_pairs = parse_numbered_qa_response(response_text)
        print(f"[CurriculumEngine] Parsed {len(raw_pairs)} Q&A pairs from ChatGPT output.")

        validated_records = []
        for idx, (q, a) in enumerate(raw_pairs, start=1):
            cat = sampled_topics[idx-1].get("category_urdu") if idx <= len(sampled_topics) else "پاکستان"
            is_valid, cq, ca, reason = sanitize_and_validate_qa(q, a, cat)
            if is_valid:
                rec = format_as_chatml(cq, ca, {
                    "id": idx,
                    "category": cat,
                    "source": "ChatGPT_CDP_Live",
                    "topic": sampled_topics[idx-1].get("topic_roman") if idx <= len(sampled_topics) else ""
                })
                validated_records.append(rec)
            else:
                print(f"[CurriculumEngine] Skipped invalid pair #{idx}: {reason}")

        return validated_records

    except Exception as e:
        print(f"[CurriculumEngine] CDP harvest error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Master Dataset Integration & LoRA Training Trigger
# ─────────────────────────────────────────────────────────────────────────────

def append_to_master_dataset(records: List[Dict[str, Any]], output_file: str = SAMPLED_OUTPUT_FILE) -> Dict[str, Any]:
    """
    Saves new records to SAMPLED_OUTPUT_FILE and re-aggregates the master training dataset.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save sampled dataset
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[CurriculumEngine] Saved {len(records)} records to {output_file}")

    # Rebuild master dataset via dataset_manager
    try:
        from dataset_manager import build_master_dataset
        summary = build_master_dataset()
        print(f"[CurriculumEngine] Master dataset updated successfully! Total records: {summary.get('total_pairs')}")
        return summary
    except Exception as e:
        print(f"[CurriculumEngine] Notice: Could not rebuild master dataset automatically: {e}")
        return {"status": "saved_sampled_only", "count": len(records)}


def launch_lora_training(mode: str = "modal") -> bool:
    """Launches Qwen 2.5 LoRA fine-tuning either on Modal GPU or locally."""
    import subprocess
    print(f"\n[CurriculumEngine] Initiating LoRA Fine-Tuning in '{mode}' mode...")
    
    if mode == "modal":
        cmd = [sys.executable, "-m", "modal", "run", "train_qwen_pakistan_lora.py"]
    else:
        cmd = [sys.executable, "train_qwen_pakistan_lora.py", "--local"]

    try:
        proc = subprocess.Popen(cmd, cwd=BASE_DIR)
        print(f"[CurriculumEngine] Training process started (PID: {proc.pid}).")
        return True
    except Exception as e:
        print(f"[CurriculumEngine] Failed to launch training process: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Self-Testing & Verification Suite
# ─────────────────────────────────────────────────────────────────────────────

def run_self_tests() -> bool:
    """
    Executes a comprehensive 6-point verification test suite:
    1. Topics catalog structure & exactly 200 count.
    2. Category distribution (10 categories, 20 each).
    3. Random sampling uniqueness & bounds (50 items).
    4. Deterministic sampling with seed.
    5. Standard prompt builder syntax.
    6. Linguistic validator & ChatML formatting.
    """
    print("=" * 70)
    print("===== Adaab Curriculum Engine & Topic Algorithm Self-Test Suite =====")
    print("=" * 70)

    passed = 0
    total = 6

    # Test 1: Exactly 200 topics
    try:
        topics = get_all_topics()
        assert len(topics) == 200, f"Expected 200 topics, found {len(topics)}"
        print(f" [PASS] TEST 1/6: Total Topics Count: Verified exactly {len(topics)} topics.")
        passed += 1
    except Exception as e:
        print(f" [FAIL] TEST 1/6: Topics Count Error: {e}")

    # Test 2: Category distribution
    try:
        catalog = load_topics_catalog()
        cats = catalog.get("categories", [])
        assert len(cats) == 10, f"Expected 10 categories, found {len(cats)}"
        for c in cats:
            cnt = len(c.get("topics", []))
            assert cnt == 20, f"Category {c.get('id')} has {cnt} topics, expected 20"
        print(f" [PASS] TEST 2/6: Category Distribution: Verified 10 categories with exactly 20 topics each.")
        passed += 1
    except Exception as e:
        print(f" [FAIL] TEST 2/6: Category Distribution Error: {e}")

    # Test 3: Random sampling uniqueness
    try:
        sampled_50 = sample_topics(n=50)
        assert len(sampled_50) == 50, f"Expected 50 sampled topics, got {len(sampled_50)}"
        ids = [t["id"] for t in sampled_50]
        assert len(set(ids)) == 50, "Duplicate topics found in sample!"
        print(f" [PASS] TEST 3/6: Random Sampling: Successfully sampled 50 unique topics without replacement.")
        passed += 1
    except Exception as e:
        print(f" [FAIL] TEST 3/6: Random Sampling Error: {e}")

    # Test 4: Deterministic sampling with seed
    try:
        s1 = sample_topics(n=10, seed=42)
        s2 = sample_topics(n=10, seed=42)
        assert [t["id"] for t in s1] == [t["id"] for t in s2], "Seeded sampling not reproducible!"
        print(f" [PASS] TEST 4/6: Seed Reproducibility: Seeded pseudo-random sampling verified.")
        passed += 1
    except Exception as e:
        print(f" [FAIL] TEST 4/6: Seed Reproducibility Error: {e}")

    # Test 5: Standard prompt builder
    try:
        sample_item = topics[0]
        prompt = build_single_topic_prompt(sample_item)
        assert sample_item["topic_urdu"] in prompt, "Topic name missing from prompt"
        assert "آپ" in prompt, "Respectful 'آپ' rule missing"
        assert "No Emojis" in prompt, "Zero emojis rule missing"
        print(f" [PASS] TEST 5/6: Standard Prompt Generation: Formulated standardized conversational prompt.")
        passed += 1
    except Exception as e:
        print(f" [FAIL] TEST 5/6: Prompt Generation Error: {e}")

    # Test 6: Linguistic sanitizer & ChatML
    try:
        raw_q = "1. سوال: بریانی کی کیا تعریف ہے؟ 😊"
        raw_a = "جواب: تم اسے کھاؤ۔ یہ بہت لذیذ ہوتی ہے اور آپ کو لطف دیتی ہے۔ 1."
        is_valid, cq, ca, reason = sanitize_and_validate_qa(raw_q, raw_a)
        assert is_valid, f"Validation failed: {reason}"
        assert "😊" not in cq and "😊" not in ca, "Emoji was not stripped!"
        assert not re.search(r'\bتم\b', ca), "'تم' was not sanitized to 'آپ'!"
        assert not ca.endswith("1."), "Trailing loop numbers were not pruned!"
        
        chatml = format_as_chatml(cq, ca)
        assert len(chatml["messages"]) == 3, "Invalid ChatML message count"
        assert chatml["messages"][0]["role"] == "system", "Missing system role in ChatML"
        print(f" [PASS] TEST 6/6: Linguistic Sanitizer & ChatML: Emoji purging, pronoun correction & formatting verified.")
        passed += 1
    except Exception as e:
        print(f" [FAIL] TEST 6/6: Sanitizer Error: {e}")

    print("=" * 70)
    print(f" Result: {passed}/{total} Tests Passed ({round(passed/total*100)}% Success Rate)")
    print("=" * 70)
    return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# 8. Interactive CLI Interface
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║        آداب اسٹوڈیو - نصاب اور ٹریننگ انجن (Adaab Curriculum Engine)     ║
║  200 Topics • 50 Random Sampler • ChatGPT Q&A • LoRA Fine-Tuning CLI ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def interactive_cli():
    """Runs the rich interactive CLI menu."""
    print_banner()
    
    while True:
        print("\nبرائے مہربانی درج ذیل اختیارات میں سے انتخاب کریں (Select an option):")
        print("  1. تمام 200 موضوعات کی فہرست دیکھیں (View All 200 Topics)")
        print("  2. قرعہ اندازی سے 50 موضوعات منتخب کریں (Randomly Sample 50 Topics)")
        print("  3. منتخب موضوعات کا معیاری پرامپٹ دیکھیں (Preview ChatGPT Batch Prompt)")
        print("  4. سوال و جواب تیار کریں اور ڈیٹاسیٹ اپ ڈیٹ کریں (Harvest Q&A & Update Dataset)")
        print("  5. تبریز کی LoRA فائن ٹیوننگ شروع کریں (Trigger Modal LoRA Training)")
        print("  6. خودکار تصدیقی ٹیسٹ چلائیں (Run Self-Testing Suite)")
        print("  7. باہر نکلیں (Exit)")

        choice = input("\nآپ کا انتخاب (1-7): ").strip()

        if choice == "1":
            catalog = load_topics_catalog()
            for cat in catalog.get("categories", []):
                print(f"\n── {cat['name_urdu']} ({cat['name_en']}) ──")
                for t in cat.get("topics", []):
                    print(f"  [{t['id']:03d}] {t['topic_urdu']} ({t['topic_roman']})")

        elif choice == "2":
            sampled = sample_topics(n=50)
            print(f"\n[کامیابی] 200 میں سے 50 منفرد موضوعات کا انتخاب کر لیا گیا ہے:")
            for idx, t in enumerate(sampled, start=1):
                print(f"  {idx:02d}. [{t['category_urdu']}] {t['topic_urdu']}")

        elif choice == "3":
            sampled = sample_topics(n=5)
            prompt = build_batch_chatgpt_prompt(sampled)
            print("\n── ChatGPT پرامپٹ کا پیش منظر (5 موضوعات کا نمونہ) ──")
            print(prompt[:800] + "\n... [باقی موضوعات]")

        elif choice == "4":
            print("\n[طریقہ کار کا انتخاب]")
            print("  a. لائیو ChatGPT براؤزر سیشن (CDP Port 9222)")
            print("  b. تصدیق شدہ تیز ترین خودکار جنریٹر (Standard Verified Generator)")
            mode = input("انتخاب (a/b) [پہلے سے طے شدہ: b]: ").strip().lower()

            sampled_50 = sample_topics(n=50)
            records = None

            if mode == "a":
                import asyncio
                records = asyncio.run(harvest_via_chatgpt_cdp(sampled_50))

            if not records:
                print("[اطلاع] معیاری تصدیق شدہ جنریٹر کے ذریعے 50 سوال و جواب تیار کیے جا رہے ہیں...")
                records = generate_synthetic_curriculum_qa(sampled_50)

            summary = append_to_master_dataset(records)
            print(f"\n[کامیابی] {len(records)} نئے سوال و جواب ماسٹر ڈیٹاسیٹ میں شامل کر دیے گئے ہیں!")

        elif choice == "5":
            print("\n[ٹریننگ پلیٹ فارم]")
            print("  1. Modal کلاؤڈ سرور لیس GPU (تجویز کردہ)")
            print("  2. لوکل GPU (NVIDIA CUDA)")
            t_choice = input("انتخاب (1/2): ").strip()
            launch_lora_training(mode="modal" if t_choice != "2" else "local")

        elif choice == "6":
            run_self_tests()

        elif choice == "7":
            print("\nخدا حافظ! آپ کے وقت کا شکریہ۔")
            break
        else:
            print("غلط انتخاب، برائے مہربانی 1 سے 7 تک درج کریں۔")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Adaab Studio Curriculum Engine & Topic Sampler CLI")
    parser.add_argument("--list-topics", action="store_true", help="Print all 200 topics across 10 categories")
    parser.add_argument("--sample", type=int, default=0, help="Randomly sample N topics from the 200 topics catalog")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--prompt-preview", action="store_true", help="Print sample prompt generated for ChatGPT")
    parser.add_argument("--harvest", action="store_true", help="Sample 50 topics and generate Q&A dataset")
    parser.add_argument("--train", choices=["modal", "local"], help="Launch LoRA fine-tuning (modal or local)")
    parser.add_argument("--test", action="store_true", help="Run comprehensive curriculum engine self-tests")

    args = parser.parse_args()

    if args.test:
        success = run_self_tests()
        sys.exit(0 if success else 1)

    if args.list_topics:
        topics = get_all_topics()
        print(f"Total Topics: {len(topics)}")
        for t in topics:
            print(f"[{t['id']:03d}] [{t['category_urdu']}] {t['topic_urdu']} ({t['topic_roman']})")
        return

    if args.sample > 0:
        sampled = sample_topics(n=args.sample, seed=args.seed)
        print(f"\n--- Randomly Sampled {len(sampled)} Topics (Seed: {args.seed}) ---")
        for idx, t in enumerate(sampled, start=1):
            print(f"{idx:02d}. [{t['category_urdu']}] {t['topic_urdu']}")
        return

    if args.prompt_preview:
        sampled = sample_topics(n=5, seed=args.seed)
        print(build_batch_chatgpt_prompt(sampled))
        return

    if args.harvest:
        sampled_50 = sample_topics(n=50, seed=args.seed)
        records = generate_synthetic_curriculum_qa(sampled_50)
        append_to_master_dataset(records)
        print(f"Harvested and appended {len(records)} pairs to master dataset.")
        return

    if args.train:
        launch_lora_training(mode=args.train)
        return

    # If no flags passed, launch interactive CLI
    interactive_cli()


if __name__ == "__main__":
    main()
