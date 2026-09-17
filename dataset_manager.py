"""
dataset_manager.py
─────────────────────────────────────────────────────────────────────────────
Extensible Dataset Manager for Adaab Studio & Qwen 2.5 LoRA Fine-Tuning
─────────────────────────────────────────────────────────────────────────────
• Organizes, aggregates, and validates training datasets for continuous LoRA
  expansion (culture, food, sports, science, everyday Urdu conversation).
• Supports adding custom topics and Roman Urdu corrections interactively or
  via modular JSONL files in H:/Adaab/data/.
• Generates a validated master training dataset ready for Modal GPU training.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional

# UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MASTER_DATASET_FILE = os.path.join(DATA_DIR, "master_training_dataset.jsonl")
DEFAULT_DATASET_FILE = os.path.join(DATA_DIR, "pakistan_knowledge_dataset.jsonl")

STANDARD_SYSTEM_PROMPT = """آپ کا نام "تبریز" (Tabraiz) ہے۔ آپ ایک باوقار، شستہ، دوستانہ اور باادب پاکستانی اردو صوتی اسسٹنٹ اور رفیق ہیں۔
آپ کا تعلق "آداب" (Adaab Studio) سے ہے۔

سنہری اور لازمی اصول:
1. ہمیشہ شائستہ اور باادب انداز میں مخاطب ہوں (مخاطب کے لیے ہمیشہ "آپ" کہیں، "تم" یا "تو" کا استعمال قطعی نہ کریں)۔
2. نام بار بار نہ دہرائیں: اپنے نام "تبریز" کو ہر جملے کے آغاز میں مت دہرائیں۔
3. خالص صوتی اسسٹنٹ رویہ: سیدھا صارف کی بات کا سلیس، باادب اور مدلل جواب دیں۔
4. زبان کا معیار: سلیس، عام فہم اور روزمرہ بول چال کی شائستہ پاکستانی اردو میں گفتگو کریں۔ غیر مانوس یا ثقیل الفاظ سے گریز کریں۔
5. صوتی ساخت: جواب واضح اور جامع ہو (2 سے 3 جملے)، جو سننے میں قدرتی اور خوشگوار لگے۔
6. کوئی ایموجی (No Emojis) اور کوئی مارک ڈاؤن بولڈ/بلٹ پوائنٹس مت استعمال کریں تاکہ صوتی نظام روانی سے پڑھ سکے۔"""


def load_dataset_records(filepath: str) -> List[Dict[str, Any]]:
    """Loads and parses JSONL dataset file safely."""
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except Exception as e:
                print(f"[DatasetManager] Warning: Parse error on line {idx} in {os.path.basename(filepath)}: {e}")
    return records


def get_all_dataset_files() -> List[str]:
    """Finds all .jsonl dataset files in the data/ directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith(".jsonl") and f != "master_training_dataset.jsonl":
            files.append(os.path.join(DATA_DIR, f))
    return sorted(files)


def build_master_dataset() -> str:
    """
    Consolidates, deduplicates, and validates all modular dataset files in data/
    into master_training_dataset.jsonl. Returns path to master dataset.
    """
    files = get_all_dataset_files()
    if not files and os.path.exists(DEFAULT_DATASET_FILE):
        files = [DEFAULT_DATASET_FILE]

    seen_queries = set()
    master_records = []

    for fpath in files:
        records = load_dataset_records(fpath)
        for r in records:
            # Extract user query and assistant response
            user_msg = ""
            assistant_msg = ""
            for m in r.get("messages", []):
                if m.get("role") == "user":
                    user_msg = m.get("content", "").strip().lower()
                elif m.get("role") == "assistant":
                    assistant_msg = m.get("content", "").strip()

            # Reject records with missing answers or generic fallback text
            if not assistant_msg or "سمجھ لیا ہے" in assistant_msg or "کیا مزید مدد یا رہنمائی" in assistant_msg:
                continue

            if user_msg and user_msg not in seen_queries:
                seen_queries.add(user_msg)
                # Ensure system prompt conforms to Tabraiz guidelines
                if r.get("messages") and r["messages"][0]["role"] == "system":
                    r["messages"][0]["content"] = STANDARD_SYSTEM_PROMPT
                master_records.append(r)

    # Write master dataset
    with open(MASTER_DATASET_FILE, "w", encoding="utf-8") as f:
        for r in master_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return MASTER_DATASET_FILE


def add_qa_pair(
    question: str,
    answer: str,
    category: str = "عمومی گفتگو و معلومات",
    is_roman_urdu: bool = False,
    target_file: Optional[str] = None
) -> bool:
    """Adds a new question-answer training record to the dataset."""
    dest = target_file or DEFAULT_DATASET_FILE
    existing = load_dataset_records(dest)

    new_record = {
        "messages": [
            {"role": "system", "content": STANDARD_SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()}
        ],
        "metadata": {
            "id": len(existing) + 1,
            "category": category.strip(),
            "question": question.strip(),
            "response_length": len(answer.strip()),
            "language_pair": "Roman_Urdu_to_Nastaliq" if is_roman_urdu else "Urdu_to_Urdu"
        }
    }

    with open(dest, "a", encoding="utf-8") as f:
        f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

    # Rebuild master dataset
    build_master_dataset()
    return True


def show_dataset_summary():
    """Displays comprehensive statistics and category breakdown of the dataset."""
    master_file = build_master_dataset()
    records = load_dataset_records(master_file)

    categories = {}
    roman_count = 0
    total_tokens_approx = 0

    for r in records:
        meta = r.get("metadata", {})
        cat = meta.get("category", "General")
        categories[cat] = categories.get(cat, 0) + 1
        if meta.get("language_pair") == "Roman_Urdu_to_Nastaliq":
            roman_count += 1
        for m in r.get("messages", []):
            total_tokens_approx += len(m.get("content", "").split())

    print("\n" + "=" * 70)
    print(" Adaab LoRA Training Dataset Summary ".center(70, "="))
    print("=" * 70)
    print(f" Master File:        {master_file}")
    print(f" Total Q&A Turns:    {len(records)}")
    print(f" Roman Urdu Pairs:   {roman_count}")
    print(f" Standard Urdu Pairs:{len(records) - roman_count}")
    print(f" Approx Word Count:  {total_tokens_approx} words")
    print("-" * 70)
    print(" Categories Breakdown:")
    for cat, count in categories.items():
        print(f"   • {cat}: {count} records")
    print("=" * 70 + "\n")


def interactive_add_topic():
    """Interactive CLI wizard to add a new topic or training example."""
    print("\n" + "=" * 70)
    print(" Add New Topic / Training Pair to Adaab LoRA Dataset ".center(70, "="))
    print("=" * 70)
    cat = input("Category (e.g. Science, Tourism, Culture, Food) [General]: ").strip() or "عمومی معلومات"
    q = input("Question (in Urdu or Roman Urdu): ").strip()
    if not q:
        print("Question cannot be empty.")
        return
    ans = input("Answer (in polite, cultured Urdu): ").strip()
    if not ans:
        print("Answer cannot be empty.")
        return

    is_roman = any(c.isascii() and c.isalpha() for c in q)
    add_qa_pair(question=q, answer=ans, category=cat, is_roman_urdu=is_roman)
    print(f"\n✓ Successfully added record! Master dataset updated.")
    show_dataset_summary()


if __name__ == "__main__":
    show_dataset_summary()
