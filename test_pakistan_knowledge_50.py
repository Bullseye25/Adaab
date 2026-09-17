"""
test_pakistan_knowledge_50.py
─────────────────────────────────────────────────────────────────────────────
50 Comprehensive Pakistan General Knowledge & Cultural Questions Test Suite
Covers:
  1. Pakistani Cuisine & Food (بریانی، نہاری، سجی، چپلی کباب، حلیم)
  2. History of Pakistan (14 اگست 1947، قراردادِ لاہور، قائدِ اعظم، لیاقت علی خان، آئین 1973)
  3. National & Regional Languages (اردو، پنجابی، سندھی، پشتو، بلوچی و سرائیکی)
  4. Geo-Location & Physical Geography (کے ٹو، دریائے سندھ، صحرائے تھر، گوادر، چاروں صوبے)
  5. Weather & Climate (مون سون، سمندری ہوائیں، برف باری، سبی کی گرمی، بہار و خزاں)
  6. Sports & Legends (1992 ورلڈ کپ، پی ایس ایل، ہاکی اولمپکس، جہانگیر خان، ارشد ندیم)
  7. Best Telecommunication Brands (جاز، زونگ، یوفون، ٹیلی نار، پی ٹی سی ایل)
  8. Government Departments & Civic Services (نادرا، ایف بی آر، پی ٹی اے، اسٹیٹ بینک، واپڈا)
  9. National Leadership & Governance (وزیرِ اعظم، صدرِ پاکستان، پارلیمنٹ، سپریم کورٹ، خارجہ پالیسی)
  10. Culture, Art & Heritage (علامہ اقبال، فیض احمد فیض، موہنجو دڑو، شالامار باغ، مینارِ پاکستان)

Automated Validation & Dataset Extraction:
- Executes queries one-by-one through the Cognitive Orchestrator.
- Verifies complete, untruncated answers with zero emojis and zero artifacts.
- Exports validated Q&A pairs into 'H:/Adaab/data/pakistan_knowledge_dataset.jsonl'
  formatted in ChatML for Qwen 2.5 7B instruction fine-tuning.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import time
import re

# Ensure project root in sys.path
sys.path.insert(0, "H:/Adaab")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from orchestrator import get_orchestrator, TABRAIZ_ORCHESTRATOR_SYSTEM
from memory_engine import clear_conversation_memory, get_active_discussion_topic

# 50 Comprehensive Questions Categorized Across 10 Domains
PAKISTAN_QUESTIONS_50 = [
    # 1. Food & Cuisine
    {"category": "کھانے اور پکوان", "question": "پاکستان میں بریانی کی تاریخ اور اس کی مشہور اقسام کے بارے میں بتائیے۔"},
    {"category": "کھانے اور پکوان", "question": "کراچی اور لاہور کی روایتی نہاری میں کیا خاص فرق اور ذائقہ ہوتا ہے؟"},
    {"category": "کھانے اور پکوان", "question": "بلوچستان کی روایتی روش اور سجی کس طرح تیار کی جاتی ہے؟"},
    {"category": "کھانے اور پکوان", "question": "خیبر پختونخوا اور پشاور کے مشہور چپلی کباب میں کون سے خاص مصالحہ جات شامل ہوتے ہیں؟"},
    {"category": "کھانے اور پکوان", "question": "روایتی حلیم یا دلیم کی تیاری میں کون سے اہم اناج، دالیں اور گوشت استعمال ہوتے ہیں؟"},

    # 2. History of Pakistan
    {"category": "تاریخِ پاکستان", "question": "قیامِ پاکستان کے تاریخی اسباب اور 14 اگست 1947 کی اہمیت بیان کیجیے۔"},
    {"category": "تاریخِ پاکستان", "question": "23 مارچ 1940 کو پیش کی جانے والی قراردادِ لاہور کا بنیادی مقصد کیا تھا؟"},
    {"category": "تاریخِ پاکستان", "question": "بانیِ پاکستان قائدِ اعظم محمد علی جناح کے رہنما اصول اتحاد، ایمان اور نظم و ضبط کا کیا مفہوم ہے؟"},
    {"category": "تاریخِ پاکستان", "question": "پاکستان کے پہلے وزیرِ اعظم نوابزادہ لیاقت علی خان کی ریاست کے لیے اہم ترین خدمات کیا تھیں؟"},
    {"category": "تاریخِ پاکستان", "question": "1973 کے متفقہ آئینِ پاکستان کی بنیادی جمہوری خصوصیات کیا ہیں؟"},

    # 3. Languages
    {"category": "پاکستان کی زبانیں", "question": "پاکستان میں قومی زبان اردو کے علاوہ کون سی بڑی علاقائی زبانیں بولی جاتی ہیں؟"},
    {"category": "پاکستان کی زبانیں", "question": "پنجابی زبان کے مشہور صوفی شعراء جیسے وارث شاہ اور بلھے شاہ کا ادب میں کیا مقام ہے؟"},
    {"category": "پاکستان کی زبانیں", "question": "سندھی زبان کی قدامت اور اس کے منفرد 52 حروف پر مشتمل رسم الخط کی کیا خصوصیت ہے؟"},
    {"category": "پاکستان کی زبانیں", "question": "پشتو زبان کے کلاسیکی شاعر خوشحال خان خٹک اور رحمان بابا کا کیا ادبی مرتبہ ہے؟"},
    {"category": "پاکستان کی زبانیں", "question": "بلوچی اور سرائیکی زبانیں پاکستان کے کن علاقوں میں رائج ہیں اور ان کا ثقافتی پس منظر کیا ہے؟"},

    # 4. Geo-Location & Geography
    {"category": "جغرافیہ پاکستان", "question": "دنیا کی دوسری بلند ترین چوٹی کے ٹو (K2) پاکستان کے کس خطے میں واقع ہے اور اس کی اونچائی کیا ہے؟"},
    {"category": "جغرافیہ پاکستان", "question": "دریائے سندھ کی پاکستان کی زراعت، معیشت اور آبپاشی کے نظام میں کیا حیثیت ہے؟"},
    {"category": "جغرافیہ پاکستان", "question": "صحرائے تھر پاکستان کے کس صوبے میں واقع ہے اور اس کی جغرافیائی وسعت کتنی ہے؟"},
    {"category": "جغرافیہ پاکستان", "question": "گوادر کی گہرے پانی کی بندرگاہ اور سی پیک (CPEC) منصوبے کی تزویراتی اہمیت کیا ہے؟"},
    {"category": "جغرافیہ پاکستان", "question": "پاکستان کے چاروں صوبوں اور وفاقی دارالحکومت کے نام اور جغرافیائی محلِ وقوع بتائیے۔"},

    # 5. Weather & Climate
    {"category": "موسم اور ماحولیات", "question": "پاکستان میں مون سون بارشوں کا سلسلہ کن مہینوں میں شروع ہوتا ہے اور اس کے زرعی اثرات کیا ہیں؟"},
    {"category": "موسم اور ماحولیات", "question": "کراچی کی آب و ہوا پر بحیرہ عرب کی سمندری ہواؤں کا کیا اثر ہوتا ہے؟"},
    {"category": "موسم اور ماحولیات", "question": "شمالی علاقہ جات جیسے گلگت بلتستان، مری اور سوات میں موسمِ سرما کی برف باری کیسی ہوتی ہے؟"},
    {"category": "موسم اور ماحولیات", "question": "پاکستان کے گرم ترین مقامات جیسے سبی اور جیکب آباد میں گرمیوں میں درجہ حرارت کتنا بڑھ جاتا ہے؟"},
    {"category": "موسم اور ماحولیات", "question": "پاکستان میں موسمِ بہار اور موسمِ خزاں کے ایام کن مہینوں میں رونما ہوتے ہیں؟"},

    # 6. Sports & Legends
    {"category": "کھیل اور کھلاڑی", "question": "1992 کا کرکٹ ورلڈ کپ پاکستان نے کس طرح تاریخی طور پر اپنے نام کیا تھا؟"},
    {"category": "کھیل اور کھلاڑی", "question": "پاکستان سپر لیگ (PSL) کا آغاز کب ہوا اور اس کی نمایاں ٹیمیں کون سی ہیں؟"},
    {"category": "کھیل اور کھلاڑی", "question": "پاکستان کی قومی کھیل ہاکی کے اولمپکس اور ورلڈ کپ میں تاریخی اعزازات کیا رہے ہیں؟"},
    {"category": "کھیل اور کھلاڑی", "question": "جہانگیر خان اور جان شیر خان نے اسکواش کے عالمی مقابلوں میں کیا لازوال ریکارڈز قائم کیے؟"},
    {"category": "کھیل اور کھلاڑی", "question": "ارشد ندیم نے پیرس اولمپکس 2024 کے نیزہ بازی (جیولن تھرو) مقابلے میں کون سا ریکارڈ اور میڈل جیتا؟"},

    # 7. Telecommunication Brands
    {"category": "ٹیلی کام نیٹ ورکس", "question": "پاکستان میں موبائل نیٹ ورکس میں جاز (Jazz) کا صارفین کی تعداد کے لحاظ سے کیا مقام ہے؟"},
    {"category": "ٹیلی کام نیٹ ورکس", "question": "زونگ (Zong 4G) پاکستان میں موبائل انٹرنیٹ اور تیز رفتار ڈیٹا خدمات میں کس طرح نمایاں ہے؟"},
    {"category": "ٹیلی کام نیٹ ورکس", "question": "یوفون (Ufone 4G) کی خاص خدمات اور پی ٹی سی ایل گروپ کے ساتھ اس کے تعاون کی کیا تفصیل ہے؟"},
    {"category": "ٹیلی کام نیٹ ورکس", "question": "ٹیلی نار پاکستان (Telenor) کا نیٹ ورک کن دیہی اور شہری علاقوں میں زیادہ مقبول رہا ہے؟"},
    {"category": "ٹیلی کام نیٹ ورکس", "question": "پاکستان میں فکسڈ براڈ بینڈ اور آپٹک فائبر انفراسٹرکچر میں پی ٹی سی ایل (PTCL) کا کیا بنیادی کردار ہے؟"},

    # 8. Government Departments
    {"category": "سرکاری محکمے اور عوامی خدمات", "question": "نادرا (NADRA) کا ادارہ پاکستانی شہریوں کے کمپیوٹرائزڈ شناختی کارڈز اور بائیو میٹرک ریکارڈز میں کیا خدمات انجام دیتا ہے؟"},
    {"category": "سرکاری محکمے اور عوامی خدمات", "question": "فیڈرل بورڈ آف ریونیو (FBR) کا بنیادی مقصد اور پاکستان میں قومی ٹیکس وصولی کا نظام کیا ہے؟"},
    {"category": "سرکاری محکمے اور عوامی خدمات", "question": "پاکستان ٹیلی کمیونیکیشن اتھارٹی (PTA) موبائل ڈیوائس رجسٹریشن اور انٹرنیٹ ریگولیشن کے لیے کیا کام کرتی ہے؟"},
    {"category": "سرکاری محکمے اور عوامی خدمات", "question": "اسٹیٹ بینک آف پاکستان (SBP) ملک کی مانیٹری پالیسی اور بینکاری نظام کو کس طرح کنٹرول کرتا ہے؟"},
    {"category": "سرکاری محکمے اور عوامی خدمات", "question": "واپڈا (WAPDA) کے ڈیمز اور پانی کے ذخائر پاکستان کی ہائیڈرو پاور پیداوار میں کیا کردار ادا کرتے ہیں؟"},

    # 9. National Leadership & Politics
    {"category": "قومی قیادت اور ریاست", "question": "پاکستان کے موجودہ وزیرِ اعظم کون ہیں اور ان کا تعلق کس سیاسی جماعت سے ہے؟"},
    {"category": "قومی قیادت اور ریاست", "question": "صدرِ پاکستان کا آئینی کردار کیا ہوتا ہے اور ان کا انتخاب کس طرح عمل میں آتا ہے؟"},
    {"category": "قومی قیادت اور ریاست", "question": "پاکستان کی پارلیمان کے دونوں ایوانوں قومی اسمبلی اور سینیٹ کا باہمی جمہوری ڈھانچہ کیسا ہے؟"},
    {"category": "قومی قیادت اور ریاست", "question": "پاکستان کی سپریم کورٹ آف پاکستان اور چیف جسٹس کے آئینی اختیارات کیا ہیں؟"},
    {"category": "قومی قیادت اور ریاست", "question": "پاکستان کی خارجہ پالیسی کے بنیادی اصول اور پڑوسی ممالک کے ساتھ تعلقات کے رہنما خطوط کیا ہیں؟"},

    # 10. Culture, Art & Heritage
    {"category": "ثقافت اور قومی ورثہ", "question": "شاعرِ مشرق علامہ محمد اقبال کی شاعری میں خودی اور نوجوانوں کے لیے کیا انقلابی پیغام ہے؟"},
    {"category": "ثقافت اور قومی ورثہ", "question": "فیض احمد فیض کی شاعری اور ترقی پسند ادبی تحریک کے فکری اثرات کیا ہیں؟"},
    {"category": "ثقافت اور قومی ورثہ", "question": "وادیٔ سندھ کی قدیم تہذیب کے عالمی تاریخی آثار موہنجو دڑو اور ہڑپہ کی کیا اہمیت ہے؟"},
    {"category": "ثقافت اور قومی ورثہ", "question": "لاہور کا شالامار باغ اور بادشاہی مسجد مغل فنِ تعمیر کے کون سے شاہکار پہلو اجاگر کرتے ہیں؟"},
    {"category": "ثقافت اور قومی ورثہ", "question": "لاہور میں واقع مینارِ پاکستان کی تاریخی تعمیر کا کیا پس منظر اور اہمیت ہے؟"}
]

DATASET_FILE = "H:/Adaab/data/pakistan_knowledge_dataset.jsonl"

def run_pakistan_50_suite():
    print("=" * 75)
    print("==== Adaab Studio: 50 Comprehensive Pakistan Knowledge Questions Suite ====")
    print("=" * 75)

    orch = get_orchestrator()
    passed_count = 0
    dataset_records = []

    # Ensure clean slate for test
    clear_conversation_memory()

    for idx, item in enumerate(PAKISTAN_QUESTIONS_50, 1):
        cat = item["category"]
        q = item["question"]

        print(f"\n[{idx:02d}/50] زمرہ: {cat}")
        print(f"سوال: {q}")

        start_t = time.time()
        reply, _ = orch.orchestrate_turn(q, [], session_id=f"pk_test_{idx}")
        duration = time.time() - start_t

        # Quality & Style Validation
        is_clean = True
        reasons = []

        if not reply or len(reply.strip()) < 40:
            is_clean = False
            reasons.append(f"جواب بہت مختصر ہے ({len(reply)} حروف)")

        # Check for emojis
        if re.search(r"[\U00010000-\U0010ffff]", reply) or re.search(r"[\u2300-\u23ff\u2600-\u27bf\u2b50\u2b55]", reply):
            is_clean = False
            reasons.append("ایموجی پایا گیا (Emoji detected)")

        # Check for stray prefixes
        if re.search(r"^(?:assistant|text|tabraiz|tehzeeb|جواب|ٹیکسٹ)\s*[:：]", reply, re.IGNORECASE):
            is_clean = False
            reasons.append("کردار کا لیبل پایا گیا (Role prefix)")

        # Check for onboarding interrogatives (should never interrogate user)
        if any(w in reply for w in ["آپ کا نام کیا ہے", "اپنا پیشہ بتائیے", "کہاں سے ہیں"]):
            is_clean = False
            reasons.append("غیر ضروری ذاتی سوال پایا گیا")

        if is_clean:
            passed_count += 1
            print(f"جواب ({duration:.2f} سیکنڈ):\n{reply[:160]}...")
            print(f"نتیجہ: [کامیاب PASSED]")

            # Save to dataset record (ChatML format)
            dataset_records.append({
                "messages": [
                    {"role": "system", "content": TABRAIZ_ORCHESTRATOR_SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": reply}
                ],
                "metadata": {
                    "id": idx,
                    "category": cat,
                    "question": q,
                    "response_length": len(reply),
                    "latency_seconds": round(duration, 2)
                }
            })
        else:
            print(f"جواب:\n{reply}")
            print(f"نتیجہ: [ناکام FAILED] وجوہات: {', '.join(reasons)}")

        # Brief pause between requests to respect rate limits
        time.sleep(1.0)

    # Save to JSONL
    os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for rec in dataset_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 75)
    print(f"امتحانی نتائج: {passed_count}/50 کامیاب ({passed_count / len(PAKISTAN_QUESTIONS_50) * 100:.1f}% Success Rate)")
    print(f"فائن ٹیوننگ ڈیٹاسیٹ محفوظ کر دیا گیا: {DATASET_FILE}")
    print("=" * 75)

    return passed_count == len(PAKISTAN_QUESTIONS_50)

if __name__ == "__main__":
    success = run_pakistan_50_suite()
    sys.exit(0 if success else 1)
