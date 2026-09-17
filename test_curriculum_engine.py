"""
test_curriculum_engine.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Automated Unit Test Suite for Adaab Curriculum Engine:
• 200 Pakistani Topics Catalog Verification (10 categories x 20 topics)
• Uniform Random Sampling (n=50) & Deduplication
• Standardized Prompt Generation for ChatGPT
• Linguistic Sanitizer, Loop Pruning & Respectful "آپ" Enforcement
• ChatML Format Compliance for Qwen 2.5 LoRA Training
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import unittest
import json
import re

# Set UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"

from pakistan_curriculum_engine import (
    load_topics_catalog,
    get_all_topics,
    sample_topics,
    build_single_topic_prompt,
    build_batch_chatgpt_prompt,
    sanitize_and_validate_qa,
    format_as_chatml,
    generate_synthetic_curriculum_qa,
    TOPICS_FILE
)


class TestPakistanCurriculumEngine(unittest.TestCase):

    def test_01_catalog_structure_and_total_count(self):
        """Verify topics catalog contains exactly 200 topics across 10 categories."""
        self.assertTrue(os.path.exists(TOPICS_FILE), "Topics catalog file missing.")
        catalog = load_topics_catalog(TOPICS_FILE)
        categories = catalog.get("categories", [])
        self.assertEqual(len(categories), 10, f"Expected 10 categories, found {len(categories)}")
        
        all_topics = get_all_topics(TOPICS_FILE)
        self.assertEqual(len(all_topics), 200, f"Expected exactly 200 topics, found {len(all_topics)}")
        
        # Verify IDs are 1 to 200 without gaps or duplicates
        ids = [t["id"] for t in all_topics]
        self.assertEqual(len(set(ids)), 200, "Topic IDs contain duplicates!")
        self.assertEqual(min(ids), 1)
        self.assertEqual(max(ids), 200)

    def test_02_each_category_contains_twenty_topics(self):
        """Verify each category has exactly 20 topics with Urdu and Roman titles."""
        catalog = load_topics_catalog(TOPICS_FILE)
        for cat in catalog.get("categories", []):
            cat_id = cat.get("id")
            topics = cat.get("topics", [])
            self.assertEqual(len(topics), 20, f"Category '{cat_id}' does not have 20 topics (has {len(topics)}).")
            for t in topics:
                self.assertTrue(bool(t.get("topic_urdu")), f"Missing Urdu title in topic #{t.get('id')}")
                self.assertTrue(bool(t.get("topic_roman")), f"Missing Roman title in topic #{t.get('id')}")
                self.assertTrue(bool(t.get("description")), f"Missing description in topic #{t.get('id')}")

    def test_03_random_sampling_algorithm_size_and_uniqueness(self):
        """Verify sampling algorithm produces exactly 50 unique topics without duplicates."""
        sample_50 = sample_topics(n=50)
        self.assertEqual(len(sample_50), 50, "Sample size is not 50.")
        
        ids = [t["id"] for t in sample_50]
        self.assertEqual(len(set(ids)), 50, "Sample contains duplicate topics.")
        
        # Verify sampled items are valid subsets of the 200 topics
        for item in sample_50:
            self.assertIn(item["id"], range(1, 201))

    def test_04_seeded_sampling_is_deterministic(self):
        """Verify seed parameter guarantees reproducible random sampling."""
        s1 = sample_topics(n=50, seed=1947)
        s2 = sample_topics(n=50, seed=1947)
        self.assertEqual([t["id"] for t in s1], [t["id"] for t in s2])
        
        # Different seed produces different ordering
        s3 = sample_topics(n=50, seed=2026)
        self.assertNotEqual([t["id"] for t in s1], [t["id"] for t in s3])

    def test_05_single_topic_prompt_formulation(self):
        """Verify single topic prompt contains topic details and Tabraiz's rules."""
        topic = get_all_topics()[0]
        prompt = build_single_topic_prompt(topic)
        self.assertIn(topic["topic_urdu"], prompt)
        self.assertIn("آپ", prompt, "Prompt must instruct respectful pronoun 'آپ'.")
        self.assertIn("No Emojis", prompt, "Prompt must enforce zero emojis.")
        self.assertIn("2 سے 3 جملے", prompt, "Prompt must constrain length to 2-3 sentences.")

    def test_06_batch_chatgpt_prompt_formulation(self):
        """Verify batch prompt formats 50 sampled topics cleanly for ChatGPT."""
        sampled = sample_topics(n=50, seed=123)
        batch_prompt = build_batch_chatgpt_prompt(sampled)
        self.assertIn("50 پاکستانی موضوعات", batch_prompt)
        self.assertIn("1.", batch_prompt)
        self.assertIn("50.", batch_prompt)
        self.assertIn("No Emojis", batch_prompt)

    def test_07_linguistic_sanitizer_removes_emojis(self):
        """Verify emojis are completely stripped from questions and answers."""
        raw_q = "کیا بریانی میں آلو ہونا چاہیے؟ 🍚✨😋"
        raw_a = "سندھی بریانی میں آلو لازمی ہوتا ہے اور آپ کو اس کا ذائقہ بہت پسند آئے گا۔ ❤️"
        is_valid, clean_q, clean_a, reason = sanitize_and_validate_qa(raw_q, raw_a)
        self.assertTrue(is_valid)
        self.assertNotIn("🍚", clean_q)
        self.assertNotIn("✨", clean_q)
        self.assertNotIn("😋", clean_q)
        self.assertNotIn("❤️", clean_a)

    def test_08_linguistic_sanitizer_enforces_respectful_pronouns(self):
        """Verify disrespectful pronouns ('تم', 'تمہیں') are sanitized to 'آپ'."""
        raw_q = "آپ کو کون سا کھیل پسند ہے؟"
        raw_a = "تم جب کرکٹ دیکھو تو تمہیں 1992 کا ورلڈ کپ یاد رکھنا چاہیے جو پاکستان نے جیتا تھا۔"
        is_valid, clean_q, clean_a, reason = sanitize_and_validate_qa(raw_q, raw_a)
        self.assertTrue(is_valid)
        self.assertNotIn("تمہیں", clean_a)
        self.assertIn("آپ کو", clean_a)
        self.assertIn("آپ", clean_a)

    def test_09_linguistic_sanitizer_prunes_trailing_loops(self):
        """Verify trailing numbers like '1.' or repeated names are pruned."""
        raw_q = "سوال: لاہوری چرغہ کیسے بنتا ہے؟"
        raw_a = "جواب: تبریز: یہ مسالے دار مرغ ہوتا ہے جو آپ کو بہت پسند آئے گا۔ 1."
        is_valid, clean_q, clean_a, reason = sanitize_and_validate_qa(raw_q, raw_a)
        self.assertTrue(is_valid)
        self.assertFalse(clean_a.endswith("1."))
        self.assertFalse(clean_a.startswith("تبریز:"))

    def test_10_chatml_formatting_and_schema(self):
        """Verify Q&A pair formats properly into ChatML structure for LoRA training."""
        chatml = format_as_chatml(
            "سندھی بریانی کی کیا پہچان ہے؟",
            "سندھی بریانی اپنے چٹپٹے مسالوں اور آلو بخارے کی وجہ سے جانی جاتی ہے۔",
            {"category": "کھانے اور پکوان", "id": 1}
        )
        self.assertIn("messages", chatml)
        self.assertEqual(len(chatml["messages"]), 3)
        self.assertEqual(chatml["messages"][0]["role"], "system")
        self.assertEqual(chatml["messages"][1]["role"], "user")
        self.assertEqual(chatml["messages"][2]["role"], "assistant")
        self.assertIn("تبریز", chatml["messages"][0]["content"])
        self.assertEqual(chatml["metadata"]["id"], 1)

    def test_11_synthetic_qa_generation_for_50_topics(self):
        """Verify deterministic synthetic generation generates 50 valid ChatML pairs."""
        sampled_50 = sample_topics(n=50, seed=99)
        records = generate_synthetic_curriculum_qa(sampled_50)
        self.assertEqual(len(records), 50)
        for r in records:
            self.assertEqual(len(r["messages"]), 3)
            # Ensure "آپ" is used in every single answer
            self.assertIn("آپ", r["messages"][2]["content"])


if __name__ == "__main__":
    unittest.main()
