"""
test_anti_looping_and_fluency.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive Unit Test Suite for Urdu Naturalness, Anti-Looping Decoding,
Dataset Sanitization, and Persona Etiquette in Adaab Studio.
─────────────────────────────────────────────────────────────────────────────
Covers:
1. User failure case loop-pruning & trailing digit stripping.
2. Cross-clause and intra-clause n-gram repetition pruning.
3. Master and Pakistan knowledge dataset cleanliness (zero corrupted fallbacks, zero emojis).
4. vLLM / Modal sampling parameter defaults (temperature, presence/frequency penalty).
5. Clean voice text sanitization for neural TTS.
6. TABRAIZ_ORCHESTRATOR_SYSTEM adab rules (respectful "آپ", forbidden "تم/تو", few-shot turns).
7. Pakistani food knowledge retrieval integrity.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import re
import unittest
import inspect

# Ensure UTF-8 console output in Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from orchestrator import CognitiveOrchestrator, TABRAIZ_ORCHESTRATOR_SYSTEM
from backend import AdaabClient
from dataset_manager import (
    DATA_DIR,
    MASTER_DATASET_FILE,
    DEFAULT_DATASET_FILE,
    load_dataset_records,
    build_master_dataset,
)


class TestAntiLoopingAndFluency(unittest.TestCase):
    def setUp(self):
        self.orchestrator = CognitiveOrchestrator()

    # ── TEST 1: User Reported Failure Case Loop Pruning ──────────────────────
    def test_user_reported_loop_failure_case(self):
        """
        Tests the exact repetitive output reported by the user when asking:
        'Pakistan mai famous kon kon se khane hain?'
        Original buggy text:
        'پاکستان میں کئی مشہور کھانے ہیں۔ یہ کھانے پاکستان کے مختلف علاقے میں پرچم پر پڑھا جاتا ہے
        اور ایک ایسی تجربہ کا اہم عہدہ ہے جس میں میز کی میز کے ساتھ کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے۔
        یہ کھانے کی میز کی میز کے ساتھ کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے۔
        یہ کھانے کی میز کی میز کے ساتھ کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے۔ 1.'
        """
        buggy_text = (
            "پاکستان میں کئی مشہور کھانے ہیں۔ یہ کھانے پاکستان کے مختلف علاقے میں پرچم پر پڑھا جاتا ہے "
            "اور ایک ایسی تجربہ کا اہم عہدہ ہے جس میں میز کی میز کے ساتھ کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے۔ "
            "یہ کھانے کی میز کی میز کے ساتھ کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے۔ "
            "یہ کھانے کی میز کی میز کے ساتھ کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے۔ 1."
        )

        cleaned = self.orchestrator.detect_and_prune_loops(buggy_text)

        # 1. Assert trailing standalone numbers are stripped
        self.assertFalse(cleaned.endswith("1."), "Trailing '1.' must be stripped")
        self.assertFalse(bool(re.search(r'\s+\d+[\.\:\-]?$', cleaned)), "Must not end with lone digits")

        # 2. Assert the repeated clause appears at most once
        rep_phrase = "کھانے کا ایک مشرقی طرز پر تجربہ کیا جاتا ہے"
        count = cleaned.count(rep_phrase)
        self.assertLessEqual(count, 1, f"Repeated phrase should occur at most once, but found {count} times")

        # 3. Assert total length is significantly curtailed
        self.assertLess(len(cleaned), len(buggy_text) * 0.75, "Cleaned text should prune duplicated run-ons")

    # ── TEST 2: Multi-Sentence and Intra-Clause Repetitions ───────────────────
    def test_sentence_and_intra_clause_repetition_pruning(self):
        """Tests pruning of duplicate sentences, stutter loops, and hallucinated list tails."""
        # Exact duplicate sentences
        dup_sentences = "لاہور ایک تاریخی اور زندہ دل شہر ہے۔ لاہور ایک تاریخی اور زندہ دل شہر ہے۔ لاہور ایک تاریخی اور زندہ دل شہر ہے۔"
        cleaned_dup = self.orchestrator.detect_and_prune_loops(dup_sentences)
        self.assertEqual(cleaned_dup.count("لاہور ایک تاریخی اور زندہ دل شہر ہے"), 1)

        # Hallucinated trailing numbers / list indicators
        list_hallucination = "کراچی کی بریانی اپنے چٹپٹے ذائقے کے لیے مشہور ہے۔ 1. 2. 3. 4."
        cleaned_list = self.orchestrator.detect_and_prune_loops(list_hallucination)
        self.assertFalse(any(num in cleaned_list for num in ["1.", "2.", "3.", "4."]))
        self.assertTrue(cleaned_list.endswith("ہے۔"))

    # ── TEST 3: Dataset Cleanliness & Zero Poisoned Fallbacks ─────────────────
    def test_dataset_cleanliness_and_no_corrupted_fallbacks(self):
        """
        Verifies that datasets contain zero fallback error strings,
        zero emojis, valid JSON structure, and authentic Urdu text.
        """
        build_master_dataset()
        dataset_files = [DEFAULT_DATASET_FILE, MASTER_DATASET_FILE]

        forbidden_phrases = [
            "معذرت خواہ ہوں، اس وقت براہ راست انٹرنیٹ سے منسلک ہونے میں دشواری ہے",
            "سمجھ لیا ہے",
            "کیا مزید مدد یا رہنمائی درکار ہے",
        ]
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff\u2600-\u27bf]')

        for fpath in dataset_files:
            if not os.path.exists(fpath):
                continue
            records = load_dataset_records(fpath)
            self.assertGreater(len(records), 0, f"{fpath} should not be empty")

            for idx, record in enumerate(records, start=1):
                self.assertIn("messages", record, f"Line {idx} in {fpath} missing 'messages'")
                messages = record["messages"]
                self.assertGreaterEqual(len(messages), 2, f"Line {idx} must have at least user and assistant")

                for msg in messages:
                    content = msg.get("content", "")
                    # Check forbidden fallback phrases
                    for phrase in forbidden_phrases:
                        self.assertNotIn(
                            phrase,
                            content,
                            f"Corrupted fallback string found in {os.path.basename(fpath)} line {idx}"
                        )
                    # Check zero emojis
                    self.assertFalse(
                        bool(emoji_pattern.search(content)),
                        f"Emoji detected in {os.path.basename(fpath)} line {idx}: {content[:50]}"
                    )

                # Check assistant response quality
                assistant_turns = [m for m in messages if m.get("role") == "assistant"]
                self.assertGreater(len(assistant_turns), 0, f"Line {idx} missing assistant turn")
                asst_content = assistant_turns[0].get("content", "")
                # Must have Arabic/Urdu unicode characters
                self.assertTrue(
                    bool(re.search(r'[\u0600-\u06ff]', asst_content)),
                    f"Assistant response in line {idx} must contain Urdu script"
                )

    # ── TEST 4: Backend Sampling Parameters Defaults ─────────────────────────
    def test_backend_sampling_parameters_for_loop_prevention(self):
        """
        Verifies that AdaabClient defaults avoid greedy token traps (temperature ~0.65)
        and enforce repetition, presence, and frequency penalties.
        """
        client = AdaabClient()
        sig = inspect.signature(client.chat_completion)
        params = sig.parameters

        # Temperature: between 0.60 and 0.70 (not 0.35 which causes greedy token traps)
        temp_default = params["temperature"].default
        self.assertGreaterEqual(temp_default, 0.60, "Temperature should be >= 0.60 to avoid greedy looping traps")
        self.assertLessEqual(temp_default, 0.75, "Temperature should be <= 0.75 to maintain factual coherence")

        # Repetition penalty: >= 1.15
        rep_default = params["repetition_penalty"].default
        self.assertGreaterEqual(rep_default, 1.15, "Repetition penalty must be >= 1.15")

        # Presence and Frequency penalties: >= 0.4
        pres_default = params["presence_penalty"].default
        freq_default = params["frequency_penalty"].default
        self.assertGreaterEqual(pres_default, 0.4, "Presence penalty must be >= 0.4")
        self.assertGreaterEqual(freq_default, 0.4, "Frequency penalty must be >= 0.4")

    # ── TEST 5: Clean Voice Text Pipeline for Neural TTS ──────────────────────
    def test_clean_voice_text_pipeline(self):
        """
        Verifies that clean_voice_text purges emojis, markdown formatting,
        HTML wrappers, role prefixes, and trailing digits.
        """
        sample_input = (
            '<div class="bot-msg-text">'
            'Assistant: **آداب!** 🇵🇰 پاکستان کے کھانے بہت لذیذ ہیں۔ '
            'کراچی کی بریانی اور لاہور کی نہاری دنیا بھر میں مشہور ہیں۔ 🍲 1.'
            '</div>'
        )

        cleaned = self.orchestrator.clean_voice_text(sample_input)

        self.assertNotIn("Assistant:", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("<div", cleaned)
        self.assertNotIn("🇵🇰", cleaned)
        self.assertNotIn("🍲", cleaned)
        self.assertFalse(cleaned.endswith("1."))
        self.assertTrue("بریانی" in cleaned and "نہاری" in cleaned)

    # ── TEST 6: Persona Etiquette and Few-Shot Prompt Rules ──────────────────
    def test_persona_system_prompt_rules(self):
        """
        Verifies that TABRAIZ_ORCHESTRATOR_SYSTEM mandates respectful 'آپ',
        forbids 'تم' or 'تو', forbids emojis, and contains few-shot examples.
        """
        prompt = TABRAIZ_ORCHESTRATOR_SYSTEM

        # 1. Must mention respectful 'آپ'
        self.assertIn("آپ", prompt)

        # 2. Must explicitly forbid 'تم' or 'تو'
        self.assertTrue("تم" in prompt and "تو" in prompt, "Prompt must explicitly forbid 'تم' and 'تو'")

        # 3. Must mandate zero emojis
        self.assertTrue("ایموجی" in prompt or "emojis" in prompt.lower())

        # 4. Must include few-shot conversational pairs
        self.assertIn("Few-Shot", prompt)
        self.assertIn("صارف:", prompt)
        self.assertIn("تبریز:", prompt)

    # ── TEST 7: Pakistan Food Knowledge Integrity ────────────────────────────
    def test_pakistan_food_knowledge_retrieval(self):
        """
        Verifies that dataset query for 'Pakistan mai famous kon kon se khane hain?'
        returns high-quality, authentic food information without generic text.
        """
        records = load_dataset_records(DEFAULT_DATASET_FILE)
        food_records = [
            r for r in records
            if any("کھانے" in m.get("content", "") or "khane" in m.get("content", "").lower() for m in r.get("messages", []))
        ]
        self.assertGreater(len(food_records), 0, "Must have food records in knowledge base")

        matched_entry = None
        for r in food_records:
            user_msg = ""
            for m in r.get("messages", []):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
            if "famous kon kon se khane" in user_msg.lower():
                matched_entry = r
                break

        self.assertIsNotNone(matched_entry, "Exact user question must exist in knowledge base")
        assistant_content = ""
        for m in matched_entry.get("messages", []):
            if m.get("role") == "assistant":
                assistant_content = m.get("content", "")

        self.assertIn("بریانی", assistant_content)
        self.assertIn("نہاری", assistant_content)
        self.assertIn("کباب", assistant_content)
        self.assertNotIn("معذرت خواہ ہوں", assistant_content)


if __name__ == "__main__":
    unittest.main()
