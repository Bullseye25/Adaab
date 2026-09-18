# -*- coding: utf-8 -*-
import unittest
import re
from orchestrator import CognitiveOrchestrator, get_tabraiz_system_prompt

class TestUrduLinguisticPurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = CognitiveOrchestrator()

    def test_01_leadership_president_pakistan(self):
        fact = self.orchestrator.retrieve_local_pakistan_knowledge('who is the current president of pakistan')
        self.assertIsNotNone(fact)
        self.assertTrue('آصف علی زرداری' in fact or 'زرداری' in fact)
        self.assertTrue('14ویں صدر' in fact or 'صدر' in fact)

    def test_02_leadership_prime_minister_pakistan(self):
        fact = self.orchestrator.retrieve_local_pakistan_knowledge('who is the priminister of pakistan')
        self.assertIsNotNone(fact)
        self.assertTrue('شہباز شریف' in fact)

    def test_03_leadership_army_chief_pakistan(self):
        fact = self.orchestrator.retrieve_local_pakistan_knowledge('who is the current army chief')
        self.assertIsNotNone(fact)
        self.assertTrue('عاصم منیر' in fact)

    def test_04_leadership_chief_justice_pakistan(self):
        fact = self.orchestrator.retrieve_local_pakistan_knowledge('who is the current chief justice')
        self.assertIsNotNone(fact)
        self.assertTrue('یحییٰ آفریدی' in fact)

    def test_05_clean_voice_text_preserves_urdu_che(self):
        raw = 'مارچ 2024 میں آرمی چیف نے پوچھا کہ کیا چاہیے'
        cleaned = self.orchestrator.clean_voice_text(raw)
        self.assertEqual(cleaned, raw)

    def test_06_clean_voice_text_strips_cyrillic_and_pashto(self):
        raw = 'یہ پر\u0435\u0445ز کریں اور استعمال گرد\u06d0'
        cleaned = self.orchestrator.clean_voice_text(raw)
        self.assertNotIn('\u0435', cleaned)
        self.assertNotIn('\u0445', cleaned)
        self.assertNotIn('\u06d0', cleaned)

    def test_07_clean_voice_text_strips_prompt_headers(self):
        raw = '### پہلا حصہ (صوتی کلام)\nآپ کی تحریر تیار ہے۔\n### دوسرا حصہ (کاپی کے لیے مارک ڈاؤن بلاک)\nیہ مضمون ہے۔'
        cleaned = self.orchestrator.clean_voice_text(raw)
        self.assertNotIn('پہلا حصہ', cleaned)
        self.assertNotIn('دوسرا حصہ', cleaned)
        self.assertIn('آپ کی تحریر تیار ہے۔', cleaned)
        self.assertIn('یہ مضمون ہے۔', cleaned)

    def test_08_clean_voice_text_strips_glued_latin_letters(self):
        raw = 'انہوں نے کہa کہ وہ کر سکta ہیں'
        cleaned = self.orchestrator.clean_voice_text(raw)
        self.assertNotIn('کہa', cleaned)
        self.assertNotIn('سکta', cleaned)

    def test_09_intent_classification_roman_urdu_article(self):
        intent = self.orchestrator.classify_intent('kia tum koi acha sa articl likh sakte ho')
        self.assertEqual(intent, 'CODE_OR_ARTICLE')

    def test_10_mode_resolution_roman_urdu_article(self):
        mode = self.orchestrator.resolve_prompt_mode('kia tum koi acha sa articl likh sakte ho')
        self.assertEqual(mode, 'CODE_TECH')

    def test_11_mode_resolution_casual_angry_article(self):
        mode = self.orchestrator.resolve_prompt_mode(
            'abay kia pagal hogaya hai. kia bolay ja rha hai.. mene kaha k koi acha sa articl lik k do jis mai nature ki bat horhi ho'
        )
        self.assertEqual(mode, 'CODE_TECH')

    def test_12_system_prompt_enforces_polite_friend_persona(self):
        prompt = get_tabraiz_system_prompt('CASUAL_VOICE')
        self.assertIn('آپ', prompt)
        self.assertIn('تم', prompt)
        self.assertIn('قطعی اور سختی سے ممنوع', prompt)
        self.assertIn('جی بالکل', prompt)

if __name__ == '__main__':
    unittest.main()
