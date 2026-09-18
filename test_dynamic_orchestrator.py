import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import (
    CognitiveOrchestrator,
    get_orchestrator,
    DYNAMIC_PRE_STANDARD_DIRECTIVES,
    get_tabraiz_system_prompt,
    TABRAIZ_PRE_STANDARD_CASUAL,
    TABRAIZ_PRE_STANDARD_DEEP,
    TABRAIZ_PRE_STANDARD_CURRENT_AFFAIRS,
    TABRAIZ_PRE_STANDARD_IMAGE,
    TABRAIZ_PRE_STANDARD_CODE,
    TABRAIZ_PRE_STANDARD_POETRY,
)

class TestDynamicPreStandardModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orch = get_orchestrator()

    def test_dynamic_directives_registry(self):
        expected_modes = [
            'CASUAL_VOICE',
            'DEEP_EXPLORATION',
            'CURRENT_AFFAIRS',
            'IMAGE_TURBO',
            'CODE_TECH',
            'ADAB_POETRY'
        ]
        for m in expected_modes:
            self.assertIn(m, DYNAMIC_PRE_STANDARD_DIRECTIVES, f'Mode {m} must be in DYNAMIC_PRE_STANDARD_DIRECTIVES')
            directive = DYNAMIC_PRE_STANDARD_DIRECTIVES[m]
            self.assertIsInstance(directive, str)
            self.assertGreater(len(directive), 50, f'Directive for {m} must be substantive')
            self.assertIn('آپ', directive, f'Directive {m} must mandate polite address with آپ')
            self.assertFalse(any(ord(c) > 0x1F600 for c in directive), f'Directive {m} must not have emojis')

    def test_get_tabraiz_system_prompt(self):
        for mode_name, directive in DYNAMIC_PRE_STANDARD_DIRECTIVES.items():
            sys_prompt = get_tabraiz_system_prompt(mode_name)
            self.assertIn('تبریز', sys_prompt)
            self.assertIn(directive, sys_prompt)
            self.assertIn('آداب', sys_prompt)

    def test_casual_voice_greetings(self):
        queries = [
            'السلام علیکم کیسے ہیں آپ؟',
            'ہیلو تبریز! کیا حال ہے؟',
            'Hi, how are you doing today?',
            'سب خیریت ہے؟',
            'کیسے مزاج ہیں آپ کے؟'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CASUAL_VOICE', f'Query {q} should resolve to CASUAL_VOICE, got {mode}')

    def test_casual_voice_daily_dialogue(self):
        queries = [
            'آج کا دن کیسا گزر رہا ہے؟',
            'کراچی کا موسم کیسا ہے؟',
            'آپ کون ہیں اور آپ کیا کر سکتے ہیں؟',
            'کیا آپ میری مدد کر سکتے ہیں؟'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CASUAL_VOICE', f'Query {q} should resolve to CASUAL_VOICE, got {mode}')

    def test_deep_exploration_explicit_urdu(self):
        queries = [
            'پاکستان میں انفارمیشن ٹیکنالوجی کے مستقبل پر تفصیل سے روشنی ڈالیں',
            'مصنوعی ذہانت اور مشین لرننگ میں کیا فرق ہے، تفصیل سے سمجھائیں',
            'اس معاملے کی پوری تفصیل بتائیں تاکہ میں سمجھ سکوں',
            'نیورل نیٹ ورک کیسے کام کرتا ہے، آسان الفاظ میں وضاحت کریں',
            'اردو ادب کی تاریخ کا گہرائی سے جائزہ پیش کیجیے'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'DEEP_EXPLORATION', f'Query {q} should resolve to DEEP_EXPLORATION, got {mode}')

    def test_deep_exploration_english_and_roman(self):
        queries = [
            'Can you explain in detail how quantum computing works?',
            'Please provide a detailed explanation of black holes',
            'Tell me more about the Indus Valley Civilization in depth',
            'mujhe is baare mai detail se samjhao',
            'tafseel se batao ke yeh kaisa hota hai'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'DEEP_EXPLORATION', f'Query {q} should resolve to DEEP_EXPLORATION, got {mode}')

    def test_deep_exploration_followup_transition(self):
        followups = [
            ('اور بتاؤ', 'قائد اعظم کے 14 نکات', 'قائد اعظم کے نکات کیا تھے؟'),
            ('مزید بتائیں', 'پاکستان کا خلائی مشن', 'سپارکو کے بارے میں بتائیں'),
            ('اور تفصیل سے بتائیں', 'موہنجودڑو', 'موہنجودڑو کی تاریخ کیا ہے؟'),
            ('aur batao na', 'artificial intelligence', 'what is ai?'),
            ('aage batao phir kya hua', '1965 کی جنگ', '1965 کی جنگ کے بارے میں بتاؤ'),
            ('mazeed batao', 'climate change', 'tell me about global warming')
        ]
        for user_text, topic, last_q in followups:
            mode = self.orch.resolve_prompt_mode(
                user_text=user_text,
                active_topic=topic,
                last_query=last_q
            )
            self.assertEqual(
                mode, 'DEEP_EXPLORATION',
                f'Follow-up {user_text} on topic {topic} should upgrade to DEEP_EXPLORATION, got {mode}'
            )

    def test_current_affairs_news_and_events(self):
        queries = [
            'آج کی تازہ ترین خبریں کیا ہیں؟',
            'پاکستان کے حالات حاضرہ کے بارے میں بتائیں',
            'What are the latest breaking news headlines today?',
            'تازہ ترین صورتحال کیا ہے؟'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CURRENT_AFFAIRS', f'Query {q} should resolve to CURRENT_AFFAIRS, got {mode}')

    def test_current_affairs_economy_and_markets(self):
        queries = [
            'پاکستان میں ڈالر کا ریٹ اور روپیہ کی کیا قیمت ہے؟',
            'پاکستان اسٹاک مارکیٹ اور معیشت کی تازہ ترین رپورٹ',
            'ملک میں مہنگائی اور پیٹرول کی قیمت کا کیا احوال ہے؟',
            'What is the current inflation and gold rate in Pakistan?'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CURRENT_AFFAIRS', f'Query {q} should resolve to CURRENT_AFFAIRS, got {mode}')

    def test_current_affairs_governance_and_sports(self):
        queries = [
            'وزیراعظم اور کابینہ کے حالیہ فیصلے کیا ہیں؟',
            'نادرا کے شناختی کارڈ اور فیس کے بارے میں نئی اپ ڈیٹ',
            'آج کے کرکٹ میچ کا تازہ اسکور کیا ہے، کون جیتا؟',
            'PSL live score and who won today\'s match?'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CURRENT_AFFAIRS', f'Query {q} should resolve to CURRENT_AFFAIRS, got {mode}')

    def test_image_turbo_urdu_requests(self):
        queries = [
            'شام کے وقت بادشاہی مسجد کی ایک خوبصورت تصویر بناؤ',
            'لاہور کے شاہی قلعے کی تصویر بنا کر دو',
            'کراچی کے ساحل اور غروبِ آفتاب کی فوٹو بنائیے',
            'ایک جدید نیون ریکشہ کی پینٹنگ بناؤ'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'IMAGE_TURBO', f'Query {q} should resolve to IMAGE_TURBO, got {mode}')

    def test_image_turbo_roman_and_english_requests(self):
        queries = [
            'lahore fort ki shaam ke waqt tasveer banao',
            'k2 mountain ki ek photo bana do',
            'Generate an image of Badshahi Mosque in Lahore at sunset',
            'Create a picture of futuristic cyberpunk Karachi street',
            'draw an image of traditional Pakistani haveli courtyard'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'IMAGE_TURBO', f'Query {q} should resolve to IMAGE_TURBO, got {mode}')

    def test_code_tech_programming_queries(self):
        queries = [
            'پائتھون میں ایک فنکشن لکھو جو لسٹ کو ریورس کرے',
            'write a C# script for Unity character movement',
            'javascript code for countdown timer',
            'ایس کیو ایل میں ایک ٹیبل بنانے کی کوئری لکھ کر دیں',
            'write a python script to scrape news headlines'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CODE_TECH', f'Query {q} should resolve to CODE_TECH, got {mode}')

    def test_code_tech_essay_and_article_queries(self):
        queries = [
            'تعلیم کے فوائد پر ایک مفصل مضمون لکھو',
            'write an essay on artificial intelligence and its impact',
            'آلودگی کے خاتمے پر ایک جامع مقالہ تحریر کریں',
            'write an article on the economic potential of IT in Pakistan'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'CODE_TECH', f'Query {q} should resolve to CODE_TECH, got {mode}')

    def test_adab_poetry_classical_poets(self):
        queries = [
            'علامہ اقبال کا کوئی خوبصورت شعر سنائیں',
            'مرزا غالب کی کوئی مشہور غزل پیش کریں',
            'احمد فراز کی شاعری سے کوئی انتخاب سناؤ',
            'فیض احمد فیض کی نظم کے چند اشعار پیش کیجیے',
            'جون ایلیا کا کوئی یادگار شعر سنائیں'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'ADAB_POETRY', f'Query {q} should resolve to ADAB_POETRY, got {mode}')

    def test_adab_poetry_general_requests(self):
        queries = [
            'محبت پر کوئی خوبصورت شعر سناؤ',
            'علمِ عروض اور بحر کے مطابق کوئی باوزن غزل پیش کریں',
            'urdu shayari sunao koi achi si',
            'recite an Urdu poem with rhyming couplets'
        ]
        for q in queries:
            mode = self.orch.resolve_prompt_mode(q)
            self.assertEqual(mode, 'ADAB_POETRY', f'Query {q} should resolve to ADAB_POETRY, got {mode}')

    # ─────────────────────────────────────────────────────────────────────────
    # 8. End-to-End Orchestrate Turn Dispatches & Mock Backend
    # ─────────────────────────────────────────────────────────────────────────
    def test_orchestrate_turn_greeting_fast_path(self):
        reply, prof = self.orch.orchestrate_turn('السلام علیکم', conversation_history=[])
        self.assertIn('وعلیکم السلام', reply)
        self.assertIn('آپ', reply)

    def test_orchestrate_turn_identity_fast_path(self):
        reply, prof = self.orch.orchestrate_turn('آپ کون ہیں؟', conversation_history=[])
        self.assertIn('تبریز', reply)

    def test_orchestrate_turn_time_date_fast_path(self):
        reply, prof = self.orch.orchestrate_turn('وقت کیا ہوا ہے؟', conversation_history=[])
        self.assertTrue(len(reply) > 5)

    def test_orchestrate_turn_nsfw_image_blocking(self):
        reply, prof = self.orch.orchestrate_turn('ایک ننگی تصویر بناؤ', conversation_history=[])
        self.assertIn('معذرت خواہ ہوں', reply)
        self.assertNotIn('[IMAGE_CARD:', reply)

    def test_orchestrate_turn_image_generation_dispatch(self):
        class MockImageClient:
            def generate_image(self, prompt, **kwargs):
                return {
                    'status': 'success',
                    'image_base64': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
                    'filename': 'test_img.png',
                    'modal_storage_path': '/root/output/adaab_images/test_img.png'
                }
        orch_with_mock = CognitiveOrchestrator(backend_client=MockImageClient())
        reply, prof = orch_with_mock.orchestrate_turn('بادشاہی مسجد کی ایک خوبصورت تصویر بناؤ', conversation_history=[])
        self.assertIn('[IMAGE_CARD:', reply)
        self.assertIn('test_img.png', reply)

    def test_orchestrate_turn_deep_exploration_parameters(self):
        recorded_calls = []
        class MockDeepClient:
            def chat_completion(self, messages, **kwargs):
                recorded_calls.append((messages, kwargs))
                return {'content': 'یہ ہے تفصیل سے مکمل علمی جائزہ جو ہر پہلو کا احاطہ کرتا ہے۔'}
        orch_with_mock = CognitiveOrchestrator(backend_client=MockDeepClient())
        reply, prof = orch_with_mock.orchestrate_turn(
            'پاکستان میں مصنوعی ذہانت کے مستقبل پر تفصیل سے روشنی ڈالیں',
            conversation_history=[]
        )
        self.assertTrue(len(recorded_calls) > 0)
        messages, kwargs = recorded_calls[0]
        self.assertEqual(kwargs.get('max_tokens'), 850, 'DEEP_EXPLORATION must allocate 850 tokens for comprehensive depth')
        self.assertIn('Deep Exploration Mode', messages[0]['content'])

    def test_orchestrate_turn_poetry_parameters(self):
        recorded_calls = []
        class MockPoetryClient:
            def chat_completion(self, messages, **kwargs):
                recorded_calls.append((messages, kwargs))
                return {'content': 'ستاروں سے آگے جہاں اور بھی ہیں\nابھی عشق کے امتحان اور بھی ہیں'}
        orch_with_mock = CognitiveOrchestrator(backend_client=MockPoetryClient())
        reply, prof = orch_with_mock.orchestrate_turn(
            'علامہ اقبال کا کوئی شعر سنائیں',
            conversation_history=[]
        )
        self.assertTrue(len(recorded_calls) > 0)
        messages, kwargs = recorded_calls[0]
        self.assertEqual(kwargs.get('temperature'), 0.75, 'Poetry mode must use temperature=0.75 for creative rhythm')
        self.assertIn('Adab & Poetry Mode', messages[0]['content'])

    def test_orchestrate_turn_code_tech_parameters(self):
        recorded_calls = []
        class MockCodeClient:
            def chat_completion(self, messages, **kwargs):
                recorded_calls.append((messages, kwargs))
                return {'content': 'آپ کا مطلوبہ کوڈ تیار ہے۔\n\n```python\ndef add(a, b):\n    return a + b\n```'}
        orch_with_mock = CognitiveOrchestrator(backend_client=MockCodeClient())
        reply, prof = orch_with_mock.orchestrate_turn(
            'پائتھون میں دو نمبرز جمع کرنے کا کوڈ لکھیں',
            conversation_history=[]
        )
        self.assertTrue(len(recorded_calls) > 0)
        messages, kwargs = recorded_calls[0]
        self.assertEqual(kwargs.get('max_tokens'), 900, 'Code mode must allocate 900 tokens for complete scripts')
        self.assertIn('```python', reply)


if __name__ == '__main__':
    unittest.main()