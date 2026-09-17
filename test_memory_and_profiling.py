"""
=============================================================================
Test Suite: Temporal Awareness, SQLite User Memory & Personality Profiling
Verifies:
 1. Local PC time/date context extraction & B1-B2 Urdu formatting
 2. SQLite user profile persistence & retrieval
 3. Disambiguation between multiple individuals sharing the same name
 4. Conversation analysis, key-points extraction & personality summarization
 5. "Who did you talk to?" query response formatting
 6. Unified handle_conversation_turn router
=============================================================================
"""

import os
import sys
import tempfile
import sqlite3

# Ensure UTF-8 console output in Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from time_context import (
    get_local_time_context,
    is_time_or_date_query,
    get_time_date_response
)
from memory_engine import (
    init_memory_tables,
    save_or_update_user_profile,
    find_profiles_by_name,
    get_all_interacted_profiles,
    get_profile_by_id,
    extract_profile_details_from_text,
    is_save_details_command,
    is_who_did_you_talk_to_query,
    extract_key_points_and_summarize_personality,
    format_who_did_you_talk_to_response,
    is_declining_info_sharing,
    extract_preferred_call_name,
    evaluate_user_sharing_consent,
    get_profiles_with_sharing_switch,
    clear_user_memory
)
from agent import (
    handle_conversation_turn,
    is_tabraiz_invoked,
    is_tehzeeb_invoked,
    get_tabraiz_identity_response,
    get_tehzeeb_identity_response
)

class MemoryTestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 9

    def log(self, test_name: str, success: bool, msg: str = ""):
        if success:
            self.passed += 1
            print(f" [PASS] {test_name}: {msg}")
        else:
            self.failed += 1
            print(f" [FAIL] {test_name}: {msg}")

def run_tests():
    print("=" * 72)
    print(" Adaab Studio: Memory, Time & Personality Profiling Test Suite ".center(72, "="))
    print("=" * 72 + "\n")
    runner = MemoryTestRunner()

    # Use a temporary SQLite database for clean test isolation
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, "test_memory.db")
    init_memory_tables(test_db)
    import memory_engine
    memory_engine.DB_PATH = test_db

    # ── TEST 1: Local PC Time and Date Awareness ─────────────────────────────
    try:
        t_ctx = get_local_time_context()
        assert t_ctx["year"] >= 2026, "Must retrieve valid current year"
        assert t_ctx["day_name_urdu"], "Must generate Urdu day of week"
        assert t_ctx["month_name_urdu"], "Must generate Urdu month name"
        assert is_time_or_date_query("ابھی کیا وقت ہے؟"), "Must detect Urdu time query"
        assert is_time_or_date_query("what is the date today?"), "Must detect English date query"
        
        resp_time = get_time_date_response("ابھی کیا وقت ہے؟", persona="Tehzeeb")
        assert "گھڑی" in resp_time or "منٹ" in resp_time or "بج" in resp_time, "Time response must reflect clock"
        runner.log("TEST-01: Local PC Time Awareness", True, f"Clock: {t_ctx['urdu_full_str']}")
    except Exception as e:
        runner.log("TEST-01: Local PC Time Awareness", False, str(e))

    # ── TEST 2: User Profile SQLite Persistence ──────────────────────────────
    try:
        p1 = save_or_update_user_profile(
            name="Ammad",
            profession="Software Engineer",
            country="Pakistan",
            education="BS Computer Science",
            db_path=test_db
        )
        assert p1["id"] is not None
        assert p1["name"] == "Ammad"
        assert p1["profession"] == "Software Engineer"
        assert p1["country"] == "Pakistan"

        fetched = get_profile_by_id(p1["id"], db_path=test_db)
        assert fetched["name"] == "Ammad"
        runner.log("TEST-02: User Profile Persistence", True, f"Saved & retrieved: {fetched['name']} ({fetched['profession']})")
    except Exception as e:
        runner.log("TEST-02: User Profile Persistence", False, str(e))

    # ── TEST 3: Disambiguating Multiple Users with Same Name ──────────────────
    try:
        # Save second user also named Ammad, but different profession & country
        p2 = save_or_update_user_profile(
            name="Ammad",
            profession="Medical Doctor",
            country="United Kingdom",
            education="MBBS",
            db_path=test_db
        )
        matches = find_profiles_by_name("Ammad", db_path=test_db)
        assert len(matches) == 2, f"Must find 2 distinct profiles for Ammad, got {len(matches)}"
        assert matches[0]["profession"] != matches[1]["profession"], "Profiles must have distinct professions"
        runner.log("TEST-03: User Disambiguation", True, f"Disambiguated 2 people named 'Ammad': {matches[0]['profession']} vs {matches[1]['profession']}")
    except Exception as e:
        runner.log("TEST-03: User Disambiguation", False, str(e))

    # ── TEST 4: Conversation Analysis & Personality Summarization Algo ───────
    try:
        test_messages = [
            {"role": "user", "content": "Please tell me a philosophical couplet by Mirza Ghalib about desire and life."},
            {"role": "assistant", "content": "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے..."},
            {"role": "user", "content": "Thank you! I work with AI and software engineering, and I love how deep Urdu poetry is."}
        ]
        summary = extract_key_points_and_summarize_personality(test_messages, p1)
        assert "غالب" in summary["topics"], "Topics must detect Ghalib discussion"
        assert "ٹیکنالوجی" in summary["topics"] or "مصنوعی ذہانت" in summary["topics"], "Topics must detect tech discussion"
        assert len(summary["personality_summary"]) > 20, "Must generate rich personality summary"
        assert len(summary["appearance_impression"]) > 20, "Must generate impression of person"
        runner.log("TEST-04: Personality Summarization Algo", True, f"Summary: {summary['personality_summary'][:65]}...")
    except Exception as e:
        runner.log("TEST-04: Personality Summarization Algo", False, str(e))

    # ── TEST 5: 'Who Did You Talk To?' Memory Querying ───────────────────────
    try:
        assert is_who_did_you_talk_to_query("Who did you talk to?"), "Must detect English query"
        assert is_who_did_you_talk_to_query("آپ نے کن کن لوگوں سے بات کی ہے؟"), "Must detect Urdu query"

        # Update profile with personality summary in test db
        save_or_update_user_profile(
            name="Ammad",
            profession="Software Engineer",
            country="Pakistan",
            personality_summary="ایک شائستہ اور فکری شعور رکھنے والے انجینئر جنہیں غالب کا فلسفہ پسند ہے۔",
            appearance_impression="متین اور باوقار چہرہ، سلجھا ہوا دوستانہ لہجہ۔",
            profile_id=p1["id"],
            db_path=test_db
        )
        who_response = format_who_did_you_talk_to_response(persona="Tehzeeb", db_path=test_db)
        assert "Ammad" in who_response, "Response must mention Ammad"
        assert "Software Engineer" in who_response, "Response must mention profession"
        assert "پاکستان" in who_response or "Pakistan" in who_response, "Response must mention country"
        assert "شخصیت کا خلاصہ" in who_response, "Response must include personality summary section"
        runner.log("TEST-05: Memory Query ('Who did you talk to?')", True, "Successfully returned structured profiles and personality summaries")
    except Exception as e:
        runner.log("TEST-05: Memory Query ('Who did you talk to?')", False, str(e))

    # ── TEST 6: Unified Conversation Router & Tabraiz Male Persona ───────────
    try:
        # Test time handling
        t_reply, _ = handle_conversation_turn("ابھی کیا وقت ہے؟", [], persona="Tabraiz")
        assert "کر سکتا ہوں" in t_reply or "وقت" in t_reply, "Tabraiz must use masculine phrasing"

        # Test onboarding name extraction
        welcome_reply, active_prof = handle_conversation_turn("میرا نام کامران ہے اور میں استاد ہوں", [], persona="Tehzeeb")
        assert "کامران" in welcome_reply, "Must acknowledge user's name"
        assert active_prof is not None and active_prof.get("name") == "کامران"

        # Test memory query through router
        who_reply, _ = handle_conversation_turn("Who did you speak with?", [], persona="Tehzeeb")
        assert "آداب" in who_reply and ("احباب" in who_reply or "گفتگو" in who_reply)
        runner.log("TEST-06: Unified Router & Tabraiz Male Persona", True, "Verified routing, Tabraiz masculine forms, and dynamic onboarding")
    except Exception as e:
        runner.log("TEST-06: Unified Router & Tabraiz Male Persona", False, str(e))

    # ── TEST 7: Mandatory Full Name & Onboarding Explanation Sequence ────────
    try:
        # 1. Opening greeting explains the reason and mandatory full name
        greeting_reply, _ = handle_conversation_turn("آداب", [], persona="Tehzeeb")
        assert "یاد رکھ" in greeting_reply or "پہچان" in greeting_reply, "Greeting must explain purpose of asking questions is to remember user"
        assert "مکمل نام" in greeting_reply and "لازمی" in greeting_reply, "Greeting must state full name is mandatory"

        # 2. User provides Full Name -> AI saves and asks the simple questions
        name_turn_reply, prof_state = handle_conversation_turn("My full name is Tariq Aziz", [], persona="Tehzeeb")
        assert "Tariq Aziz" in name_turn_reply, "Must acknowledge full name"
        assert ("پیشہ" in name_turn_reply or "Profession" in name_turn_reply) and ("ملک" in name_turn_reply or "شہر" in name_turn_reply), "Must ask subsequent simple questions"
        assert prof_state.get("name") == "Tariq Aziz", "Must save full name to profile"

        # 3. User provides answers to subsequent simple questions
        details_turn_reply, updated_prof_state = handle_conversation_turn(
            "I work as a Professor from Pakistan",
            [],
            persona="Tehzeeb",
            active_profile=prof_state
        )
        assert "محفوظ" in details_turn_reply or "یاد رکھے" in details_turn_reply, "Must confirm saving details"
        assert updated_prof_state.get("profession") == "Professor"
        assert updated_prof_state.get("country") == "Pakistan"

        # 4. Single name warning check
        single_reply, _ = handle_conversation_turn("Ali", [], persona="Tehzeeb")
        assert "مکمل نام" in single_reply or "لازمی" in single_reply, "Must remind that full name is mandatory if only single name provided"

        runner.log("TEST-07: Mandatory Full Name & Onboarding Explanation", True, "Verified onboarding reason explanation, full name mandate, and subsequent simple questions")
    except Exception as e:
        runner.log("TEST-07: Mandatory Full Name & Onboarding Explanation", False, str(e))

    # ── TEST 8: Privacy Opt-Out, Preferred Call Name & Database Switch-Case ──
    try:
        # 1. Decline detection across multilingual inputs
        assert is_declining_info_sharing("I don't want to share my information"), "English decline detection failed"
        assert is_declining_info_sharing("I prefer not to say"), "English preference detection failed"
        assert is_declining_info_sharing("نہیں، میں اپنی معلومات شیئر نہیں کرنا چاہتا"), "Urdu decline detection failed"
        assert is_declining_info_sharing("meri personal info hai share nahi karni"), "Roman Urdu decline detection failed"

        # 2. Preferred Call Name extraction
        assert extract_preferred_call_name("You can call me Tiger") == "Tiger", "English call name extraction failed"
        assert extract_preferred_call_name("آپ مجھے مسافر کہہ سکتے ہیں") == "مسافر", "Urdu call name extraction failed"
        assert extract_preferred_call_name("Mujhe Dost keh lein") == "Dost", "Roman Urdu call name extraction failed"

        # 3. Conversation Turn: User declines personal info -> AI respects privacy and asks for preferred call name
        decline_reply, priv_prof = handle_conversation_turn("I don't want to share my information", [], persona="Tehzeeb")
        assert "پرائیویسی" in decline_reply or "پردۂ راز" in decline_reply, "AI must acknowledge privacy with respect"
        assert "کس نام" in decline_reply or "لقب" in decline_reply or "عرفیت" in decline_reply, "AI must ask what user should be called"
        assert priv_prof.get("awaiting_call_name") == True, "Profile must be flagged as awaiting call name"
        assert priv_prof.get("info_sharing_consent") == "declined", "Consent must be recorded as declined"

        # 4. Subsequent Turn: User provides preferred call name -> AI accepts and addresses user by chosen nickname
        call_reply, updated_priv_prof = handle_conversation_turn("You can call me Tiger", [], persona="Tehzeeb", active_profile=priv_prof)
        assert "Tiger" in call_reply, "AI must address user by preferred call name"
        assert updated_priv_prof.get("name") == "Tiger" or updated_priv_prof.get("display_name") == "Tiger", "Call name must be stored"
        assert updated_priv_prof.get("awaiting_call_name") == False, "Awaiting call name flag must be cleared"
        assert updated_priv_prof.get("info_sharing_consent") == "preferred_name_only", "Consent must be updated to preferred_name_only"

        # 5. Database SQL CASE Switch statement verification
        # Save profiles with each status in test_db
        save_or_update_user_profile(name="Zaid Khan", profession="Architect", country="UAE", info_sharing_consent="consented", db_path=test_db)
        save_or_update_user_profile(name="Falcon", display_name="Falcon", info_sharing_consent="preferred_name_only", db_path=test_db)
        save_or_update_user_profile(name="محترم مہمان", display_name="محترم مہمان", info_sharing_consent="declined", db_path=test_db)

        db_switch_rows = get_profiles_with_sharing_switch(test_db)
        zaid_row = next(r for r in db_switch_rows if r["name"] == "Zaid Khan")
        falcon_row = next(r for r in db_switch_rows if r["name"] == "Falcon")
        guest_row = next(r for r in db_switch_rows if r["name"] == "محترم مہمان")

        assert zaid_row["can_share_details"] == 1, "SQL CASE must allow details for consented users"
        assert "مکمل کوائف" in zaid_row["consent_status_label"], "SQL CASE label must reflect full consent"

        assert falcon_row["can_share_details"] == 0, "SQL CASE must withhold details for preferred_name_only users"
        assert "صرف تخاطب" in falcon_row["consent_status_label"], "SQL CASE label must reflect call-name only"

        assert guest_row["can_share_details"] == 0, "SQL CASE must withhold details for declined users"
        assert "مسترد" in guest_row["consent_status_label"] or "پرائیویسی" in guest_row["consent_status_label"], "SQL CASE label must reflect opt-out"

        # 6. Python match/case Switch verification
        c_eval = evaluate_user_sharing_consent("consented")
        d_eval = evaluate_user_sharing_consent("declined")
        p_eval = evaluate_user_sharing_consent("preferred_name_only")
        assert c_eval["key"] == "consented" and c_eval["allows_details"] is True
        assert d_eval["key"] == "declined" and d_eval["allows_details"] is False
        assert p_eval["key"] == "preferred_name_only" and p_eval["allows_details"] is False

        # 7. "Who did you talk to?" Switch-case output check
        who_output = format_who_did_you_talk_to_response("Tehzeeb", db_path=test_db)
        assert "Falcon" in who_output, "Must list call-name only user"
        assert "کوائف کا تحفظ" in who_output or "مخفی" in who_output, "Must respect privacy for opted-out users"

        runner.log("TEST-08: Privacy Opt-Out, Preferred Call Name & DB Switch-Case", True, "Verified multilingual opt-out detection, call name extraction, DB SQL CASE, and Python match/case")
    except Exception as e:
        runner.log("TEST-08: Privacy Opt-Out, Preferred Call Name & DB Switch-Case", False, str(e))

    # ── TEST 9: Clear User Memory & Empty State ──────────────────────────────
    try:
        success = clear_user_memory(test_db)
        assert success is True, "clear_user_memory must return True"
        remaining_profiles = get_all_interacted_profiles(test_db)
        assert len(remaining_profiles) == 0, "No profiles should remain after memory clear"

        empty_output = format_who_did_you_talk_to_response("Tabraiz", db_path=test_db)
        assert "صاف (خالی)" in empty_output or "کوئی سابقہ ریکارڈ" in empty_output, "Empty state must reflect cleared memory"
        runner.log("TEST-09: Clear User Memory & Reset State", True, "Successfully wiped SQLite tables and verified empty state response")
    except Exception as e:
        runner.log("TEST-09: Clear User Memory & Reset State", False, str(e))

    print("\n" + "=" * 72)
    print(f" Test Results: {runner.passed}/{runner.total} Passed ({int(runner.passed/runner.total*100)}% Success Rate)")
    print("=" * 72 + "\n")
    return runner.passed == runner.total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
