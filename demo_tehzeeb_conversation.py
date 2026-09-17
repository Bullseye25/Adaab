"""
=============================================================================
Tehzeeb (تہذیب) & Tabraiz (تبریز) - Memory, Time & Personality Showcase
Demonstrating:
 1. Opening Etiquette: "آداب! میرا نام تہذیب ہے، میں آپ کی کیا مدد کر سکتی ہوں؟"
 2. Dynamic Onboarding: User introduces name, profession, country, education
 3. Dynamic SQLite persistence & Disambiguation
 4. Local PC Time & Date Awareness (exact clock in B1-B2 Urdu)
 5. Classical Poetry Recitation with B1-B2 level explanations
 6. "Who did you talk to?" Query: AI recalls acquaintances and summarizes personalities
 7. High-Fidelity Female/Male Neural Voice Audio Synthesis
=============================================================================
"""

import os
import sys
import time

# Ensure UTF-8 output across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from backend import AdaabClient
from time_context import get_local_time_context
from memory_engine import (
    save_or_update_user_profile,
    get_all_interacted_profiles,
    format_who_did_you_talk_to_response
)
from agent import (
    ADAAB_SYSTEM_PROMPT,
    TABRAIZ_SYSTEM_PROMPT,
    handle_conversation_turn
)
from voice_reader import generate_conversation_speech, DEFAULT_VOICE, VOICE_FEMALE_URDU, VOICE_MALE_URDU

def run_dialogue_step(client: AdaabClient, history: list, user_message: str, active_profile: dict, persona: str = "Tehzeeb") -> tuple[str, dict]:
    """Executes a dialogue turn through the unified handle_conversation_turn router."""
    print(f"\n[آپ / User]:\n  \"{user_message}\"")
    history.append({"role": "user", "content": user_message})

    print(f"  ... [{persona} غور فرما رہی ہیں]")
    start_time = time.time()

    bot_reply, updated_profile = handle_conversation_turn(
        user_message=user_message,
        history=history,
        client=client,
        persona=persona,
        active_profile=active_profile
    )

    elapsed = time.time() - start_time
    print(f"[{persona}] ({elapsed:.2f}s):\n  {bot_reply}")
    history.append({"role": "assistant", "content": bot_reply})
    return bot_reply, updated_profile

def main():
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    print("\n" + "=" * 78)
    print(" 🌸 آداب اسٹوڈیو: تہذیب و تبریز — وقت، یادداشت اور شخصی تجزیہ 🌸 ".center(78, "="))
    print(" Local PC Clock | SQLite Memory | Personality Profiling | B1-B2 Urdu ".center(78))
    print("=" * 78 + "\n")

    client = AdaabClient()

    # Step 1: Local PC Time Context
    time_ctx = get_local_time_context()
    print("--> 1. Local PC System Clock Checked:")
    print(f"    📅 تاریخ: {time_ctx['urdu_date_str']}")
    print(f"    ⏰ وقت: {time_ctx['urdu_time_str']} ({time_ctx['english_str']})")
    print(f"    مناسبت: {time_ctx['greeting_urdu']}")

    # Step 2: Opening Etiquette Greeting
    history = []
    active_profile = {}
    initial_greeting = "آداب! میرا نام تہذیب ہے۔ میں آپ سے کچھ آسان سوالات اس لیے پوچھ رہی ہوں تاکہ اگلی بار جب آپ تشریف لائیں تو میں آپ کو یاد رکھ سکوں۔ اس مقصد کے لیے آپ کا مکمل نام (Full Name) بتانا لازمی ہے۔ برائے مہربانی اپنا پورا نام عنایت فرمائیے تاکہ میں آپ کا اندراج کر سکوں؟"
    print("\n" + "-" * 78)
    print("--> 2. Initial Etiquette Greeting (بات چیت کا باادب آغاز):")
    print(f"[تہذیب / Tehzeeb]:\n  \"{initial_greeting}\"")
    print("-" * 78)
    history.append({"role": "assistant", "content": initial_greeting})

    # Step 3: Synthesize Initial Greeting Audio
    greeting_audio_path = os.path.join(os.getcwd(), "tehzeeb_opening_greeting.mp3")
    print(f"\n--> 3. Synthesizing Opening Audio ({DEFAULT_VOICE})...")
    try:
        generate_conversation_speech(initial_greeting, greeting_audio_path, voice=DEFAULT_VOICE)
        if os.path.exists(greeting_audio_path):
            file_size_kb = os.path.getsize(greeting_audio_path) / 1024
            print(f"    ✓ Audio ready: {greeting_audio_path} ({file_size_kb:.1f} KB)")
    except Exception as e:
        print(f"    ⚠️ Notice: {e}")

    if interactive:
        print("\n" + "=" * 78)
        print(" Interactive Mode Active! Type your message. (Type 'exit' to quit)")
        print("=" * 78)
        while True:
            try:
                user_msg = input("\n[آپ / You]: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit", "q"]:
                    print("\nتہذیب: آداب عرض ہے جناب! آپ سے گفتگو کر کے دلی مسرت ہوئی۔ خدا حافظ و ناصر۔")
                    break
                _, active_profile = run_dialogue_step(client, history, user_msg, active_profile, persona="Tehzeeb")
            except KeyboardInterrupt:
                print("\n\nExiting session...")
                break
    else:
        print("\n" + "-" * 78)
        print("--> 4. Running Multi-Turn Demonstration Showcase:")
        print("-" * 78)

        # Showcase Turn 1: Onboarding User Information (Name, Profession, Country, Education)
        print("\n--- [مرحلہ 1: صارف کا تعارف و ڈیٹا بیس اندراج (Onboarding & SQLite Save)] ---")
        reply1, active_profile = run_dialogue_step(
            client, history,
            "آداب! میرا نام عماد رضا ہے، میں ایک سافٹ ویئر انجینئر ہوں، میرا تعلق پاکستان سے ہے اور میں نے کمپیوٹر سائنس میں تعلیم حاصل کی ہے۔",
            active_profile
        )

        # Showcase Turn 2: Asking for Current Local PC Time & Date
        print("\n--- [مرحلہ 2: کمپیوٹر کے حاضر وقت اور تاریخ کا استفسار (Local PC Clock Awareness)] ---")
        reply2, active_profile = run_dialogue_step(
            client, history,
            "تہذیب، ابھی کیا وقت اور کیا تاریخ ہوئی ہے؟",
            active_profile
        )

        # Showcase Turn 3: Poetic Conversation with B1-B2 level explanation
        print("\n--- [مرحلہ 3: کلامِ غالب اور B1-B2 سلیس تشریح (Poetry & Accessible Urdu)] ---")
        reply3, active_profile = run_dialogue_step(
            client, history,
            "Can you recite a philosophical couplet by Mirza Ghalib about desire and explain it in simple Urdu?",
            active_profile
        )

        # Showcase Turn 4: Memory Query - "Who did you talk to?"
        print("\n--- [مرحلہ 4: یادداشت کا استفسار: آپ نے کن کن لوگوں سے بات کی ہے؟ (Who Did You Talk To?)] ---")
        reply4, active_profile = run_dialogue_step(
            client, history,
            "تہذیب، آپ نے اب تک کن کن لوگوں سے بات کی ہے؟ اور ان کی شخصیت کیسی ہے؟",
            active_profile
        )

        # Step 5: Synthesize Final Memory Output Audio
        final_audio_path = os.path.join(os.getcwd(), "tehzeeb_memory_summary.mp3")
        print(f"\n--> 5. Synthesizing Spoken Audio for Tehzeeb's Memory Summary...")
        try:
            generate_conversation_speech(reply4[:250], final_audio_path, voice=DEFAULT_VOICE)
            if os.path.exists(final_audio_path):
                file_size_kb = os.path.getsize(final_audio_path) / 1024
                print(f"    ✓ Spoken memory audio generated: {final_audio_path} ({file_size_kb:.1f} KB)")
        except Exception as e:
            print(f"    ⚠️ Notice: {e}")

    print("\n" + "=" * 78)
    print(" ✓ Demonstration Showcase Completed Successfully! ".center(78, "="))
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
