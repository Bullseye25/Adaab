import os
import sys
import json
import base64
import tempfile
import gradio as gr
from PIL import Image

# Ensure UTF-8 across Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")

def load_icon_b64(name: str) -> str:
    p = os.path.join(ICONS_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('ascii')}"
    return ""

def get_icon_path(name: str) -> str:
    p = os.path.join(ICONS_DIR, name)
    return p if os.path.exists(p) else ""

ICON_AVATAR = load_icon_b64("tabraiz_avatar.png")
ICON_MIC = load_icon_b64("mic.png")
ICON_MIC_REC = load_icon_b64("mic_rec.png")
ICON_SEND = load_icon_b64("send.png")
ICON_SPEAKER = load_icon_b64("speaker.png")
ICON_REFRESH = load_icon_b64("refresh.png")
ICON_TRASH = load_icon_b64("trash.png")
ICON_SPINNER = load_icon_b64("spinner.png")
ICON_CHAT = load_icon_b64("chat.png")
ICON_MEMORY = load_icon_b64("memory.png")
ICON_MOBILE = load_icon_b64("mobile.png")
ICON_GEAR = load_icon_b64("gear.png")

from agent import (
    ADAAB_SYSTEM_PROMPT,
    get_styled_system_prompt,
    is_tehzeeb_invoked,
    is_tabraiz_invoked,
    get_tehzeeb_identity_response,
    get_tabraiz_identity_response,
    is_greeting_or_opening,
    get_tehzeeb_etiquette_greeting,
    get_tabraiz_etiquette_greeting,
    handle_conversation_turn
)
from memory_engine import (
    get_all_interacted_profiles,
    format_who_did_you_talk_to_response,
    save_or_update_user_profile,
    clear_user_memory
)
from backend import AdaabClient

TABRAIZ_INITIAL_GREETING = "آداب! میں تبریز ہوں۔ بتائیں، میں آپ کی کیا مدد کر سکتا ہوں؟"
TABRAIZ_OPENING_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabraiz_opening_greeting.mp3")
OPENING_AUDIO_VAL = TABRAIZ_OPENING_AUDIO if os.path.exists(TABRAIZ_OPENING_AUDIO) else None
SESSION_STATE = {"active_profile": {}}
from voice_reader import (
    generate_conversation_speech,
    VOICE_FEMALE_URDU,
    VOICE_MALE_URDU,
    DEFAULT_VOICE
)
from voice_listener import transcribe_audio, transcribe_base64_audio
from crash_tracker import get_gpu_telemetry, get_latest_crash_report
from network_helper import (
    get_local_lan_ip,
    ensure_self_signed_cert,
    generate_qr_code_ascii,
    generate_qr_code_base64,
    is_port_in_use,
    resolve_available_port,
    start_cloudflared_tunnel,
    is_valid_cloudflare_tunnel_url,
    start_localtunnel
)

def get_audio_data_uri(file_path: str | None) -> str:
    """Converts a local audio file to an in-memory base64 data URI for instant client-side playback."""
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:audio/mp3;base64,{b64}"
    except Exception as err:
        print(f"[Adaab Audio] Base64 encoding notice: {err}")
        return ""

PATH_BOT_AVATAR = get_icon_path("bot_robot_avatar.png")
PATH_USER_AVATAR = get_icon_path("user_avatar.png")

def format_assistant_message(urdu_text: str, audio_path: str | None = None, prebuilt_uri: str | None = None) -> str:
    """
    Renders assistant response matching the tablet mockup:
    Urdu Nastaliq text, sleek dark audio player pill with play/pause and animated waveform bars,
    and a subtle copy button below.
    """
    audio_uri = prebuilt_uri or get_audio_data_uri(audio_path)
    
    bar_heights = [5, 11, 16, 9, 15, 20, 12, 17, 13, 19, 8, 14, 18, 12, 7, 15, 11, 14, 8, 5]
    bars_html = "".join([f'<span class="w-bar" style="height:{h}px; animation-delay:{(i*0.06):.2f}s;"></span>' for i, h in enumerate(bar_heights)])

    audio_pill_html = ""
    if audio_uri:
        audio_pill_html = f"""
        <div class="bot-audio-player-pill" onclick="window.playSpeechAudio(this)" data-audiosrc="{audio_uri}" title="آواز سنیں (Play Audio)">
            <button type="button" class="audio-play-circle-btn">
                <svg class="play-svg" viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                <svg class="pause-svg" viewBox="0 0 24 24" width="13" height="13" fill="currentColor" style="display:none;"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
            </button>
            <div class="audio-waveform-bars">
                {bars_html}
            </div>
            <span class="audio-pill-duration">صوتی کلام</span>
        </div>
        """

    copy_btn_html = """
    <div class="msg-action-bar">
        <button type="button" class="copy-msg-btn" onclick="window.copyMessageText(this)" title="کاپی کریں (Copy)">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        </button>
    </div>
    """

    return (
        f'<div class="bot-msg-card">'
        f'<div class="bot-msg-text">{urdu_text}</div>'
        f'{audio_pill_html}'
        f'{copy_btn_html}'
        f'</div>'
    )

def format_user_message(urdu_text: str) -> str:
    """
    Renders user message matching the tablet mockup:
    Urdu Nastaliq text on right with copy button below.
    """
    copy_btn_html = """
    <div class="msg-action-bar user-action-bar">
        <button type="button" class="copy-msg-btn" onclick="window.copyMessageText(this)" title="کاپی کریں (Copy)">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        </button>
    </div>
    """
    return (
        f'<div class="user-msg-card">'
        f'<div class="user-msg-text">{urdu_text}</div>'
        f'{copy_btn_html}'
        f'</div>'
    )


client = AdaabClient()
STUDIO_THEME = gr.themes.Base()

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;700&family=Share+Tech+Mono&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-canvas: #0e1015;
    --bg-main-card: #171a21;
    --bg-sidebar: transparent;
    --bg-user-bubble: #2b303c;
    --bg-bot-bubble: #20242e;
    --bg-dock: #1d212a;
    --accent-cyan: #00d2ff;
    --accent-emerald: #00e5a0;
    --accent-crimson: #ff3b5c;
    --accent-violet: #7c3aed;
    --text-primary: #f0f3f8;
    --text-muted: #8e95a5;
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(0, 210, 255, 0.25);
}

* {
    box-sizing: border-box;
}

body, .gradio-container {
    background: #0e1015 !important;
    margin: 0 !important;
    padding: 0 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    height: 100vh !important;
    height: 100dvh !important;
    overflow: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
}

/* Tablet / Desktop Frame Layout */
.adaab-tablet-frame {
    display: flex !important;
    flex-direction: row !important;
    width: 100% !important;
    max-width: 1320px !important;
    height: 100vh !important;
    height: 100dvh !important;
    margin: 0 auto !important;
    padding: 16px 20px !important;
    gap: 16px !important;
    align-items: stretch !important;
    box-sizing: border-box !important;
}

/* Left Sidebar */
.adaab-sidebar {
    flex: 0 0 190px !important;
    width: 190px !important;
    min-width: 190px !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    padding: 8px 4px !important;
    background: transparent !important;
    border: none !important;
}

.sidebar-inner {
    display: flex;
    flex-direction: column;
    height: 100%;
    justify-content: flex-start;
    gap: 16px;
}

/* Sidebar Logo with glowing blue mic & sound waves */
.adaab-sidebar-logo {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    margin-bottom: 20px;
}

.sidebar-mic-glow-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: rgba(0, 210, 255, 0.07);
    border: 1px solid rgba(0, 210, 255, 0.22);
    border-radius: 14px;
    box-shadow: 0 0 16px rgba(0, 210, 255, 0.15);
}

.sidebar-mic-svg {
    filter: drop-shadow(0 0 6px #00d2ff);
    flex-shrink: 0;
}

.sidebar-soundwaves-left, .sidebar-soundwaves-right {
    display: flex;
    align-items: center;
    gap: 2.5px;
    height: 16px;
}

.sw-bar {
    width: 2px;
    background: #00d2ff;
    border-radius: 2px;
    animation: barPulseAnim 1.2s ease-in-out infinite alternate;
}
.sw-bar:nth-child(1) { height: 8px; animation-delay: 0.1s; }
.sw-bar:nth-child(2) { height: 14px; animation-delay: 0.3s; }

/* Navigation links */
.sidebar-nav-menu {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 12px;
    color: var(--text-muted);
    text-decoration: none;
    font-size: 0.93rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
}

.nav-item:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.05);
}

.nav-item.active {
    color: #ffffff;
    background: #212530;
    border-color: rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

/* Sidebar Notification Badge & Popover */
.nav-icon-badge-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.nav-badge-dot {
    position: absolute;
    top: -2px;
    right: -3px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00d2ff;
    box-shadow: 0 0 8px #00d2ff;
    animation: badgePulse 2s infinite;
}

@keyframes badgePulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.7; }
}

/* Interactive Notification Popover Window */
.adaab-notification-popover {
    position: fixed;
    top: 64px;
    left: 220px;
    width: 320px;
    background: #141720;
    border: 1px solid rgba(0, 210, 255, 0.25);
    border-radius: 16px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65), 0 0 20px rgba(0, 210, 255, 0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    z-index: 10000;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    animation: fadeSlideInLeft 0.25s ease-out;
}

@media (max-width: 768px) {
    .adaab-notification-popover {
        left: 14px;
        right: 14px;
        width: auto;
        top: 64px;
    }
}

.notif-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: rgba(255, 255, 255, 0.03);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.notif-title-row {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e2eaf8;
    font-size: 0.9rem;
    font-weight: 600;
}

.notif-close-btn {
    background: transparent;
    border: none;
    color: #888e9b;
    font-size: 1.3rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 4px;
    transition: color 0.2s ease;
}

.notif-close-btn:hover {
    color: #ffffff;
}

.notif-list {
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 280px;
    overflow-y: auto;
}

.notif-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
}

.notif-dot-active {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00e5a0;
    box-shadow: 0 0 6px #00e5a0;
    margin-top: 5px;
    flex-shrink: 0;
}

.notif-content {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.notif-item-title {
    font-size: 0.86rem;
    font-weight: 600;
    color: #ffffff;
}

.notif-item-desc {
    font-size: 0.8rem;
    color: #9ba4b8;
    line-height: 1.4;
}

.notif-time {
    font-size: 0.72rem;
    color: #5d677d;
    margin-top: 2px;
}

/* Right Main Card Container */
.adaab-main-card {
    flex: 1 1 auto !important;
    background: #171a21 !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 24px !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5) !important;
}

/* Chatbot container: full height with bottom padding ensuring messages never hide under dock */
.messenger-chat {
    position: relative !important;
    flex: 1 1 100% !important;
    width: 100% !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    padding: 20px 24px 88px 24px !important;
    box-sizing: border-box !important;
    -webkit-overflow-scrolling: touch !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
    background: transparent !important;
    border: none !important;
}

/* Hide default Gradio chatbot header buttons (share, trash, copy) */
.messenger-chat > .header,
.messenger-chat button[aria-label="Share"],
.messenger-chat button[aria-label="Clear"] {
    display: none !important;
}

/* Gradio Chatbot row overrides */
.messenger-chat .message-row, 
.messenger-chat [data-testid="bot"], 
.messenger-chat [data-testid="user"] {
    display: flex !important;
    gap: 12px !important;
    margin-bottom: 14px !important;
}

.messenger-chat [data-testid="user"], 
.messenger-chat .user {
    flex-direction: row-reverse !important;
}

/* Avatars */
.messenger-chat img.avatar-image,
.messenger-chat .avatar-container img,
.messenger-chat .avatar img {
    width: 38px !important;
    height: 38px !important;
    min-width: 38px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Message bubbles */
.bot-msg-card, .user-msg-card {
    display: flex !important;
    flex-direction: column !important;
}

.messenger-chat .bot,
.messenger-chat [data-testid="bot"] .message,
.messenger-chat .message.bot,
.messenger-chat div:has(> .bot-msg-card) {
    background: #20242e !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #f0f3f8 !important;
    padding: 14px 18px !important;
    max-width: 80% !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    animation: fadeSlideInLeft 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.messenger-chat .user,
.messenger-chat [data-testid="user"] .message,
.messenger-chat .message.user,
.messenger-chat div:has(> .user-msg-card) {
    background: #2b303c !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #ffffff !important;
    padding: 14px 18px !important;
    max-width: 80% !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    animation: fadeSlideInRight 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

/* Urdu Text */
.bot-msg-text, .user-msg-text, .urdu-text {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    font-size: 1.25rem !important;
    line-height: 2.1 !important;
    direction: rtl !important;
    text-align: right !important;
    letter-spacing: 0 !important;
}

/* Audio Player Pill */
.bot-audio-player-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 12px !important;
    background: #141720 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 30px !important;
    padding: 4px 14px 4px 6px !important;
    margin-top: 10px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    user-select: none !important;
    width: fit-content !important;
}

.bot-audio-player-pill:hover {
    border-color: rgba(0, 210, 255, 0.3) !important;
    background: #171b26 !important;
}

.bot-audio-player-pill.playing {
    border-color: rgba(0, 210, 255, 0.5) !important;
    box-shadow: 0 0 16px rgba(0, 210, 255, 0.22) !important;
}

.audio-play-circle-btn {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    border-radius: 50% !important;
    background: #232835 !important;
    border: none !important;
    color: #ffffff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.bot-audio-player-pill.playing .audio-play-circle-btn {
    background: #00d2ff !important;
    color: #0b0e14 !important;
}

.audio-waveform-bars {
    display: flex !important;
    align-items: center !important;
    gap: 2.5px !important;
    height: 22px !important;
}

.w-bar {
    display: inline-block !important;
    width: 2.5px !important;
    background: #5b6477 !important;
    border-radius: 2px !important;
    transition: background 0.2s ease !important;
}

.bot-audio-player-pill.playing .w-bar {
    background: #00d2ff !important;
    animation: barPulseAnim 0.75s ease-in-out infinite alternate !important;
}

.audio-pill-duration {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    margin-left: 4px !important;
}

/* Copy Button */
.msg-action-bar {
    display: flex !important;
    margin-top: 6px !important;
}
.bot-action-bar {
    justify-content: flex-start !important;
}
.user-action-bar {
    justify-content: flex-end !important;
}

.copy-msg-btn {
    width: 26px !important;
    height: 26px !important;
    border-radius: 7px !important;
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    color: #727a8e !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.copy-msg-btn:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
}

/* Bottom Floating Dock: Permanently Pinned at the Bottom */
.messenger-dock {
    position: absolute !important;
    bottom: 14px !important;
    left: 18px !important;
    right: 18px !important;
    height: 56px !important;
    margin: 0 !important;
    padding: 6px 10px 6px 10px !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 10px !important;
    background: #1d212a !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 40px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    z-index: 999 !important;
}

/* Clean up Gradio container wrappers inside dock */
.messenger-dock > * {
    min-width: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.messenger-dock > .gradio-html,
.messenger-dock > div:has(#ptt-btn) {
    flex: 0 0 44px !important;
    width: 44px !important;
    min-width: 44px !important;
    height: 44px !important;
}

.messenger-dock > .gradio-textbox,
.messenger-dock > div:has(#message-input) {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 0 !important;
}

.messenger-dock > .gradio-button,
.messenger-dock > button#send-btn {
    flex: 0 0 auto !important;
    width: auto !important;
}

/* Mic Button */
#ptt-btn {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    background: #181b24 !important;
    border: 1.5px solid rgba(0, 210, 255, 0.35) !important;
    box-shadow: 0 0 12px rgba(0, 210, 255, 0.2) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    touch-action: manipulation !important;
}

#ptt-btn:active {
    transform: scale(0.93) !important;
}

.ptt-mic-svg {
    fill: #00d2ff !important;
    transition: fill 0.2s ease !important;
}

#ptt-btn.recording {
    border-color: #ff3b5c !important;
    box-shadow: 0 0 20px rgba(255, 59, 92, 0.75) !important;
    animation: crimson-ring-pulse 1.2s infinite !important;
}

#ptt-btn.recording .ptt-mic-svg {
    fill: #ff3b5c !important;
}

/* Message Textbox (Text Area) */
#message-input,
#message-input .wrapper,
#message-input label,
#message-input span {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
}

#message-input textarea, 
#message-input input {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important;
    font-size: 16px !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    line-height: 1.8 !important;
    padding: 6px 12px !important;
    outline: none !important;
    resize: none !important;
}

#message-input textarea::placeholder, 
#message-input input::placeholder {
    color: rgba(255, 255, 255, 0.45) !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
}

/* Send Button */
#send-btn {
    flex: 0 0 auto !important;
    background: #f0f2f5 !important;
    color: #121418 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 28px !important;
    padding: 8px 24px !important;
    border: none !important;
    height: 40px !important;
    min-height: 40px !important;
    min-width: 76px !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#send-btn:hover {
    background: #ffffff !important;
    box-shadow: 0 4px 16px rgba(255, 255, 255, 0.28) !important;
    transform: translateY(-1px) scale(1.02) !important;
}

#send-btn:active {
    transform: scale(0.96) !important;
}

/* Disabled & Thinking States while Tabraiz is Processing */
#ptt-btn:disabled,
#ptt-btn.disabled-processing {
    opacity: 0.35 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
    filter: grayscale(0.8) !important;
}

#message-input textarea:disabled,
#message-input input:disabled,
#message-input.disabled-processing textarea {
    opacity: 0.45 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
    color: rgba(255, 255, 255, 0.35) !important;
}

#send-btn:disabled,
#send-btn.disabled-processing {
    opacity: 0.35 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
    background: #505663 !important;
    color: #888e9b !important;
}

/* Animations */
@keyframes barPulseAnim {
    0% { transform: scaleY(0.4); opacity: 0.6; }
    100% { transform: scaleY(1.3); opacity: 1; }
}

@keyframes fadeSlideInLeft {
    from { opacity: 0; transform: translateX(-14px) scale(0.98); }
    to { opacity: 1; transform: translateX(0) scale(1); }
}

@keyframes fadeSlideInRight {
    from { opacity: 0; transform: translateX(14px) scale(0.98); }
    to { opacity: 1; transform: translateX(0) scale(1); }
}

@keyframes crimson-ring-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 59, 92, 0.7), 0 0 16px rgba(255, 59, 92, 0.5); }
    50% { box-shadow: 0 0 0 7px rgba(255, 59, 92, 0), 0 0 26px rgba(255, 59, 92, 0.8); }
}

/* Hidden elements */
.hidden-control, #voice_reply_player, #ptt_raw_input, #ptt_trigger_btn, #ptt_status_box, .compact-audio-player { 
    display: none !important; 
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .adaab-tablet-frame {
        flex-direction: column !important;
        padding: 0 !important;
        gap: 0 !important;
    }
    .adaab-sidebar {
        width: 100% !important;
        min-width: 100% !important;
        height: 54px !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 8px 14px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: #13161d !important;
    }
    .sidebar-inner {
        flex-direction: row !important;
        width: 100% !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 0 !important;
    }
    .adaab-sidebar-logo {
        margin-bottom: 0 !important;
        padding: 0 !important;
    }
    .sidebar-nav-menu {
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
    }
    .adaab-main-card {
        border-radius: 0 !important;
        border: none !important;
        height: calc(100dvh - 54px) !important;
        max-height: calc(100dvh - 54px) !important;
        position: relative !important;
    }
    .messenger-chat {
        padding: 12px 14px 76px 14px !important;
        height: 100% !important;
        max-height: 100% !important;
    }
    .messenger-dock {
        position: absolute !important;
        bottom: 8px !important;
        left: 8px !important;
        right: 8px !important;
        height: 52px !important;
        margin: 0 !important;
    }
}

@supports (padding: env(safe-area-inset-bottom)) {
    .messenger-dock {
        padding-bottom: max(6px, env(safe-area-inset-bottom)) !important;
    }
}
"""

HEAD_JS = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#090d16">
<script>
const ICON_MIC_DATA = "__ICON_MIC__";
const ICON_MIC_REC_DATA = "__ICON_MIC_REC__";
const ICON_SPINNER_DATA = "__ICON_SPINNER__";
const ICON_SPEAKER_DATA = "__ICON_SPEAKER__";

let pttStream = null;
let pttRecorder = null;
let pttChunks = [];
let pttRecording = false;
let pttStarting = false;
let audioCtx = null;
let micSource = null;
let micAnalyser = null;
let micDataArray = null;
let isAiSpeaking = false;
let canvas = null;
let ctx = null;
let animWidth = 0;
let animHeight = 0;

// Equalizer Music Visualizer Configuration & State
const numEqBars = 48;
const eqHeights = new Float32Array(numEqBars);
const eqPeaks = new Float32Array(numEqBars);
let speakerAudioSource = null;
let speakerAnalyser = null;
let speakerDataArray = null;


function ensureAudioContext() {
    try {
        if (!audioCtx || audioCtx.state === 'closed') {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        return audioCtx;
    } catch(e) {
        console.warn('[Adaab] AudioContext error:', e);
        return null;
    }
}

window.playInitialGreeting = function() {
    console.log('[Adaab] playInitialGreeting invoked.');
    startWelcomeAudioNow();
};

function updateWaveBadge(text, color) {
    const badge = document.getElementById('waveform-state-text');
    const dot = document.querySelector('.waveform-state-dot');
    if (badge) badge.innerText = text;
    if (dot && color) {
        dot.style.background = color;
        dot.style.boxShadow = '0 0 10px ' + color;
    }
}

function initNeonCanvas() {
    canvas = document.getElementById('neonWaveCanvas');
    if (!canvas) return false;
    ctx = canvas.getContext('2d');
    resizeCanvas();
    return true;
}

function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    animWidth  = (rect.width  > 0 ? rect.width  : canvas.offsetWidth  || canvas.parentElement?.offsetWidth  || 360);
    animHeight = (rect.height > 0 ? rect.height : canvas.offsetHeight || 180);
    canvas.width  = animWidth  * dpr;
    canvas.height = animHeight * dpr;
    if (ctx) { ctx.setTransform(1,0,0,1,0,0); ctx.scale(dpr, dpr); }
}


function drawSineWave(w, h, baseline, color, amplitude, frequency, phase, opacity) {
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowBlur = amplitude > 0.15 ? 12 : 4;
    ctx.shadowColor = color;
    for (let x = 0; x <= w; x++) {
        const y = baseline - amplitude * h * 0.45 * Math.sin(2 * Math.PI * frequency * (x / w) + phase);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
}
\nfunction drawEqualizerVisualizer(w, h) {
    const baseline = h * 0.74;
    const maxBarH = h * 0.62;
    const padding = 16;
    const gap = 3;
    const barW = Math.max(3, (w - padding * 2 - (numEqBars - 1) * gap) / numEqBars);

    // Check actual audio player element or custom speech audio object
    const playerWrapper = document.getElementById('voice_reply_player');
    const audio = playerWrapper ? playerWrapper.querySelector('audio') : null;
    const isCustomPlaying = window.__adaabSpeechAudio && !window.__adaabSpeechAudio.paused && !window.__adaabSpeechAudio.ended && window.__adaabSpeechAudio.currentTime > 0;
    const isGradioPlaying = audio && !audio.paused && !audio.ended && audio.currentTime > 0;
    const isAudioPlaying = isCustomPlaying || isGradioPlaying;

    if (isAudioPlaying && !pttRecording) {
        isAiSpeaking = true;
    } else if (!isAudioPlaying && !pttRecording) {
        isAiSpeaking = false;
    }

    let hasRealData = false;
    if (isAiSpeaking && speakerAnalyser && speakerDataArray) {
        try {
            speakerAnalyser.getByteFrequencyData(speakerDataArray);
            let sum = 0;
            for (let i = 0; i < speakerDataArray.length; i++) sum += speakerDataArray[i];
            if (sum > 20) hasRealData = true;
        } catch(e) {}
    }

    for (let i = 0; i < numEqBars; i++) {
        const norm = i / numEqBars;
        let targetVal = 0.05; // Idle resting baseline

        if (isAiSpeaking) {
            if (hasRealData) {
                const bin = Math.min(speakerDataArray.length - 1, Math.floor(Math.pow(norm, 1.4) * (speakerDataArray.length - 1)));
                targetVal = Math.max(0.08, (speakerDataArray[bin] || 0) / 255.0);
            } else {
                // Syllabic human speech cadence synced to playback time
                const activeAudio = (window.__adaabSpeechAudio && !window.__adaabSpeechAudio.paused) ? window.__adaabSpeechAudio : audio;
                const t = activeAudio ? activeAudio.currentTime : (Date.now() * 0.001);
                const syllablePulse = Math.pow(Math.sin(t * 14.0 + Math.sin(t * 7.5)), 2);
                const vowelBody = Math.sin(t * 9.0 + i * 0.28) * 0.5 + 0.5;
                const consonantAttack = Math.sin(t * 22.0 + i * 0.45) * 0.5 + 0.5;
                const formantWeight = Math.exp(-Math.pow((norm - 0.38) / 0.28, 2));
                targetVal = Math.min(1.0, Math.max(0.08, (syllablePulse * 0.65 + vowelBody * 0.25 + consonantAttack * 0.10) * (0.45 + 0.55 * formantWeight)));
            }
        } else if (pttRecording && micAnalyser && micDataArray) {
            // User mic recording input
            try {
                micAnalyser.getByteFrequencyData(micDataArray);
                const bin = Math.min(micDataArray.length - 1, Math.floor(Math.pow(norm, 1.3) * (micDataArray.length - 1)));
                targetVal = Math.max(0.08, (micDataArray[bin] || 0) / 255.0);
            } catch(e) {}
        } else {
            // Calm resting idle state: gentle breathing baseline (never erratic)
            targetVal = 0.04 + 0.015 * Math.sin(Date.now() * 0.002 + i * 0.15);
        }

        const targetH = Math.max(3, targetVal * maxBarH);
        if (targetH > eqHeights[i]) {
            eqHeights[i] += (targetH - eqHeights[i]) * 0.52; // Fast attack
        } else {
            eqHeights[i] += (targetH - eqHeights[i]) * 0.14; // Natural smooth decay
        }

        if (eqHeights[i] >= eqPeaks[i]) {
            eqPeaks[i] = eqHeights[i];
        } else {
            eqPeaks[i] = Math.max(eqHeights[i], eqPeaks[i] - 1.2); // Gravity fall
        }
    }

    // Laser baseline
    ctx.beginPath();
    ctx.strokeStyle = isAiSpeaking ? 'rgba(16, 185, 129, 0.7)' : (pttRecording ? 'rgba(239, 68, 68, 0.7)' : 'rgba(16, 185, 129, 0.25)');
    ctx.lineWidth = 1.5;
    ctx.shadowBlur = isAiSpeaking ? 10 : (pttRecording ? 10 : 2);
    ctx.shadowColor = pttRecording ? '#ef4444' : '#10b981';
    ctx.moveTo(padding, baseline);
    ctx.lineTo(w - padding, baseline);
    ctx.stroke();

    // Equalizer bars with dynamic gradient and floating caps
    for (let i = 0; i < numEqBars; i++) {
        const x = padding + i * (barW + gap);
        const bh = Math.max(3, eqHeights[i]);
        const y = baseline - bh;

        const grad = ctx.createLinearGradient(x, baseline, x, baseline - maxBarH);
        if (pttRecording) {
            grad.addColorStop(0.00, '#ef4444');
            grad.addColorStop(0.50, '#f59e0b');
            grad.addColorStop(1.00, '#ffffff');
        } else if (isAiSpeaking) {
            grad.addColorStop(0.00, '#10b981'); // Emerald base
            grad.addColorStop(0.35, '#06b6d4'); // Electric cyan
            grad.addColorStop(0.70, '#eab308'); // Amber gold
            grad.addColorStop(1.00, '#ec4899'); // Neon magenta peak
        } else {
            // Calm resting emerald bars
            grad.addColorStop(0.00, 'rgba(16, 185, 129, 0.35)');
            grad.addColorStop(1.00, 'rgba(6, 182, 212, 0.15)');
        }

        ctx.save();
        ctx.fillStyle = grad;
        ctx.shadowBlur = isAiSpeaking ? 8 : (pttRecording ? 8 : 0);
        ctx.shadowColor = pttRecording ? '#ef4444' : '#06b6d4';
        const r = Math.min(barW / 2, 3);
        ctx.beginPath();
        ctx.moveTo(x, baseline);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.lineTo(x + barW - r, y);
        ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
        ctx.lineTo(x + barW, baseline);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // Floating peak LED cap (only displayed when speaking or recording)
        if (isAiSpeaking || pttRecording) {
            const peakY = baseline - Math.max(bh, eqPeaks[i]) - 3;
            ctx.save();
            ctx.fillStyle = '#ffffff';
            ctx.shadowColor = pttRecording ? '#ef4444' : '#eab308';
            ctx.shadowBlur = 6;
            ctx.fillRect(x, peakY, barW, 2);
            ctx.restore();
        }


        // Mirrored floor reflection
        if (isAiSpeaking) {
            const reflH = bh * 0.25;
            const reflGrad = ctx.createLinearGradient(x, baseline, x, baseline + reflH);
            reflGrad.addColorStop(0.0, 'rgba(16, 185, 129, 0.3)');
            reflGrad.addColorStop(1.0, 'rgba(16, 185, 129, 0.0)');
            ctx.fillStyle = reflGrad;
            ctx.fillRect(x, baseline + 2, barW, reflH);
        }
    }

    if (isAiSpeaking) {
        const t = Date.now() * 0.001;
        drawSineWave(w, h, baseline, '#00e5a0', 0.18, 2.5, t * 2.1, 0.55);
        drawSineWave(w, h, baseline, '#00c8ff', 0.12, 4.0, t * 1.7 + 1.2, 0.35);
        drawSineWave(w, h, baseline, '#7c3aed', 0.08, 6.0, t * 3.1 + 2.5, 0.22);
    } else if (pttRecording) {
        const t = Date.now() * 0.001;
        drawSineWave(w, h, baseline, '#ff3b5c', 0.25, 3.0, t * 3.5, 0.7);
    } else {
        const t = Date.now() * 0.001;
        drawSineWave(w, h, baseline, '#00e5a0', 0.03, 1.5, t * 0.5, 0.3);
    }
\n}

function drawNeonWaveform() {
    if (!ctx || !canvas) {
        if (initNeonCanvas()) {
            requestAnimationFrame(drawNeonWaveform);
        } else {
            setTimeout(drawNeonWaveform, 200);
        }
        return;
    }
    if (animWidth < 10) {
        resizeCanvas();
        if (animWidth < 10) { setTimeout(drawNeonWaveform, 300); return; }
    }

    const w = animWidth;
    const h = animHeight;

    ctx.clearRect(0, 0, w, h);
    drawEqualizerVisualizer(w, h);
    requestAnimationFrame(drawNeonWaveform);
}

function resetPTTButton() {
    pttRecording = false;
    pttStarting = false;
    targetLevel = 0.12;
    updateWaveBadge('IDLE', '#10b981');
    const btn = document.getElementById('ptt-btn');
    const icon = document.getElementById('ptt-btn-icon');
    const hint = document.getElementById('ptt-status-hint');
    if (btn) {
        btn.style.borderColor = '#10b981';
        btn.style.color = '#10b981';
        btn.style.background = 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)';
        btn.style.boxShadow = '0 0 12px rgba(16, 185, 129, 0.25)';
        btn.style.animation = '';
    }
    if (icon) {
        icon.src = ICON_MIC_DATA;
        icon.classList.remove('ptt-icon-spinning');
    }
    if (hint) {
        hint.innerHTML = '<img src="' + ICON_MIC_DATA + '" class="hint-icon-img" alt="" /> بولنے کے لیے مائیک دبائیں • Tap mic to speak';
    }
}

function hookAudioPlayer() {
    const playerWrapper = document.getElementById('voice_reply_player');
    if (!playerWrapper) return;
    const audio = playerWrapper.querySelector('audio');
    if (audio && !audio.__adaabHooked) {
        audio.__adaabHooked = true;

        function connectSpeaker() {
            try {
                const ctx = ensureAudioContext();
                if (!speakerAudioSource && audio && ctx && ctx.state !== 'closed') {
                    speakerAudioSource = ctx.createMediaElementSource(audio);
                    speakerAnalyser = ctx.createAnalyser();
                    speakerAnalyser.fftSize = 128;
                    speakerAnalyser.smoothingTimeConstant = 0.65;
                    speakerAudioSource.connect(speakerAnalyser);
                    speakerAnalyser.connect(ctx.destination);
                    speakerDataArray = new Uint8Array(speakerAnalyser.frequencyBinCount);
                }
            } catch(e) {}
        }

        function handleAudioStart() {
            isAiSpeaking = true;
            ensureAudioContext();
            connectSpeaker();
            updateWaveBadge('TABRAIZ SPEAKING • EQ', '#10b981');
            const btn = document.getElementById('ptt-btn');
            const icon = document.getElementById('ptt-btn-icon');
            const hint = document.getElementById('ptt-status-hint');
            if (btn && !pttRecording) {
                btn.style.borderColor = '#10b981';
                btn.style.color = '#10b981';
                btn.style.boxShadow = '0 0 16px rgba(16, 185, 129, 0.4)';
            }
            if (icon && !pttRecording) {
                icon.src = ICON_SPEAKER_DATA;
                icon.classList.remove('ptt-icon-spinning');
            }
            if (hint && !pttRecording) {
                hint.innerHTML = '<img src="' + ICON_SPEAKER_DATA + '" class="hint-icon-img" alt="" /> تبریز محوِ کلام ہیں... (Listening to Tabraiz)';
            }
            hideAutoplayPrompt();
        }

        function handleAudioStop() {
            isAiSpeaking = false;
            if (!pttRecording) {
                resetPTTButton();
            }
        }

        audio.addEventListener('play', handleAudioStart);
        audio.addEventListener('playing', handleAudioStart);
        audio.addEventListener('timeupdate', () => {
            if (!audio.paused && !audio.ended && audio.currentTime > 0) {
                if (!isAiSpeaking) handleAudioStart();
            }
        });
        audio.addEventListener('ended', handleAudioStop);
        audio.addEventListener('pause', () => {
            if (audio.ended) return;
            handleAudioStop();
        });
    }
}


function hookStatusBox() {
    const box = document.getElementById('ptt_status_box');
    if (!box || box.__adaabHooked) return;
    box.__adaabHooked = true;
    const observer = new MutationObserver(() => {
        // Wait for TTS audio to actually start before resetting mic button
        setTimeout(() => {
            const customPlaying = window.__adaabSpeechAudio && !window.__adaabSpeechAudio.paused;
            if (!pttRecording && !isAiSpeaking && !customPlaying) {
                resetPTTButton();
            }
        }, 1800);
    });
    observer.observe(box, { childList: true, subtree: true, characterData: true });
}

function unlockMobileAudio() {
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        // Play silent 0.01s buffer to prime iPhone Safari loudspeaker
        const buf = audioCtx.createBuffer(1, 1, 22050);
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        src.start(0);
    } catch(e) {}
}

async function requestMicPermissionUpfront() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    try {
        if (!pttStream) {
            pttStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
            });
            console.log('[Adaab Mic] Microphone permission active and ready.');
        }
    } catch(err) {
        console.log('[Adaab Mic] Permission prompt deferred until user taps mic:', err.name);
    }
}

async function handleMicToggle(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    unlockMobileAudio();
    ensureAudioContext();

    // If AI is speaking, stop playback immediately so user can speak cleanly
    if (window.__adaabSpeechAudio && !window.__adaabSpeechAudio.paused) {
        try {
            window.__adaabSpeechAudio.pause();
            window.__adaabSpeechAudio.currentTime = 0;
            isAiSpeaking = false;
        } catch(e) {}
    }

    if (pttRecording) {
        // Tap 2: Stop and send immediately!
        console.log('[Adaab Mic] Tap 2: Stopping and sending recording immediately...');
        finishAndSendPTT();
    } else {
        // Tap 1: Start recording!
        console.log('[Adaab Mic] Tap 1: Starting recording session...');
        await startRecordingSession();
    }
}

async function startRecordingSession() {
    if (pttStarting || pttRecording) return;
    pttStarting = true;

    const btn = document.getElementById('ptt-btn');
    const hint = document.getElementById('ptt-status-hint');

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Microphone access requires HTTPS (Cloudflare Tunnel or SSL).');
            pttStarting = false;
            resetPTTButton();
            return;
        }

        // Acquire or reuse microphone stream — will prompt user if not yet granted
        if (!pttStream || !pttStream.active || pttStream.getTracks().every(t => t.readyState === 'ended')) {
            pttStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: true, 
                    noiseSuppression: true, 
                    autoGainControl: true 
                } 
            });
            micSource = null;
        }

        const ctx = ensureAudioContext();
        if (ctx && !micSource && pttStream) {
            try {
                micSource = ctx.createMediaStreamSource(pttStream);
                micAnalyser = ctx.createAnalyser();
                micAnalyser.fftSize = 256;
                micDataArray = new Uint8Array(micAnalyser.frequencyBinCount);
                micSource.connect(micAnalyser);
            } catch(e) {
                console.log('[Adaab Mic] Analyser connection notice:', e);
            }
        }

        // Detect supported MIME type (Safari MP4, Chrome WebM)
        let mimeType = '';
        if (typeof MediaRecorder !== 'undefined') {
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mimeType = 'audio/mp4';
            }
        }

        pttChunks = [];
        pttRecorder = mimeType ? new MediaRecorder(pttStream, { mimeType }) : new MediaRecorder(pttStream);
        pttRecorder.ondataavailable = ev => {
            if (ev.data && ev.data.size > 0) pttChunks.push(ev.data);
        };
        pttRecorder.onstop = onPTTRecorderStop;

        pttRecorder.start();
        pttRecording = true;
        pttStarting = false;
        updateWaveBadge('RECORDING • بول رہے ہیں', '#ef4444');

        if (btn) {
            btn.style.background = 'linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%)';
            btn.style.borderColor = '#ef4444';
            btn.style.color = '#ffffff';
            btn.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.9)';
            btn.style.animation = 'ptt-pulse 1.2s infinite ease-in-out';
        }
        const icon = document.getElementById('ptt-btn-icon');
        if (icon) {
            icon.src = ICON_MIC_REC_DATA;
            icon.classList.remove('ptt-icon-spinning');
        }
        if (hint) {
            hint.innerHTML = '<img src="' + ICON_MIC_REC_DATA + '" class="hint-icon-img" alt="" /> ریکارڈنگ جاری ہے... روکنے اور بھیجنے کے لیے دوبارہ دبائیں (Recording... tap mic again to send)';
        }

    } catch(err) {
        console.warn('Microphone permission or hardware error:', err);
        pttStarting = false;
        pttRecording = false;
        resetPTTButton();
        if (hint) hint.innerText = 'براہ کرم براؤزر میں مائیکروفون کی اجازت دیں (Allow mic)';
        alert('مائیکروفون کی اجازت: برائے مہربانی اپنے براؤزر میں مائیکروفون کی اجازت (Allow) دیجیے۔');
    }
}

function finishAndSendPTT() {
    if (!pttRecording && !pttStarting) return;
    setControlsProcessing(true);
    pttRecording = false;
    pttStarting = false;
    targetLevel = 0.25;
    updateWaveBadge('THINKING • سوچ رہے ہیں', '#f59e0b');

    const btn = document.getElementById('ptt-btn');
    const icon = document.getElementById('ptt-btn-icon');
    const hint = document.getElementById('ptt-status-hint');

    if (btn) {
        btn.style.background = 'linear-gradient(135deg, #201400 0%, #0d0800 100%)';
        btn.style.borderColor = '#f59e0b';
        btn.style.color = '#f59e0b';
        btn.style.boxShadow = '0 0 16px rgba(245, 158, 11, 0.5)';
        btn.style.animation = '';
    }
    if (icon) {
        icon.src = ICON_SPINNER_DATA;
        icon.classList.add('ptt-icon-spinning');
    }
    if (hint) {
        hint.innerHTML = '<img src="' + ICON_SPINNER_DATA + '" class="hint-icon-img ptt-icon-spinning" alt="" /> تبریز سوچ رہے ہیں... (Tabraiz is thinking...)';
    }

    if (pttRecorder && pttRecorder.state === 'recording') {
        pttRecorder.stop();
    }

    // Safety timeout: reset button if server takes longer than 15s
    setTimeout(() => {
        if (!pttRecording && !isAiSpeaking) {
            resetPTTButton();
        }
    }, 15000);
}

window.__adaabSpeechAudio = null;
let currentPlayingBtn = null;
let lastAutoplayedAudioSrc = null;

window.playSpeechAudio = function(btn, audioSrc, isAutoplay) {
    if (!audioSrc && btn) {
        audioSrc = btn.getAttribute('data-audiosrc');
    }
    if (!audioSrc) return;
    unlockMobileAudio();
    ensureAudioContext();

    // Toggle pause if user clicked the currently playing button
    if (!isAutoplay && currentPlayingBtn === btn && window.__adaabSpeechAudio && !window.__adaabSpeechAudio.paused) {
        window.__adaabSpeechAudio.pause();
        return;
    }

    // Pause any currently playing audio
    if (window.__adaabSpeechAudio) {
        try {
            window.__adaabSpeechAudio.pause();
            window.__adaabSpeechAudio.currentTime = 0;
        } catch(e) {}
    }
    if (currentPlayingBtn) {
        currentPlayingBtn.classList.remove('playing');
        const lbl = currentPlayingBtn.querySelector('.replay-btn-label');
        if (lbl) lbl.textContent = 'دوبارہ سنیں (Replay Voice)';
    }

    if (!window.__adaabSpeechAudio) {
        window.__adaabSpeechAudio = new Audio();
        window.__adaabSpeechAudio.preload = 'auto';

        // Connect once to Web Audio context so the 36-bar music visualizer dances to it
        try {
            const ctx = ensureAudioContext();
            if (ctx && ctx.state !== 'closed') {
                const srcNode = ctx.createMediaElementSource(window.__adaabSpeechAudio);
                speakerAnalyser = ctx.createAnalyser();
                speakerAnalyser.fftSize = 128;
                speakerAnalyser.smoothingTimeConstant = 0.65;
                srcNode.connect(speakerAnalyser);
                speakerAnalyser.connect(ctx.destination);
                speakerDataArray = new Uint8Array(speakerAnalyser.frequencyBinCount);
            }
        } catch(err) {
            console.log('[Adaab Equalizer] MediaElementSource notice:', err);
        }

        window.__adaabSpeechAudio.addEventListener('play', () => {
            if (currentPlayingBtn) {
                const playSvg = currentPlayingBtn.querySelector('.play-svg');
                const pauseSvg = currentPlayingBtn.querySelector('.pause-svg');
                if (playSvg) playSvg.style.display = 'none';
                if (pauseSvg) pauseSvg.style.display = 'block';
            }
            isAiSpeaking = true;
            updateWaveBadge('TABRAIZ SPEAKING • EQ', '#10b981');
            const pttIcon = document.getElementById('ptt-btn-icon');
            if (pttIcon && !pttRecording) {
                pttIcon.src = ICON_SPEAKER_DATA;
                pttIcon.classList.remove('ptt-icon-spinning');
            }
            const hint = document.getElementById('ptt-status-hint');
            if (hint && !pttRecording) {
                hint.innerHTML = '<img src="' + ICON_SPEAKER_DATA + '" class="hint-icon-img" alt="" /> تبریز محوِ کلام ہیں... (Listening to Tabraiz)';
            }
            if (currentPlayingBtn) {
                currentPlayingBtn.classList.add('playing');
                const lbl = currentPlayingBtn.querySelector('.replay-btn-label');
                if (lbl) lbl.textContent = 'محوِ کلام... (Playing)';
            }
            hideAutoplayPrompt();
        });

        window.__adaabSpeechAudio.addEventListener('ended', () => {
            isAiSpeaking = false;
            if (currentPlayingBtn) {
                currentPlayingBtn.classList.remove('playing');
                const playSvg = currentPlayingBtn.querySelector('.play-svg');
                const pauseSvg = currentPlayingBtn.querySelector('.pause-svg');
                if (playSvg) playSvg.style.display = 'block';
                if (pauseSvg) pauseSvg.style.display = 'none';
                const lbl = currentPlayingBtn.querySelector('.replay-btn-label');
                if (lbl) lbl.textContent = 'دوبارہ سنیں (Replay Voice)';
                currentPlayingBtn = null;
            }
            if (!pttRecording) {
                resetPTTButton();
            }
        });

        window.__adaabSpeechAudio.addEventListener('pause', () => {
            if (window.__adaabSpeechAudio && window.__adaabSpeechAudio.ended) return;
            isAiSpeaking = false;
            if (currentPlayingBtn) {
                currentPlayingBtn.classList.remove('playing');
                const playSvg = currentPlayingBtn.querySelector('.play-svg');
                const pauseSvg = currentPlayingBtn.querySelector('.pause-svg');
                if (playSvg) playSvg.style.display = 'block';
                if (pauseSvg) pauseSvg.style.display = 'none';
                const lbl = currentPlayingBtn.querySelector('.replay-btn-label');
                if (lbl) lbl.textContent = 'دوبارہ سنیں (Replay Voice)';
                currentPlayingBtn = null;
            }
            if (!pttRecording) {
                resetPTTButton();
            }
        });
    }

    currentPlayingBtn = btn || null;
    window.__adaabSpeechAudio.src = audioSrc;

    const playPromise = window.__adaabSpeechAudio.play();
    if (playPromise !== undefined) {
        playPromise.then(() => {
            console.log('[Adaab Audio] Speech playing successfully on its own!');
            isAiSpeaking = true;
        }).catch((err) => {
            console.warn('[Adaab Audio] Autoplay deferred by browser policy:', err.name);
            if (currentPlayingBtn) {
                currentPlayingBtn.classList.remove('playing');
                const lbl = currentPlayingBtn.querySelector('.replay-btn-label');
                if (lbl) lbl.textContent = 'دوبارہ سنیں (Replay Voice)';
            }
        });
    }
};


function setControlsProcessing(isProcessing) {
    const micBtn = document.getElementById('ptt-btn');
    const textInput = document.querySelector('#message-input textarea, #message-input input');
    const sendBtn = document.getElementById('send-btn');
    const dock = document.querySelector('.messenger-dock');

    if (isProcessing) {
        if (dock) dock.classList.add('dock-processing');
        if (micBtn) {
            micBtn.disabled = true;
            micBtn.classList.add('disabled-processing');
            micBtn.style.pointerEvents = 'none';
            micBtn.title = 'تبریز سوچ رہے ہیں... (Tabraiz is thinking)';
        }
        if (textInput) {
            textInput.disabled = true;
            textInput.classList.add('disabled-processing');
            textInput.style.pointerEvents = 'none';
            if (!textInput.dataset.origPlaceholder) {
                textInput.dataset.origPlaceholder = textInput.placeholder;
            }
            textInput.placeholder = 'تبریز سوچ رہے ہیں... (Tabraiz is thinking...)';
        }
        if (sendBtn) {
            sendBtn.disabled = true;
            sendBtn.classList.add('disabled-processing');
            sendBtn.style.pointerEvents = 'none';
        }
    } else {
        if (dock) dock.classList.remove('dock-processing');
        if (micBtn) {
            micBtn.disabled = false;
            micBtn.classList.remove('disabled-processing');
            micBtn.style.pointerEvents = 'auto';
            micBtn.title = 'بولنے کے لیے ٹیپ کریں (Tap to speak / Tap again to send)';
        }
        if (textInput) {
            textInput.disabled = false;
            textInput.classList.remove('disabled-processing');
            textInput.style.pointerEvents = 'auto';
            if (textInput.dataset.origPlaceholder) {
                textInput.placeholder = textInput.dataset.origPlaceholder;
            } else {
                textInput.placeholder = '...اپنا پیغام یہاں لکھیں';
            }
            // Auto focus back to input when done
            setTimeout(() => { textInput.focus(); }, 150);
        }
        if (sendBtn) {
            sendBtn.disabled = false;
            sendBtn.classList.remove('disabled-processing');
            sendBtn.style.pointerEvents = 'auto';
        }
    }
}

// Watch for Send click and Enter key to instantly lock controls
function attachSendInterceptors() {
    const sendBtn = document.getElementById('send-btn');
    const textInput = document.querySelector('#message-input textarea, #message-input input');
    if (sendBtn && !sendBtn.__lockAttached) {
        sendBtn.__lockAttached = true;
        sendBtn.addEventListener('click', () => {
            if (textInput && textInput.value.trim().length > 0) {
                setControlsProcessing(true);
            }
        });
    }
    if (textInput && !textInput.__lockAttached) {
        textInput.__lockAttached = true;
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && textInput.value.trim().length > 0) {
                setControlsProcessing(true);
            }
        });
    }
}
setTimeout(attachSendInterceptors, 300);
setInterval(attachSendInterceptors, 2000);

function setupChatObserver() {
    const chatContainer = document.querySelector('.messenger-chat') || document.querySelector('[data-testid="chatbot"]');
    if (!chatContainer) {
        setTimeout(setupChatObserver, 300);
        return;
    }
    if (chatContainer.__adaabObserved) return;
    chatContainer.__adaabObserved = true;

    console.log('[Adaab] Chatbot MutationObserver initialized for automated voice playback.');

    const checkAndPlayLatest = () => {
        // Smooth auto-scroll to bottom so mobile user immediately sees the response
        try {
            const scrollEls = [chatContainer, chatContainer.querySelector('.wrap'), chatContainer.querySelector('.messages')];
            scrollEls.forEach(el => {
                if (el && el.scrollHeight > el.clientHeight) {
                    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                }
            });
        } catch(e) {}

        // Re-enable controls as soon as assistant response arrives
        setControlsProcessing(false);

        const replayBtns = chatContainer.querySelectorAll('.bot-audio-player-pill, .bot-audio-replay-btn');
        if (replayBtns.length > 0) {
            const latestBtn = replayBtns[replayBtns.length - 1];
            const audioSrc = latestBtn.getAttribute('data-audiosrc');
            if (audioSrc && audioSrc !== lastAutoplayedAudioSrc) {
                lastAutoplayedAudioSrc = audioSrc;
                console.log('[Adaab AutoPlay] Bot speech ready! Auto-playing on its own...');
                setTimeout(() => {
                    window.playSpeechAudio(latestBtn, audioSrc, true);
                }, 80);
            }
        }
    };

    const observer = new MutationObserver(checkAndPlayLatest);
    observer.observe(chatContainer, { childList: true, subtree: true });
    setTimeout(checkAndPlayLatest, 400);
}

function onPTTRecorderStop() {
    if (pttChunks.length === 0) {
        console.warn('[Adaab PTT] No audio chunks captured.');
        resetPTTButton();
        return;
    }
    const finalMime = (pttRecorder && pttRecorder.mimeType) || 'audio/mp4';
    const audioBlob = new Blob(pttChunks, { type: finalMime });
    console.log('[Adaab PTT] Audio captured:', audioBlob.size, 'bytes, MIME:', finalMime);

    if (audioBlob.size < 200) {
        console.warn('[Adaab PTT] Audio too short.');
        resetPTTButton();
        return;
    }

    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => {
        const b64 = reader.result;
        console.log('[Adaab PTT] Base64 encoded length:', b64 ? b64.length : 0);
        window.__latestRecordedAudioB64 = b64;

        const area = document.querySelector('#ptt_raw_input textarea') || document.querySelector('#ptt_raw_input input');
        if (area) {
            area.value = b64;
            area.dispatchEvent(new Event('input', { bubbles: true }));
            area.dispatchEvent(new Event('change', { bubbles: true }));
        }

        setTimeout(() => {
            const trigger = document.querySelector('#ptt_trigger_btn button') || document.getElementById('ptt_trigger_btn');
            if (trigger) {
                console.log('[Adaab PTT] Triggering Gradio request...');
                trigger.click();
            } else {
                console.error('[Adaab PTT] Trigger button not found!');
            }
        }, 40);
    };
}

window.addEventListener('touchstart', unlockMobileAudio, { passive: true });
window.addEventListener('click', unlockMobileAudio, { passive: true });

window.addEventListener('resize', () => { resizeCanvas(); });

// MutationObserver: detect when #neonWaveCanvas is inserted into the DOM (Gradio lazy render)
const _waveObserver = new MutationObserver(() => {
    if (!canvas && document.getElementById('neonWaveCanvas')) {
        initNeonCanvas();
        // IntersectionObserver: re-resize when the canvas becomes visible (tab switch)
        if (canvas && window.IntersectionObserver) {
            new IntersectionObserver((entries) => {
                entries.forEach(e => {
                    if (e.isIntersecting) { resizeCanvas(); }
                });
            }, { threshold: 0.1 }).observe(canvas);
        }
    }
});
_waveObserver.observe(document.body || document.documentElement, { childList: true, subtree: true });

// Kick off waveform loop — will retry every 200ms internally until canvas is found
setTimeout(() => { drawNeonWaveform(); }, 100);
// Also try after Gradio finish rendering (usually ~1s)
setTimeout(() => { if (!canvas) { initNeonCanvas(); } resizeCanvas(); }, 1200);

let greetingAutoplayDone = false;
// Guard prevents initAutoplayController and setupChatObserver from double-playing
let greetingAttemptInProgress = false;

function showAutoplayPrompt() {
    let pill = document.getElementById('autoplay-welcome-pill');
    if (!pill) {
        pill = document.createElement('div');
        pill.id = 'autoplay-welcome-pill';
        pill.className = 'autoplay-welcome-pill';
        pill.innerHTML = '<span class="pulse-sound-dot"></span> <img src="' + ICON_SPEAKER_DATA + '" class="hint-icon-img" style="width:16px;height:16px;vertical-align:middle;display:inline-block;margin-left:4px;" alt="" /> تعارفی پیغام سننے کے لیے ٹیپ فرمائیں (Tap to hear Tabraiz welcome you)';
        pill.onclick = (e) => {
            e.stopPropagation();
            greetingAutoplayDone = false; // allow retry
            greetingAttemptInProgress = false;
            startWelcomeAudioNow();
        };
        const target = document.querySelector('.messenger-wrapper') || document.body;
        if (target) target.insertBefore(pill, target.firstChild);
    }
}

function hideAutoplayPrompt() {
    const pill = document.getElementById('autoplay-welcome-pill');
    if (pill) {
        pill.style.opacity = '0';
        pill.style.transform = 'translateY(-8px)';
        setTimeout(() => { if (pill && pill.parentNode) pill.parentNode.removeChild(pill); }, 300);
    }
}

function startWelcomeAudioNow() {
    if (greetingAutoplayDone || greetingAttemptInProgress) return;
    greetingAttemptInProgress = true;
    unlockMobileAudio();
    ensureAudioContext();

    // Primary path: play directly from the initial greeting bubble's data-audiosrc
    const firstReplayBtn = document.querySelector('.bot-audio-replay-btn');
    if (firstReplayBtn) {
        const src = firstReplayBtn.getAttribute('data-audiosrc');
        if (src) {
            console.log('[Adaab Welcome] Playing initial greeting via playSpeechAudio...');
            lastAutoplayedAudioSrc = src; // prevent MutationObserver double-play
            const playPromise = window.playSpeechAudio(firstReplayBtn, src, true);
            // playSpeechAudio is synchronous start; mark done after it resolves audio element play
            const checkPlaying = setInterval(() => {
                if (window.__adaabSpeechAudio && !window.__adaabSpeechAudio.paused) {
                    clearInterval(checkPlaying);
                    greetingAutoplayDone = true;
                    greetingAttemptInProgress = false;
                    hideAutoplayPrompt();
                }
            }, 150);
            setTimeout(() => {
                // After 2s if still not playing, browser policy blocked it — show pill
                if (!greetingAutoplayDone) {
                    clearInterval(checkPlaying);
                    greetingAttemptInProgress = false;
                    showAutoplayPrompt();
                }
            }, 2000);
            return;
        }
    }
    // If no replay button found yet, release lock so next attempt can try
    greetingAttemptInProgress = false;
}

function initAutoplayController() {
    let attempts = 0;
    const timer = setInterval(() => {
        attempts++;
        if (greetingAutoplayDone) { clearInterval(timer); return; }
        const firstReplayBtn = document.querySelector('.bot-audio-replay-btn');
        if (firstReplayBtn && firstReplayBtn.getAttribute('data-audiosrc')) {
            clearInterval(timer);
            startWelcomeAudioNow();
            return;
        }
        if (attempts > 60) {
            clearInterval(timer);
            if (!greetingAutoplayDone) showAutoplayPrompt();
        }
    }, 200);
}

// Any tap/click anywhere on the screen immediately plays greeting if still waiting
['pointerdown', 'touchstart', 'click', 'keydown'].forEach(evt => {
    window.addEventListener(evt, () => {
        if (!greetingAutoplayDone) {
            greetingAttemptInProgress = false; // reset lock so tap can retry
            startWelcomeAudioNow();
        }
    }, { passive: true, once: true });
});

window.toggleNotificationPanel = function() {
    let panel = document.getElementById('adaab-notification-modal');
    if (panel) {
        panel.style.display = (panel.style.display === 'none' || !panel.style.display) ? 'flex' : 'none';
        return;
    }
    panel = document.createElement('div');
    panel.id = 'adaab-notification-modal';
    panel.className = 'adaab-notification-popover';
    panel.innerHTML = `
        <div class="notif-header">
            <div class="notif-title-row">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#00d2ff" stroke-width="2">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                    <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                </svg>
                <span>اطلاعات و اعلانات (Notifications)</span>
            </div>
            <button class="notif-close-btn" onclick="document.getElementById('adaab-notification-modal').style.display='none'">&times;</button>
        </div>
        <div class="notif-list">
            <div class="notif-item">
                <div class="notif-dot-active"></div>
                <div class="notif-content">
                    <div class="notif-item-title">نظام مکمل طور پر فعال ہے</div>
                    <div class="notif-item-desc">تبریز اے آئی اسسٹنٹ، نیورل وائس اور نالج سرچ سروسز مکمل تیار ہیں۔</div>
                    <span class="notif-time">ابھی ابھی</span>
                </div>
            </div>
            <div class="notif-item">
                <div class="notif-dot-active"></div>
                <div class="notif-content">
                    <div class="notif-item-title">معیاری صوتی ہدایات لاگو ہیں</div>
                    <div class="notif-item-desc">تمام جوابات شائستہ، مناسب اور جامع خلاصے (2 سے 3 جملوں) میں پیش کیے جائیں گے۔</div>
                    <span class="notif-time">مستقل فعال</span>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(panel);
    document.addEventListener('click', function(e) {
        const btn = document.getElementById('sidebar-notification-btn');
        if (panel && panel.style.display !== 'none' && !panel.contains(e.target) && (!btn || !btn.contains(e.target))) {
            panel.style.display = 'none';
        }
    });
};

setTimeout(initAutoplayController, 250);
// setupChatObserver: attach once and stay attached — no need to poll
setTimeout(setupChatObserver, 300);

// hookAudioPlayer: attach once when the audio element mounts
setTimeout(hookAudioPlayer, 600);

// hookStatusBox: attach once after DOM settles
setTimeout(hookStatusBox, 800);
</script>


"""

CUSTOM_CSS = CUSTOM_CSS.replace("__ICON_CHAT__", ICON_CHAT).replace("__ICON_MEMORY__", ICON_MEMORY).replace("__ICON_MOBILE__", ICON_MOBILE).replace("__ICON_GEAR__", ICON_GEAR)

HEAD_JS = HEAD_JS.replace("__ICON_MIC__", ICON_MIC).replace("__ICON_MIC_REC__", ICON_MIC_REC).replace("__ICON_SPINNER__", ICON_SPINNER).replace("__ICON_SPEAKER__", ICON_SPEAKER)

TYPING_INDICATOR = '<div class="adaab-typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>'

def text_chat_fn(message, history, style_choice, voice_choice="male", profile_state=None):
    if not message.strip():
        return history, "", None, profile_state or {}

    voice_choice = voice_choice or "male"
    persona = "Tabraiz"
    history = list(history or [])
    profile_state = profile_state or SESSION_STATE.get("active_profile") or {}

    # Append formatted user message matching tablet mockup
    history.append({"role": "user", "content": format_user_message(message)})

    bot_reply, updated_profile = handle_conversation_turn(
        user_message=message,
        history=history,
        client=client,
        persona=persona,
        style_name=style_choice,
        active_profile=profile_state
    )
    if updated_profile:
        profile_state = updated_profile
        SESSION_STATE["active_profile"] = updated_profile

    # Generate spoken voice audio for the reply
    audio_out = None
    audio_uri_encoded = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            audio_out = tf.name
        generate_conversation_speech(bot_reply, audio_out, voice=voice_choice)
        audio_uri_encoded = get_audio_data_uri(audio_out)
    except Exception as err:
        print(f"[App] Audio synthesis notice: {err}")
        audio_out = None
    finally:
        # A9: clean up temp file — audio is already in-memory as base64
        if audio_out and os.path.exists(audio_out):
            try:
                os.unlink(audio_out)
            except Exception:
                pass
        audio_out = None  # don't pass file path; audio is embedded in HTML

    bot_formatted = format_assistant_message(bot_reply, None, prebuilt_uri=audio_uri_encoded)
    history.append({"role": "assistant", "content": bot_formatted})

    return history, "", None, profile_state


def ptt_voice_fn(base64_audio_data, history, style_choice, voice_choice="male", profile_state=None):
    """Processes audio recorded from the Push-to-Talk button."""
    voice_choice = voice_choice or "male"
    persona = "Tabraiz"
    history = list(history or [])
    profile_state = profile_state or SESSION_STATE.get("active_profile") or {}

    payload_len = len(base64_audio_data) if base64_audio_data else 0
    print(f"\n[Adaab PTT] >>> New Voice Turn Received from client! (Payload: {payload_len} chars)", flush=True)

    if not base64_audio_data or not base64_audio_data.strip():
        print("[Adaab PTT] Notice: Empty audio payload received from browser.", flush=True)
        empty_reply = "کوئی آواز موصول نہیں ہوئی۔ براہ کرم مائیک بٹن پر ٹیپ کر کے بولیں اور مکمل ہونے پر دوبارہ ٹیپ فرمائیں۔"
        history.append({"role": "assistant", "content": format_assistant_message(empty_reply)})
        return history, "Empty audio", None, profile_state

    print("[Adaab PTT] Transcribing audio with Google Speech ASR...", flush=True)
    asr_res = transcribe_base64_audio(base64_audio_data)
    print(f"[Adaab PTT] ASR Result: {asr_res}", flush=True)

    if not asr_res.get("success"):
        error_msg = asr_res.get("error", "آواز واضح نہیں تھی")
        asr_err_reply = f"آواز کی شناخت نہیں ہو سکی ({error_msg})۔ براہ کرم مائیک پر ٹیپ کر کے واضح انداز میں دوبارہ بولیں۔"
        history.append({"role": "assistant", "content": format_assistant_message(asr_err_reply)})
        return history, f"ASR Error: {error_msg}", None, profile_state

    user_transcript = asr_res["text"]
    print(f"[Adaab PTT] User transcript: '{user_transcript}'", flush=True)
    print(f"[Adaab PTT] Dispatching to Modal GPU Backend (Persona: {persona})...", flush=True)

    history.append({"role": "user", "content": format_user_message(user_transcript)})

    bot_reply, updated_profile = handle_conversation_turn(
        user_message=user_transcript,
        history=history,
        client=client,
        persona=persona,
        style_name=style_choice,
        active_profile=profile_state
    )
    if updated_profile:
        profile_state = updated_profile
        SESSION_STATE["active_profile"] = updated_profile

    print(f"[Adaab PTT] Bot reply generated: '{bot_reply[:80]}...'", flush=True)

    # Synthesize and immediately encode to base64; delete temp file
    audio_uri_encoded = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            reply_audio_path = tf.name
        generate_conversation_speech(bot_reply, reply_audio_path, voice=voice_choice)
        print(f"[Adaab PTT] Voice reply audio ready, encoding...", flush=True)
        audio_uri_encoded = get_audio_data_uri(reply_audio_path)
    except Exception as err:
        print(f"[Adaab PTT] Voice synthesis notice: {err}", flush=True)
        audio_uri_encoded = None
    finally:
        # A9: delete temp file immediately after base64 encode
        try:
            if 'reply_audio_path' in dir() and reply_audio_path and os.path.exists(reply_audio_path):
                os.unlink(reply_audio_path)
        except Exception:
            pass

    bot_formatted = format_assistant_message(bot_reply, None, prebuilt_uri=audio_uri_encoded)
    history.append({"role": "assistant", "content": bot_formatted})

    status_msg = f"{persona}: {bot_reply}"
    return history, status_msg, None, profile_state


def build_app(active_port: int = 7865, public_url: str = None):
    with gr.Blocks(title="آداب — تبریز | Adaab Voice AI") as demo:
        with gr.Row(elem_classes=["adaab-tablet-frame"]):
            # Left Navigation Sidebar (Exact match to Mockup)
            with gr.Column(elem_classes=["adaab-sidebar"], scale=1, min_width=190):
                sidebar_html = gr.HTML(
                    f"""
                    <div class="sidebar-inner">
                        <div class="adaab-sidebar-logo">
                            <div class="sidebar-mic-glow-wrap" title="آداب صوتی اسٹوڈیو">
                                <div class="sidebar-soundwaves-left">
                                    <span class="sw-bar"></span><span class="sw-bar"></span>
                                </div>
                                <svg class="sidebar-mic-svg" viewBox="0 0 24 24" width="24" height="24" fill="#00d2ff">
                                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                                    <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                                </svg>
                                <div class="sidebar-soundwaves-right">
                                    <span class="sw-bar"></span><span class="sw-bar"></span>
                                </div>
                            </div>
                        </div>

                                                <nav class="sidebar-nav-menu">
                            <a href="javascript:void(0)" class="nav-item active" id="sidebar-notification-btn" title="اطلاعات (Notification)" onclick="toggleNotificationPanel()">
                                <div class="nav-icon-badge-wrap">
                                    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                                    </svg>
                                    <span class="nav-badge-dot"></span>
                                </div>
                                <span>Notification</span>
                            </a>
                        </nav>
                    </div>
                    """
                )

            # Main Encapsulated Rounded Chat Card
            with gr.Column(elem_classes=["adaab-main-card"], scale=6):
                chat_style = gr.State(value="standard")
                chat_voice = gr.State(value="male")
                profile_state = gr.State(value={})

                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": format_assistant_message(TABRAIZ_INITIAL_GREETING, TABRAIZ_OPENING_AUDIO)}],
                    avatar_images=(PATH_USER_AVATAR, PATH_BOT_AVATAR),
                    elem_classes=["messenger-chat"],
                    label="گفتگو",
                    show_label=False,
                    sanitize_html=False,
                    buttons=[]
                )

                voice_reply_player = gr.Audio(
                    value=None,
                    autoplay=False,
                    elem_id="voice_reply_player",
                    elem_classes=["compact-audio-player"],
                    visible=False
                )

                with gr.Row(elem_classes=["messenger-dock"]):
                    ptt_btn_html = gr.HTML(
                        f"""
                        <button id="ptt-btn" 
                                type="button"
                                title="بولنے کے لیے ٹیپ کریں (Tap to speak / Tap again to send)"
                                onclick="handleMicToggle(event)">
                            <svg id="ptt-mic-svg" class="ptt-mic-svg" viewBox="0 0 24 24" width="22" height="22" fill="#00d2ff">
                                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                            </svg>
                        </button>
                        """
                    )
                    text_msg = gr.Textbox(
                        placeholder="...اپنا پیغام یہاں لکھیں", 
                        elem_id="message-input",
                        show_label=False,
                        container=False,
                        scale=8,
                        lines=1,
                        max_lines=3
                    )
                    send_btn = gr.Button("Send", elem_id="send-btn", scale=1, variant="primary")

                # Hidden bridge controls for PTT JavaScript trigger & clean reset
                ptt_raw_input = gr.Textbox(elem_id="ptt_raw_input", elem_classes=["hidden-control"])
                ptt_trigger_btn = gr.Button("TRIGGER_PTT", elem_id="ptt_trigger_btn", elem_classes=["hidden-control"])
                ptt_status_box = gr.Textbox(elem_id="ptt_status_box", elem_classes=["hidden-control"])
                hidden_clear_btn = gr.Button("CLEAR", elem_id="hidden_clear_btn", elem_classes=["hidden-control"])

        # Event connections
        ptt_trigger_btn.click(
            fn=ptt_voice_fn,
            inputs=[ptt_raw_input, chatbot, chat_style, chat_voice, profile_state],
            outputs=[chatbot, ptt_status_box, voice_reply_player, profile_state],
            js="""(raw_audio, chatbot, style, voice, profile) => {
                const audio = window.__latestRecordedAudioB64 || raw_audio || '';
                console.log('[Adaab PTT JS Event] Sending audio to server, length:', audio.length);
                return [audio, chatbot, style, voice, profile];
            }"""
        )

        send_btn.click(
            fn=text_chat_fn,
            inputs=[text_msg, chatbot, chat_style, chat_voice, profile_state],
            outputs=[chatbot, text_msg, voice_reply_player, profile_state]
        )
        text_msg.submit(
            fn=text_chat_fn,
            inputs=[text_msg, chatbot, chat_style, chat_voice, profile_state],
            outputs=[chatbot, text_msg, voice_reply_player, profile_state]
        )

        hidden_clear_btn.click(
            fn=lambda: ([{"role": "assistant", "content": format_assistant_message(TABRAIZ_INITIAL_GREETING, TABRAIZ_OPENING_AUDIO)}], "", {}),
            inputs=None,
            outputs=[chatbot, text_msg, profile_state]
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tehzeeb AI Studio")
    parser.add_argument("--localtunnel", "--branded", dest="localtunnel", action="store_true", help="Launch via Localtunnel with branded URL (https://adaab-studio.loca.lt)")
    parser.add_argument("--tunnel", "--cloudflare", dest="tunnel", action="store_true", help="Launch via Cloudflare Quick Tunnel (bypasses firewalls, valid SSL, zero mobile warnings)")
    parser.add_argument("--http", action="store_true", help="Launch in plain HTTP mode (default is HTTPS with self-signed SSL for mobile mic)")
    parser.add_argument("--port", type=int, default=7865, help="Port to bind (default: 7865)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link (frpc)")
    args = parser.parse_args()

    active_port = resolve_available_port(args.port, auto_free_stale=True)
    enable_localtunnel = args.localtunnel or os.environ.get("ADAAB_LOCALTUNNEL", "0") == "1"
    enable_tunnel = args.tunnel or os.environ.get("ADAAB_TUNNEL", "0") == "1"
    enable_share = args.share or os.environ.get("ADAAB_SHARE", "0") == "1"

    tunnel_url = None
    if enable_localtunnel:
        print("[Adaab] Starting Branded Localtunnel (https://adaab-studio.loca.lt)...", flush=True)
        tunnel_url, lt_proc = start_localtunnel(active_port, subdomain="adaab-studio")
        if tunnel_url:
            print(f"[Adaab] [OK] Branded URL active: {tunnel_url}", flush=True)
    elif enable_tunnel:
        print("[Adaab] Starting Cloudflare Secure Tunnel (firewall-proof)...", flush=True)
        tunnel_url, cf_proc = start_cloudflared_tunnel(active_port)
        if tunnel_url and is_valid_cloudflare_tunnel_url(tunnel_url):
            print(f"[Adaab] [OK] Cloudflare Tunnel active: {tunnel_url}", flush=True)
        else:
            if tunnel_url:
                print(f"[Adaab] [Notice] Rejected internal tunnel URL: {tunnel_url}", flush=True)
            tunnel_url = None

    use_ssl = (not args.http and not enable_tunnel and not enable_localtunnel and os.environ.get("ADAAB_HTTP", "0") != "1")
    local_ip = get_local_lan_ip()
    protocol = "https" if use_ssl else "http"
    local_url = f"{protocol}://localhost:{active_port}"
    mobile_url = tunnel_url or f"{protocol}://{local_ip}:{active_port}"

    cert_path, key_path = None, None
    if use_ssl and not enable_share and not enable_tunnel and not enable_localtunnel:
        cert_path, key_path = ensure_self_signed_cert()

    print("\n" + "=" * 70, flush=True)
    print(" آداب — تبریز (Adaab AI Studio) فعال ہو رہا ہے! ".center(70, "="), flush=True)
    print(f" [PC]     مقامی پی سی ربط (Local PC):   {local_url}", flush=True)
    print(f" [Mobile] موبائل فون ربط (Mobile URL):  {mobile_url}", flush=True)
    if enable_localtunnel and tunnel_url:
        print(f" [Branded Link] موڈ (Branded Link Mode — {tunnel_url})", flush=True)
    elif tunnel_url:
        print(" [Cloudflare] کلاؤڈ فلیئر ٹنل موڈ (Cloudflare Verified SSL — Zero firewall blocks & instant mobile mic!)", flush=True)
    elif enable_share:
        print(" [Public] پبلک شیئر موڈ (Public Share Mode - Let's Encrypt Verified SSL)", flush=True)
    elif not use_ssl:
        print(" [HTTP] موڈ (Plain HTTP Mode - ideal for USB localhost port forwarding).", flush=True)
    else:
        print(" [Notice] موبائل فون نوٹس: چونکہ لوکل آئی پی پر خود ساختہ SSL ہے، اس لیے فون پر 'Show Details -> Visit Website' دبائیں۔", flush=True)
    print("=" * 70, flush=True)
    print(" [QR Code] اپنے موبائل فون سے اسکین کرنے کے لیے کیو آر کوڈ:", flush=True)
    try:
        qr_ascii = generate_qr_code_ascii(mobile_url)
        if qr_ascii:
            print(qr_ascii, flush=True)
    except Exception as e:
        print(f"QR render notice: {e}", flush=True)
    # Start background GPU keep-alive heartbeat (every 48s) to maintain warm container while terminal is open
    client.start_heartbeat(interval_sec=48)

    app = build_app(active_port=active_port, public_url=tunnel_url)
    app.launch(
        server_name="0.0.0.0",
        server_port=active_port,
        ssl_certfile=cert_path if (use_ssl and not enable_share and not enable_tunnel) else None,
        ssl_keyfile=key_path if (use_ssl and not enable_share and not enable_tunnel) else None,
        ssl_verify=False,
        share=enable_share,
        theme=STUDIO_THEME,
        css=CUSTOM_CSS,
        head=HEAD_JS,
        inbrowser=True,
        allowed_paths=[os.path.dirname(os.path.abspath(__file__))]
    )

