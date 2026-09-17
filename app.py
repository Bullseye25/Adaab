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

def format_assistant_message(urdu_text: str, audio_path: str | None = None, prebuilt_uri: str | None = None) -> str:
    """
    Renders assistant response with Urdu Nastaliq text and an interactive,
    zero-emoji replay button placed directly under the response text.
    Accepts either a file path (audio_path) or a pre-encoded base64 URI (prebuilt_uri).
    """
    audio_uri = prebuilt_uri or get_audio_data_uri(audio_path)
    if audio_uri:
        return (
            f'<div class="bot-msg-card">'
            f'<div class="bot-msg-text">{urdu_text}</div>'
            f'<div class="bot-audio-replay-row">'
            f'<button type="button" class="bot-audio-replay-btn" onclick="window.playSpeechAudio(this)" data-audiosrc="{audio_uri}">'
            f'<img src="{ICON_SPEAKER}" class="replay-btn-icon" alt="" />'
            f'<span class="replay-btn-label">دوبارہ سنیں (Replay Voice)</span>'
            f'</button>'
            f'<span class="bot-audio-duration-pill">تبریز صوتی کلام</span>'
            f'</div>'
            f'</div>'
        )
    return urdu_text

client = AdaabClient()
STUDIO_THEME = gr.themes.Base()

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;700&family=Share+Tech+Mono&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #10b981;
    --primary-dark: #059669;
    --primary-light: #34d399;
    --bg-dark: #090d16;
    --card-bg: #111827;
    --bubble-bot: #1e293b;
    --bubble-user: #047857;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --border-color: rgba(255, 255, 255, 0.08);
}

body, .gradio-container {
    background: #090d16 !important;
    background-image: radial-gradient(at 50% 0%, #0f172a 0%, #090d16 100%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    min-height: 100vh;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Centered Messenger Frame ── */
.messenger-wrapper {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding: 10px 12px 24px 12px !important;
}

/* ── Top Header Contact Card ── */
.messenger-header {
    background: #111827 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 12px 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}
.messenger-contact-info {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
}
.messenger-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, #065f46 0%, #047857 100%);
    border: 2px solid #10b981;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    position: relative;
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.35);
}
.online-dot {
    position: absolute;
    bottom: 1px;
    right: 1px;
    width: 12px;
    height: 12px;
    background: #10b981;
    border: 2px solid #111827;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: pulse-green 2s infinite ease-in-out;
}
@keyframes pulse-green {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.8; }
}
.messenger-name {
    font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.8;
}
.messenger-status {
    font-family: 'Share Tech Mono', 'Inter', sans-serif;
    font-size: 0.76rem;
    color: #34d399;
    letter-spacing: 0.5px;
}

/* ── Messenger Controls Row ── */
.messenger-controls-row {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 6px 12px !important;
    margin-bottom: 10px !important;
    gap: 12px !important;
}

/* ── Anti-Stretching & Aspect Ratio Lockdown ── */
img, 
.messenger-avatar-img, 
.ptt-icon-img, 
.send-icon-img, 
.btn-icon-img, 
.hint-icon-img,
button img,
.gr-button img {
    aspect-ratio: 1 / 1 !important;
    object-fit: contain !important;
    flex-shrink: 0 !important;
}

.messenger-avatar-img {
    width: 44px !important;
    height: 44px !important;
    border-radius: 50% !important;
    display: block !important;
}

.ptt-icon-img {
    width: 24px !important;
    height: 24px !important;
    display: block !important;
    margin: auto !important;
    transition: transform 0.2s ease, opacity 0.2s ease !important;
}

.ptt-icon-spinning {
    animation: icon-spin 0.9s linear infinite !important;
}

@keyframes icon-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.hint-icon-img {
    width: 14px !important;
    height: 14px !important;
    display: inline-block !important;
    vertical-align: -2px !important;
    margin-right: 4px !important;
}

.status-pulse-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    margin-left: 6px;
    vertical-align: 1px;
    animation: pulse-green 2s infinite ease-in-out;
}

/* ── Autoplay Helper Pill ── */
.autoplay-welcome-pill {
    background: linear-gradient(135deg, #065f46 0%, #047857 100%) !important;
    border: 1px solid #10b981 !important;
    border-radius: 24px !important;
    padding: 8px 18px !important;
    margin-bottom: 12px !important;
    color: #ffffff !important;
    font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    text-align: center !important;
    cursor: pointer !important;
    box-shadow: 0 4px 18px rgba(16, 185, 129, 0.45) !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    animation: pill-glow 2s infinite ease-in-out !important;
    user-select: none !important;
}
.autoplay-welcome-pill:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 24px rgba(16, 185, 129, 0.65) !important;
}
.pulse-sound-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 8px #ffffff;
    animation: pulse-green 1.4s infinite ease-in-out;
}
@keyframes pill-glow {
    0%, 100% { box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4); }
    50% { box-shadow: 0 4px 26px rgba(16, 185, 129, 0.85); transform: translateY(-1px); }
}


/* Button icons styling - non-stretched and correctly proportioned */
button img, .gr-button img {
    width: 18px !important;
    height: 18px !important;
    max-width: 18px !important;
    max-height: 18px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin-left: 6px !important;
    margin-right: 2px !important;
}

/* Send button image styling inside circular 44px dock button */
#send-btn img {
    width: 22px !important;
    height: 22px !important;
    max-width: 22px !important;
    max-height: 22px !important;
    margin: auto !important;
    display: block !important;
}

/* ── Tabs ── */
.messenger-tabs .tab-nav button {
    font-family: 'Noto Nastaliq Urdu', 'Share Tech Mono', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text-muted) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.messenger-tabs .tab-nav button.selected {
    color: #10b981 !important;
    border-bottom: 2px solid #10b981 !important;
    font-weight: 700 !important;
    background: rgba(16, 185, 129, 0.08) !important;
}
.messenger-tabs .tab-nav button::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    aspect-ratio: 1 / 1 !important;
    object-fit: contain !important;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    flex-shrink: 0;
}
.messenger-tabs .tab-nav button:nth-child(1)::before {
    background-image: url('__ICON_CHAT__');
}
.messenger-tabs .tab-nav button:nth-child(2)::before {
    background-image: url('__ICON_MEMORY__');
}
.messenger-tabs .tab-nav button:nth-child(3)::before {
    background-image: url('__ICON_MOBILE__');
}
.messenger-tabs .tab-nav button:nth-child(4)::before {
    background-image: url('__ICON_GEAR__');
}


/* ── Chatbot Bubbles (WhatsApp / Telegram Styling) ── */
.messenger-chat {
    background: #0b0f19 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    padding: 12px !important;
}

/* Assistant Message Bubble (Left / RTL) */
.messenger-chat .bot,
.messenger-chat [data-testid="bot"],
.messenger-chat .bot-message {
    background: var(--bubble-bot) !important;
    color: #f8fafc !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.15rem !important;
    line-height: 2.3 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    max-width: 86% !important;
    margin-right: auto !important;
}

/* User Message Bubble (Right) */
.messenger-chat .user,
.messenger-chat [data-testid="user"],
.messenger-chat .user-message {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.1rem !important;
    line-height: 2.1 !important;
    border: none !important;
    max-width: 82% !important;
    margin-left: auto !important;
}

/* ── Inline Voice Playback Card in Chatbot ── */
.bot-msg-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 100%;
}
.bot-msg-text {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.15rem !important;
    line-height: 2.3 !important;
    color: #f8fafc;
}
.bot-audio-replay-row {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 4px;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    direction: rtl;
}
.bot-audio-replay-btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 20px;
    padding: 5px 14px;
    color: #34d399;
    font-size: 0.82rem;
    font-family: 'Share Tech Mono', 'Noto Nastaliq Urdu', sans-serif;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
    -webkit-user-select: none;
    outline: none;
}
.bot-audio-replay-btn:hover {
    background: rgba(16, 185, 129, 0.25);
    border-color: #10b981;
    transform: translateY(-1px);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.35);
}
.bot-audio-replay-btn.playing {
    background: rgba(236, 72, 153, 0.18);
    border-color: #ec4899;
    color: #f472b6;
    animation: replay-btn-pulse 1.4s infinite ease-in-out;
}
@keyframes replay-btn-pulse {
    0%, 100% { box-shadow: 0 0 6px rgba(236, 72, 153, 0.4); }
    50% { box-shadow: 0 0 16px rgba(236, 72, 153, 0.8); }
}
.replay-btn-icon {
    width: 16px !important;
    height: 16px !important;
    aspect-ratio: 1/1 !important;
    object-fit: contain !important;
    flex-shrink: 0 !important;
}
.bot-audio-duration-pill {
    font-size: 0.72rem;
    color: #94a3b8;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.5px;
}

/* ── Compact Audio Wave Ribbon ── */
.messenger-wave-card {
    position: relative;
    width: 100%;
    height: 52px;
    background: #111827;
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
    margin: 8px 0 6px 0;
}
#neonWaveCanvas {
    display: block;
    width: 100%;
    height: 100%;
}
.waveform-state-badge {
    position: absolute;
    top: 6px; left: 10px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.75);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10b981;
    font-family: 'Share Tech Mono', monospace;
    pointer-events: none;
    letter-spacing: 0.5px;
}
.waveform-state-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 6px #10b981;
}
.messenger-status-hint {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.76rem;
    color: #94a3b8;
    font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif;
    direction: rtl;
    pointer-events: none;
}

/* ── Sticky Messenger Dock ── */
.messenger-dock {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: #111827 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 28px !important;
    padding: 6px 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    margin-top: 4px !important;
}

/* Circular WhatsApp Mic Button */
#ptt-btn {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border: 2px solid #10b981 !important;
    color: #10b981 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.3rem !important;
    cursor: pointer !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.25) !important;
    transition: all 0.15s ease !important;
    padding: 0 !important;
    margin: 0 !important;
    user-select: none !important;
    -webkit-user-select: none !important;
    touch-action: manipulation !important;
    -webkit-touch-callout: none !important;
}
#ptt-btn:hover {
    transform: scale(1.08) !important;
    border-color: #34d399 !important;
    color: #34d399 !important;
    box-shadow: 0 0 18px rgba(16, 185, 129, 0.5) !important;
}
#ptt-btn:active {
    transform: scale(0.94) !important;
}
@keyframes ptt-pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 0 14px rgba(239, 68, 68, 0.7); }
    50% { transform: scale(1.08); box-shadow: 0 0 24px rgba(239, 68, 68, 1); }
}



/* Circular Send Button */
#send-btn {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    color: #ffffff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.15rem !important;
    cursor: pointer !important;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.15s ease !important;
    padding: 0 !important;
}
#send-btn:hover {
    transform: scale(1.08) !important;
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
}

/* ── Action Strip ── */
.messenger-action-strip {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-top: 8px !important;
    gap: 8px !important;
}
.messenger-action-strip button {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--border-color) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-family: 'Noto Nastaliq Urdu', 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.messenger-action-strip button:hover {
    background: rgba(16, 185, 129, 0.1) !important;
    border-color: #10b981 !important;
    color: #10b981 !important;
}

/* ── Compact Audio Player — fully hidden; only inline replay buttons are visible ── */
#voice_reply_player,
.compact-audio-player {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}

/* ── Mobile Access Card ── */
.mobile-access-card {
    background: #111827 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

/* ── Urdu Typography Generic ── */
.urdu-text {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.15rem !important;
    line-height: 2.1 !important;
}

/* ── Message Input: direction auto so English flows LTR, Urdu flows RTL ── */
#message-input, #message-input textarea, #message-input input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #f8fafc !important;
    font-family: 'Noto Nastaliq Urdu', 'Inter', serif !important;
    font-size: 1.05rem !important;
    line-height: 1.9 !important;
    direction: auto !important;
    text-align: right !important;
    padding: 4px 10px !important;
    width: 100% !important;
    -webkit-tap-highlight-color: transparent !important;
}

/* ── Cards / Panels ── */
.block, .gr-box, .gr-panel, .gr-form {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

/* ── Responsive Chatbot Height ── */
.messenger-chat {
    min-height: 300px !important;
    max-height: 60vh !important;
    height: auto !important;
    flex: 1 !important;
    overflow-y: auto !important;
}

/* ── Touch action on all interactive controls (mobile 300ms delay fix) ── */
.bot-audio-replay-btn,
#ptt-btn,
#send-btn,
.messenger-action-strip button,
.autoplay-welcome-pill {
    -webkit-tap-highlight-color: transparent !important;
    touch-action: manipulation !important;
}

/* ── "Tabraiz is typing..." indicator bubble ── */
.adaab-typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 10px 16px;
    background: var(--bubble-bot);
    border-radius: 18px 18px 18px 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.adaab-typing-indicator .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #34d399;
    animation: adaab-bounce 1.2s infinite ease-in-out;
}
.adaab-typing-indicator .dot:nth-child(2) { animation-delay: 0.2s; }
.adaab-typing-indicator .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes adaab-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.6; }
    40% { transform: translateY(-6px); opacity: 1; }
}

/* ── Hidden Bridge Controls ── */
.hidden-control {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}

/* ══════════════════════════════════════════════════════════════════════════════
   MOBILE RESPONSIVE & TOUCH-FIRST UX ENHANCEMENTS
   ══════════════════════════════════════════════════════════════════════════════ */

/* Mobile Safe Area Inset Handling */
@supports (padding: max(0px)) {
    body, .gradio-container {
        padding-left: max(0px, env(safe-area-inset-left)) !important;
        padding-right: max(0px, env(safe-area-inset-right)) !important;
    }
}

/* Tablet & Mobile Screens (<= 768px) */
@media screen and (max-width: 768px) {
    .messenger-wrapper {
        padding: 4px 8px 16px 8px !important;
        max-width: 100% !important;
    }

    /* Compact Header Card */
    .messenger-header {
        padding: 8px 12px !important;
        border-radius: 14px !important;
        margin-bottom: 8px !important;
    }
    .messenger-avatar {
        width: 42px !important;
        height: 42px !important;
    }
    .messenger-avatar-img {
        width: 38px !important;
        height: 38px !important;
    }
    .messenger-name {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    .messenger-status {
        font-size: 0.7rem !important;
    }

    /* Horizontally Scrollable Modern Tab Bar */
    .messenger-tabs .tab-nav {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 6px !important;
        padding: 4px 2px 8px 2px !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    .messenger-tabs .tab-nav::-webkit-scrollbar {
        display: none !important;
    }
    .messenger-tabs .tab-nav button {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
        font-size: 0.85rem !important;
        padding: 6px 12px !important;
        border-radius: 20px !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid var(--border-color) !important;
    }
    .messenger-tabs .tab-nav button.selected {
        background: rgba(16, 185, 129, 0.15) !important;
        border-color: #10b981 !important;
        color: #10b981 !important;
    }

    /* Mobile Chatbot Area */
    .messenger-chat {
        min-height: 280px !important;
        max-height: 52vh !important;
        padding: 8px !important;
        border-radius: 14px !important;
        -webkit-overflow-scrolling: touch !important;
        overscroll-behavior: contain !important;
    }
    .messenger-chat .bot,
    .messenger-chat [data-testid="bot"],
    .messenger-chat .bot-message {
        max-width: 92% !important;
        font-size: 1.02rem !important;
        line-height: 1.95 !important;
        padding: 10px 14px !important;
        border-radius: 16px 16px 16px 4px !important;
    }
    .messenger-chat .user,
    .messenger-chat [data-testid="user"],
    .messenger-chat .user-message {
        max-width: 88% !important;
        font-size: 1.0rem !important;
        line-height: 1.85 !important;
        padding: 8px 14px !important;
        border-radius: 16px 16px 4px 16px !important;
    }

    /* Mobile Replay Row */
    .bot-audio-replay-row {
        gap: 6px !important;
        flex-wrap: wrap !important;
        justify-content: space-between !important;
    }
    .bot-audio-replay-btn {
        padding: 4px 10px !important;
        font-size: 0.76rem !important;
        min-height: 32px !important;
    }
    .bot-audio-duration-pill {
        font-size: 0.68rem !important;
    }

    /* Compact Mobile Waveform Ribbon */
    .messenger-wave-card {
        height: 42px !important;
        margin: 6px 0 !important;
        border-radius: 8px !important;
    }
    .waveform-state-badge {
        top: 4px !important;
        left: 6px !important;
        font-size: 0.65rem !important;
        padding: 1px 6px !important;
    }
    .messenger-status-hint {
        font-size: 0.68rem !important;
        right: 8px !important;
    }

    /* Mobile Sticky Bottom Input Dock (Ergonomic Thumb-Zone) */
    .messenger-dock {
        position: sticky !important;
        bottom: max(6px, env(safe-area-inset-bottom, 8px)) !important;
        z-index: 99 !important;
        background: rgba(17, 24, 39, 0.96) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        border-radius: 26px !important;
        padding: 5px 8px !important;
        gap: 8px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.7) !important;
    }

    /* Large Ergonomic Touch Mic Button */
    #ptt-btn {
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        min-height: 48px !important;
    }
    .ptt-icon-img {
        width: 26px !important;
        height: 26px !important;
    }

    /* Critical iOS Fix: Text input must be >= 16px to prevent auto-zoom */
    #message-input, #message-input textarea, #message-input input {
        font-size: 16px !important;
        line-height: 1.5 !important;
        padding: 6px 8px !important;
    }

    /* Send Button */
    #send-btn {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        min-height: 44px !important;
    }
    #send-btn img {
        width: 20px !important;
        height: 20px !important;
    }

    /* Action Strip Mobile Wrap */
    .messenger-action-strip {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        margin-top: 6px !important;
    }
    .messenger-action-strip button {
        flex: 1 1 calc(50% - 6px) !important;
        font-size: 0.78rem !important;
        padding: 6px 8px !important;
        min-height: 36px !important;
    }
    .messenger-action-strip button:first-child {
        flex: 1 1 100% !important; /* Welcome button takes full width on top */
    }

    /* Welcome Pill */
    .autoplay-welcome-pill {
        font-size: 0.86rem !important;
        padding: 7px 14px !important;
        margin-bottom: 8px !important;
    }
}

/* Small Phones (< 420px) Extra Tuning */
@media screen and (max-width: 420px) {
    .messenger-wrapper {
        padding: 2px 4px 12px 4px !important;
    }
    .messenger-status-hint {
        display: none !important; /* Avoid overlap with state badge on tiny screens */
    }
    .messenger-chat .bot,
    .messenger-chat [data-testid="bot"],
    .messenger-chat .bot-message {
        max-width: 96% !important;
        font-size: 0.98rem !important;
    }
    .messenger-chat .user,
    .messenger-chat [data-testid="user"],
    .messenger-chat .user-message {
        max-width: 92% !important;
        font-size: 0.96rem !important;
    }
    #ptt-btn {
        width: 46px !important;
        height: 46px !important;
        min-width: 46px !important;
    }
    #send-btn {
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
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
const numEqBars = 36;
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

function drawEqualizerVisualizer(w, h) {
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
}

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

        const replayBtns = chatContainer.querySelectorAll('.bot-audio-replay-btn');
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

    # Append user message first; typing indicator appears immediately
    history.append({"role": "user", "content": message})

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

    history.append({"role": "user", "content": user_transcript})

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
        with gr.Column(elem_classes=["messenger-wrapper"]):
            # Top Header Card (Contact Bar)
            header_html = gr.HTML(
                f"""
                <div class="messenger-header">
                    <div class="messenger-contact-info">
                        <div class="messenger-avatar">
                            <img src="{ICON_AVATAR}" class="messenger-avatar-img" alt="Tabraiz" />
                            <span class="online-dot"></span>
                        </div>
                        <div>
                            <div class="messenger-name" id="header-name">تبریز (Tabraiz)</div>
                            <div class="messenger-status"><span class="status-pulse-dot"></span>آن لائن • باادب اردو صوتی معاون</div>
                        </div>
                    </div>
                </div>
                """
            )

            with gr.Tabs(elem_classes=["messenger-tabs"]):
                # ── TAB 1: Chat Messenger ──
                with gr.Tab("گفتگو (Chat)", elem_id="tab-chat"):
                    chat_style = gr.State(value="standard")
                    chat_voice = gr.State(value="male")

                    # Single Unified Messenger Chatbot
                    chatbot = gr.Chatbot(
                        value=[{"role": "assistant", "content": format_assistant_message(TABRAIZ_INITIAL_GREETING, TABRAIZ_OPENING_AUDIO)}],
                        elem_classes=["messenger-chat"],
                        label="چیٹ (Chat)",
                        show_label=False,
                        sanitize_html=False
                    )

                    # Compact Audio Waveform Ribbon
                    gr.HTML(
                        f"""
                        <div class="messenger-wave-card">
                            <div class="waveform-state-badge">
                                <span class="waveform-state-dot"></span>
                                <span id="waveform-state-text">IDLE</span>
                            </div>
                            <canvas id="neonWaveCanvas"></canvas>
                            <div id="ptt-status-hint" class="messenger-status-hint">
                                <img src="{ICON_MIC}" class="hint-icon-img" alt="" /> بولنے کے لیے مائیک دبائیں • Tap mic to speak
                            </div>
                        </div>
                        """
                    )

                    # Hidden Gradio Audio bridge (CSS-hidden, still needed for Gradio state transport)
                    voice_reply_player = gr.Audio(
                        value=None,
                        autoplay=False,
                        elem_id="voice_reply_player",
                        elem_classes=["compact-audio-player"],
                        visible=False
                    )

                    # Per-session profile state (A8: isolates users from each other)
                    profile_state = gr.State(value={})

                    # Sticky Messenger Bottom Dock
                    with gr.Row(elem_classes=["messenger-dock"]):
                        ptt_btn_html = gr.HTML(
                            f"""
                            <button id="ptt-btn" 
                                    type="button"
                                    title="بولنے کے لیے ٹیپ کریں (Tap to speak / Tap again to send)"
                                    onclick="handleMicToggle(event)">
                                <img id="ptt-btn-icon" src="{ICON_MIC}" class="ptt-icon-img" alt="Mic" />
                            </button>
                            """
                        )
                        text_msg = gr.Textbox(
                            placeholder="پیغام تحریر فرمائیں... (Type a message...)", 
                            elem_id="message-input",
                            show_label=False,
                            container=False,
                            scale=8,
                            lines=1,
                            max_lines=3
                        )
                        send_btn = gr.Button("", icon=get_icon_path("send.png"), elem_id="send-btn", scale=1, variant="primary")

                    # Hidden bridge controls for PTT JavaScript trigger
                    ptt_raw_input = gr.Textbox(elem_id="ptt_raw_input", elem_classes=["hidden-control"])
                    ptt_trigger_btn = gr.Button("TRIGGER_PTT", elem_id="ptt_trigger_btn", elem_classes=["hidden-control"])
                    ptt_status_box = gr.Textbox(elem_id="ptt_status_box", elem_classes=["hidden-control"])

                    # Action strip: Listen welcome audio + Clear chat + Clear memory
                    with gr.Row(elem_classes=["messenger-action-strip"]):
                        welcome_listen_btn = gr.Button("تعارفی پیغام سنیں (Hear Welcome)", icon=get_icon_path("speaker.png"), size="sm")
                        clear_btn = gr.Button("نئی گفتگو (New Chat)", icon=get_icon_path("refresh.png"), size="sm")
                        clear_mem_action_btn = gr.Button("یادداشت صاف کریں (Clear Memory)", icon=get_icon_path("trash.png"), size="sm")

                # Connect PTT trigger (A8: include profile_state in/out)
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

                # Connect Text send button and enter submit (A8: profile_state in/out)
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

                # B3: Welcome button is JS-only (no server round-trip needed)
                welcome_listen_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    js="() => { greetingAutoplayDone = false; greetingAttemptInProgress = false; startWelcomeAudioNow(); return []; }"
                )

                # Clear chat: reset to greeting, reset profile state
                clear_btn.click(
                    fn=lambda: ([{"role": "assistant", "content": format_assistant_message(TABRAIZ_INITIAL_GREETING, TABRAIZ_OPENING_AUDIO)}], "", {}),
                    inputs=None,
                    outputs=[chatbot, text_msg, profile_state]
                )

            # ── TAB 2: Memory ──
            with gr.Tab("یادداشت (Memory)", elem_id="tab-memory"):
                gr.Markdown("### <center style='color:#10b981; font-family:Noto Nastaliq Urdu,serif;'>محفوظ احباب اور کوائف (Known Profiles & Memory)</center>")
                with gr.Row():
                    refresh_memory_btn = gr.Button("تازہ کریں (Refresh)", icon=get_icon_path("refresh.png"), variant="secondary")
                    clear_memory_btn = gr.Button("تمام یادداشت صاف کریں (Clear Memory)", icon=get_icon_path("trash.png"), variant="stop")
                memory_markdown_box = gr.Markdown(value=format_who_did_you_talk_to_response("Tabraiz"), elem_classes="urdu-text")
                refresh_memory_btn.click(lambda: format_who_did_you_talk_to_response("Tabraiz"), None, memory_markdown_box)

                def handle_clear_memory():
                    clear_user_memory()
                    SESSION_STATE["active_profile"] = {}
                    return format_who_did_you_talk_to_response("Tabraiz")

                clear_memory_btn.click(handle_clear_memory, None, memory_markdown_box)

                def handle_chat_and_memory_wipe():
                    clear_user_memory()
                    SESSION_STATE["active_profile"] = {}
                    wiped_msg = "تمام محفوظ شدہ یادداشت اور سابقہ کوائف صاف کر دیے گئے ہیں۔ (All memory wiped successfully)."
                    initial_msg = format_assistant_message(f"{wiped_msg}\n\n{TABRAIZ_INITIAL_GREETING}", TABRAIZ_OPENING_AUDIO)
                    return (
                        [{"role": "assistant", "content": initial_msg}],
                        "",
                        {},
                        format_who_did_you_talk_to_response("Tabraiz")
                    )

                clear_mem_action_btn.click(
                    fn=handle_chat_and_memory_wipe,
                    inputs=None,
                    outputs=[chatbot, text_msg, profile_state, memory_markdown_box]
                )

            # ── TAB 3: Mobile Access & QR ──
            with gr.Tab("موبائل (Mobile QR)", elem_id="tab-mobile"):
                local_ip = get_local_lan_ip()
                local_mobile_url = f"https://{local_ip}:{active_port}"
                local_qr_b64 = generate_qr_code_base64(local_mobile_url)

                if public_url:
                    public_qr_b64 = generate_qr_code_base64(public_url)
                    tab3_html = f"""
                    <div style="display:flex; flex-direction:column; gap:16px; margin-top:14px;">
                        <!-- Cloudflare Public Tunnel Card -->
                        <div class="mobile-access-card" style="border:1px solid rgba(16,185,129,0.35); background:linear-gradient(135deg, rgba(16,185,129,0.06) 0%, rgba(15,23,42,0.85) 100%);">
                            <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:20px;">
                                <div style="flex:1; min-width:240px;">
                                    <div style="display:flex; align-items:center; gap:8px;">
                                        <span class="status-pulse-dot" style="background:#10b981;"></span>
                                        <div style="font-family:'Share Tech Mono',monospace; color:#10b981; font-size:0.95rem; font-weight:700; letter-spacing:1px;">
                                            CLOUDFLARE SECURE TUNNEL (GLOBAL ACCESS)
                                        </div>
                                    </div>
                                    <code style="display:block; margin-top:10px; background:rgba(0,0,0,0.4); padding:8px 12px; border:1px solid rgba(16,185,129,0.3); color:#34d399; font-family:monospace; font-size:0.92rem; border-radius:6px; word-break:break-all;">{public_url}</code>
                                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px; line-height:1.5;">
                                        موبائل ڈیٹا (4G/5G) یا کسی بھی نیٹ ورک پر مکمل کام کرتا ہے۔ باقاعدہ تصدیق شدہ SSL سرٹیفکیٹ اور مائیکروفون فعال۔
                                    </div>
                                    <div style="font-size:0.75rem; color:#64748b; margin-top:4px; font-family:'Share Tech Mono',monospace;">
                                        Cloudflare Verified SSL • Zero Browser Warnings • Full Audio & Mic
                                    </div>
                                </div>
                                <div style="text-align:center; flex-shrink:0;">
                                    <div style="background:#ffffff; padding:10px; border-radius:10px; display:inline-block; box-shadow:0 4px 14px rgba(0,0,0,0.35);">
                                        <img src="{public_qr_b64}" alt="Cloudflare QR" style="width:130px; height:130px; display:block; aspect-ratio:1/1; object-fit:contain;" />
                                    </div>
                                    <div style="font-size:0.7rem; color:#10b981; font-weight:bold; margin-top:6px; font-family:'Share Tech Mono',monospace;">SCAN FOR GLOBAL ACCESS</div>
                                </div>
                            </div>
                        </div>

                        <!-- Local Wi-Fi Card -->
                        <div class="mobile-access-card" style="border:1px solid rgba(56,189,248,0.25); background:linear-gradient(135deg, rgba(56,189,248,0.04) 0%, rgba(15,23,42,0.85) 100%);">
                            <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:20px;">
                                <div style="flex:1; min-width:240px;">
                                    <div style="display:flex; align-items:center; gap:8px;">
                                        <span class="status-pulse-dot" style="background:#38bdf8;"></span>
                                        <div style="font-family:'Share Tech Mono',monospace; color:#38bdf8; font-size:0.92rem; font-weight:700; letter-spacing:1px;">
                                            LOCAL WI-FI ACCESS (SAME NETWORK)
                                        </div>
                                    </div>
                                    <code style="display:block; margin-top:10px; background:rgba(0,0,0,0.4); padding:8px 12px; border:1px solid rgba(56,189,248,0.3); color:#38bdf8; font-family:monospace; font-size:0.92rem; border-radius:6px; word-break:break-all;">{local_mobile_url}</code>
                                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px; line-height:1.5;">
                                        اگر آپ کا فون اسی وائی فائی (Wi-Fi) راؤٹر سے منسلک ہے تو براہِ راست رابطہ کے لیے بہترین ہے۔
                                    </div>
                                    <div style="font-size:0.75rem; color:#64748b; margin-top:4px; font-family:'Share Tech Mono',monospace;">
                                        Safari: Show Details → Visit Website | Chrome: Advanced → Proceed
                                    </div>
                                </div>
                                <div style="text-align:center; flex-shrink:0;">
                                    <div style="background:#ffffff; padding:10px; border-radius:10px; display:inline-block; box-shadow:0 4px 14px rgba(0,0,0,0.35);">
                                        <img src="{local_qr_b64}" alt="Local Wi-Fi QR" style="width:130px; height:130px; display:block; aspect-ratio:1/1; object-fit:contain;" />
                                    </div>
                                    <div style="font-size:0.7rem; color:#38bdf8; font-weight:bold; margin-top:6px; font-family:'Share Tech Mono',monospace;">SCAN FOR LOCAL WI-FI</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                else:
                    tab3_html = f"""
                    <div style="display:flex; flex-direction:column; gap:16px; margin-top:14px;">
                        <!-- Local Wi-Fi Access Card -->
                        <div class="mobile-access-card" style="border:1px solid rgba(16,185,129,0.35); background:linear-gradient(135deg, rgba(16,185,129,0.06) 0%, rgba(15,23,42,0.85) 100%);">
                            <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:20px;">
                                <div style="flex:1; min-width:240px;">
                                    <div style="display:flex; align-items:center; gap:8px;">
                                        <span class="status-pulse-dot" style="background:#10b981;"></span>
                                        <div style="font-family:'Share Tech Mono',monospace; color:#10b981; font-size:0.95rem; font-weight:700; letter-spacing:1px;">
                                            MOBILE LOCAL WI-FI ACCESS
                                        </div>
                                    </div>
                                    <code style="display:block; margin-top:10px; background:rgba(0,0,0,0.4); padding:8px 12px; border:1px solid rgba(16,185,129,0.3); color:#34d399; font-family:monospace; font-size:0.92rem; border-radius:6px; word-break:break-all;">{local_mobile_url}</code>
                                    <div style="font-size:0.8rem; color:#94a3b8; margin-top:8px; line-height:1.5;">
                                        موبائل فون کو اسی وائی فائی سے جوڑ کر کیو آر کوڈ اسکین فرمائیں۔
                                    </div>
                                    <div style="font-size:0.75rem; color:#64748b; margin-top:4px; font-family:'Share Tech Mono',monospace;">
                                        Safari: Show Details → Visit Website | Chrome: Advanced → Proceed
                                    </div>
                                </div>
                                <div style="text-align:center; flex-shrink:0;">
                                    <div style="background:#ffffff; padding:10px; border-radius:10px; display:inline-block; box-shadow:0 4px 14px rgba(0,0,0,0.35);">
                                        <img src="{local_qr_b64}" alt="Mobile QR" style="width:130px; height:130px; display:block; aspect-ratio:1/1; object-fit:contain;" />
                                    </div>
                                    <div style="font-size:0.7rem; color:#10b981; font-weight:bold; margin-top:6px; font-family:'Share Tech Mono',monospace;">SCAN WITH PHONE</div>
                                </div>
                            </div>
                        </div>

                        <!-- Cloudflare Tunnel Tip Card -->
                        <div style="background:rgba(30,41,59,0.5); border:1px dashed rgba(148,163,184,0.3); border-radius:10px; padding:14px 18px; color:#cbd5e1; font-size:0.85rem;">
                            <div style="font-family:'Share Tech Mono',monospace; color:#38bdf8; font-weight:bold; margin-bottom:4px;">
                                PUBLIC INTERNET ACCESS VIA CLOUDFLARE
                            </div>
                            <div style="line-height:1.6;">
                                اگر آپ موبائل فون سے 4G/5G یا کسی دوسرے انٹرنیٹ کنکشن کے ذریعے بغیر کسی انتباہ (Zero Warnings) کے رابطہ کرنا چاہتے ہیں تو سرور کو اس طرح چلائیں:
                                <br/>
                                <code style="display:inline-block; margin-top:6px; background:#0f172a; padding:4px 10px; border-radius:4px; color:#10b981; font-family:monospace;">python start.py</code> (Option 4 منتخب فرمائیں) یا <code style="display:inline-block; background:#0f172a; padding:4px 10px; border-radius:4px; color:#10b981; font-family:monospace;">python app.py --tunnel</code>
                            </div>
                        </div>
                    </div>
                    """

                gr.HTML(tab3_html)

            # ── TAB 4: Diagnostics ──
            with gr.Tab("نظام (Diagnostics)", elem_id="tab-diag"):
                diag_refresh = gr.Button("REFRESH DIAGNOSTICS", icon=get_icon_path("refresh.png"))
                gpu_telemetry_box = gr.JSON(value=get_gpu_telemetry(), label="GPU Telemetry")
                log_box = gr.Textbox(value=get_latest_crash_report(), label="System / Crash Log", lines=10)
                diag_refresh.click(lambda: (get_gpu_telemetry(), get_latest_crash_report()), None, [gpu_telemetry_box, log_box])

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

