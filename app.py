import os
import sys
import json
import gradio as gr
from PIL import Image

# Ensure UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

from agent import ADAAB_SYSTEM_PROMPT, compose_unique_couplet, CLASSICAL_THEMES
from backend import AdaabClient
from poster_maker import get_random_aesthetic, render_urdu_poetry_poster
from voice_reader import generate_poetry_recitation
from video_maker import assemble_poetry_reel
from poetry_db import record_post, get_history_summary
from crash_tracker import get_gpu_telemetry, get_latest_crash_report

client = AdaabClient()

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

.urdu-text, .gradio-container textarea, .gradio-container input {
    font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Urdu Typesetting', serif !important;
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.15rem !important;
    line-height: 2.0 !important;
}

.urdu-title {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: center !important;
    color: #D4AF37 !important;
}
"""

def chat_fn(message, history):
    if not message.strip():
        return history, ""
    
    messages = [{"role": "system", "content": ADAAB_SYSTEM_PROMPT}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat_completion(messages, temperature=0.7, max_tokens=800)
        bot_reply = resp["content"]
    except Exception as e:
        bot_reply = f"معذرت، رابطہ نہیں ہو سکا: {e}"

    history.append((message, bot_reply))
    return history, ""

def generate_studio_post(theme_selection, custom_m1, custom_m2):
    """Generates a complete post package on demand."""
    if custom_m1.strip() and custom_m2.strip():
        couplet_data = {
            "misra_1": custom_m1.strip(),
            "misra_2": custom_m2.strip(),
            "theme": "Custom",
            "roman_urdu": "",
            "english_translation": ""
        }
    else:
        couplet_data = compose_unique_couplet(theme=theme_selection, backend_client=client)

    aesthetic = get_random_aesthetic()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "studio_temp")
    os.makedirs(out_dir, exist_ok=True)

    poster_path = os.path.join(out_dir, "temp_poster.png")
    audio_path = os.path.join(out_dir, "temp_audio.mp3")
    reel_path = os.path.join(out_dir, "temp_reel.mp4")

    render_urdu_poetry_poster(couplet_data["misra_1"], couplet_data["misra_2"], aesthetic, output_path=poster_path)
    generate_poetry_recitation(couplet_data["misra_1"], couplet_data["misra_2"], output_path=audio_path)
    assemble_poetry_reel(poster_path, audio_path, reel_path, intro_delay_sec=0.5, outro_hold_sec=0.5)

    meta_text = (
        f"مصرع اول: {couplet_data['misra_1']}\n"
        f"مصرع دوم: {couplet_data['misra_2']}\n\n"
        f"Aesthetic: {aesthetic['palette']['name']} ({aesthetic['lighting']})\n"
        f"Style: {aesthetic['art_style']}\n"
        f"Translation: {couplet_data.get('english_translation', 'N/A')}"
    )

    return poster_path, audio_path, reel_path, meta_text

def build_app():
    with gr.Blocks(title="آداب - Adaab Studio", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # <div class="urdu-title">آداب (Adaab) - اردو شاعری اور ملٹی میڈیا اسٹوڈیو</div>
            ### <center>Serverless Qwen 2.5 (7B) on Modal.com | Scale-to-Zero ($0.00 Idle)</center>
            """
        )

        with gr.Tab("گفتگو و مشاعرہ (Urdu Chat)"):
            chatbot = gr.Chatbot(height=450, elem_classes="urdu-text")
            msg = gr.Textbox(placeholder="یہاں اردو میں پیغام یا شاعری کی فرمائش لکھیں...", elem_classes="urdu-text")
            with gr.Row():
                send_btn = gr.Button("بھیجیں (Send)", variant="primary")
                clear_btn = gr.Button("صاف کریں (Clear)")

            send_btn.click(chat_fn, [msg, chatbot], [chatbot, msg])
            msg.submit(chat_fn, [msg, chatbot], [chatbot, msg])
            clear_btn.click(lambda: [], None, chatbot)

        with gr.Tab("پوسٹر و ویڈیو اسٹوڈیو (Post & Reel Studio)"):
            with gr.Row():
                with gr.Column(scale=1):
                    theme_dropdown = gr.Dropdown(CLASSICAL_THEMES, label="موضوع منتخب کریں (Select Theme)", value=CLASSICAL_THEMES[0])
                    gr.Markdown("---")
                    gr.Markdown("**یا اپنی ذاتی شاعری لکھیں (Optional Custom Verse):**")
                    cust_m1 = gr.Textbox(label="مصرع اول (Line 1)", placeholder="پہلا مصرع...", elem_classes="urdu-text")
                    cust_m2 = gr.Textbox(label="مصرع دوم (Line 2)", placeholder="دوسرا مصرع...", elem_classes="urdu-text")
                    gen_btn = gr.Button("تخلیق کریں (Generate Complete Post)", variant="primary")

                with gr.Column(scale=1):
                    out_poster = gr.Image(label="پوسٹر (9:16 Portrait Poster)", type="filepath")
                    out_audio = gr.Audio(label="آوازِ نسواں (Female Urdu Recitation)", type="filepath")
                    out_video = gr.Video(label="ویڈیو ریل (1080x1920 MP4 Reel)")
                    out_meta = gr.Textbox(label="تفصیلات و ترجمہ (Metadata & Translation)", lines=4)

            gen_btn.click(
                generate_studio_post,
                [theme_dropdown, cust_m1, cust_m2],
                [out_poster, out_audio, out_video, out_meta]
            )

        with gr.Tab("ڈیٹا بیس و تاریخچہ (History & Stats)"):
            refresh_btn = gr.Button("ریفریش کریں (Refresh Stats)")
            stats_json = gr.JSON(value=get_history_summary())
            refresh_btn.click(get_history_summary, None, stats_json)

        with gr.Tab("سسٹم مانیٹر و لاگز (Cloud Diagnostics)"):
            gpu_telemetry_box = gr.JSON(value=get_gpu_telemetry(), label="GPU Telemetry")
            log_box = gr.Textbox(value=get_latest_crash_report(), label="Recent Crash & Error Report", lines=12)
            diag_refresh = gr.Button("Refresh Telemetry")
            diag_refresh.click(lambda: (get_gpu_telemetry(), get_latest_crash_report()), None, [gpu_telemetry_box, log_box])

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7865, inbrowser=True)
