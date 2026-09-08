import os
import json

DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poetry_training_dataset.jsonl")

POETRY_SAMPLES = [
    {
        "instruction": "مرزا غالب کے انداز میں زندگی اور تمنا پر ایک باوزن شعر کہیں۔",
        "output": "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے\nبہت نکلے مرے ارمان لیکن پھر بھی کم نکلے"
    },
    {
        "instruction": "علامہ اقبال کے فکر کے مطابق خودی اور امید کا درس دینے والا شعر لکھیں۔",
        "output": "ستاروں سے آگے جہاں اور بھی ہیں\nابھی عشق کے امتحان اور بھی ہیں"
    },
    {
        "instruction": "فیض احمد فیض کے رنگ میں موسمِ بہار اور انتظار پر دو مصرعے لکھیں۔",
        "output": "گلوں میں رنگ بھرے باد نوبہار چلے\nچلے بھی آؤ کہ گلشن کا کاروبار چلے"
    },
    {
        "instruction": "جون ایلیا کے طرز میں اداسی اور تنہائی پر شعر تخلیق کریں۔",
        "output": "بے دلی کیا یوں ہی دن گزر جائیں گے\nصرف زندہ رہے ہم تو مر جائیں گے"
    },
    {
        "instruction": "بارش اور گزری یادوں پر ایک رومانی شعر تحریر کریں۔",
        "output": "برس رہی ہے شبِ غم میں بارشِ باراں\nتمہاری یاد کے آنسو ہیں اور دلِ ویراں"
    },
    {
        "instruction": "صبح، نئی شروعات اور روشنی پر شعر کہیں۔",
        "output": "سحر ہوئی ہے نیا اک دیا جلانے کو\nاندھیری رات کے ہر داغ کو مٹانے کو"
    }
]

def generate_poetry_jsonl():
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for item in POETRY_SAMPLES:
            chatml_sample = {
                "messages": [
                    {"role": "system", "content": "آپ آداب ہیں، ایک عظیم اور باکمال اردو شاعر جو علمِ عروض کے مطابق شاعری کرتے ہیں۔"},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item["output"]}
                ]
            }
            f.write(json.dumps(chatml_sample, ensure_ascii=False) + "\n")
    print(f"✓ Generated poetry training dataset with {len(POETRY_SAMPLES)} samples at: {DATASET_FILE}")

if __name__ == "__main__":
    generate_poetry_jsonl()
