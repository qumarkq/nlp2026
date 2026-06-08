"""Gradio web UI for trained THUCNews classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from opencc import OpenCC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import LABEL_ORDER, clean_text

DISPLAY_LABELS = {
    "财经": "財經",
    "彩票": "彩票",
    "房产": "房產",
    "股票": "股票",
    "家居": "家居",
    "教育": "教育",
    "科技": "科技",
    "社会": "社會",
    "时尚": "時尚",
    "时政": "時政",
    "体育": "體育",
    "星座": "星座",
    "游戏": "遊戲",
    "娱乐": "娛樂",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="THUCNews classification web app")
    parser.add_argument("--model-dir", type=Path, default=Path("models/macbert_news_classifier"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


class NewsCategoryPredictor:
    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. Train the model first."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.converter = OpenCC("t2s")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, news_text: str) -> tuple[str, dict[str, float]]:
        news_text = clean_text(news_text)
        if not news_text:
            return "請輸入新聞內容", {
                DISPLAY_LABELS[label]: 0.0 for label in LABEL_ORDER
            }

        model_text = self.converter.convert(news_text)
        inputs = self.tokenizer(
            model_text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        scores = {
            DISPLAY_LABELS[label]: round(float(probability), 4)
            for label, probability in zip(LABEL_ORDER, probabilities, strict=True)
        }
        best_label = DISPLAY_LABELS[LABEL_ORDER[int(torch.argmax(logits).item())]]
        return best_label, scores


def build_app(model_dir: Path) -> gr.Blocks:
    predictor = NewsCategoryPredictor(model_dir)

    with gr.Blocks(title="中文新聞分類") as demo:
        gr.Markdown("# 中文新聞分類")
        gr.Markdown("輸入中文新聞內容，模型會預測新聞屬於哪一個類別。")
        news_text = gr.Textbox(label="新聞內容", lines=10)
        predict_button = gr.Button("預測")
        predicted_label = gr.Textbox(label="預測新聞類別", interactive=False)
        probabilities = gr.Label(label="各類別機率")

        predict_button.click(
            predictor.predict,
            inputs=[news_text],
            outputs=[predicted_label, probabilities],
        )

    return demo


def main() -> None:
    args = parse_args()
    app = build_app(args.model_dir)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
