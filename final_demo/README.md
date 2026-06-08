# 中文新聞分類

本專題使用本機下載的 THUCNews 資料夾，讓 MacBERT 看中文新聞內容，判斷新聞屬於體育、財經、科技、娛樂等類別。

THUCNews 官方來源：[THUCTC: An Efficient Chinese Text Classifier](http://thuctc.thunlp.org)

新聞類別共有 14 類：

| 類別 | Label |
| --- | ---: |
| `财经` | 0 |
| `彩票` | 1 |
| `房产` | 2 |
| `股票` | 3 |
| `家居` | 4 |
| `教育` | 5 |
| `科技` | 6 |
| `社会` | 7 |
| `时尚` | 8 |
| `时政` | 9 |
| `体育` | 10 |
| `星座` | 11 |
| `游戏` | 12 |
| `娱乐` | 13 |

## 環境

```bash
uv sync
```

正式訓練只支援 NVIDIA CUDA。若目前環境沒有 CUDA，可以使用 `--skip-train` 只跑 EDA。

## 開發驗證

先用 5000 筆資料驗證資料清理、EDA 與訓練流程：

```bash
uv run python src/main.py --dev-sample 5000
```

只跑 EDA，不訓練模型：

```bash
uv run python src/main.py --dev-sample 5000 --skip-train
```

若要重新產生正式全量 EDA 圖表，但不重新訓練模型：

```bash
uv run python src/main.py --skip-train --output-dir outputs_eda_news
```

本機完整 THUCNews 應放成每個分類一個資料夾：

```text
THUCNews/
├── 体育/
├── 财经/
├── 科技/
└── ...
```

讀取本機資料夾：

```bash
uv run python src/main.py --data-path THUCNews --dev-sample 5000 --skip-train
```

`--data-path` 只支援原始 THUCNews 分類資料夾。

## 正式全量訓練

正式執行預設會載入全量資料，清理後使用所有有效資料訓練 5 epochs：

```bash
uv run python src/main.py
```

建議正式訓練指令：

```bash
uv run python src/main.py --data-path THUCNews --skip-eda --epochs 3 --batch-size 32 --max-length 256 --dataloader-num-workers 4 --output-dir outputs_train_news --logging-dir outputs_train_news/tensorboard --model-dir models/macbert_news_classifier
```

若 12GB VRAM 不足，將 batch size 降為 16：

```bash
uv run python src/main.py --data-path THUCNews --skip-eda --epochs 3 --batch-size 16 --max-length 256 --dataloader-num-workers 4 --output-dir outputs_train_news --logging-dir outputs_train_news/tensorboard --model-dir models/macbert_news_classifier
```

若訓練中斷，可從最新 checkpoint 續訓：

```bash
uv run python src/main.py --resume-from-checkpoint auto --output-dir outputs_train_news --logging-dir outputs_train_news/tensorboard --model-dir models/macbert_news_classifier
```

查看 TensorBoard：

```bash
uv run tensorboard --logdir outputs_train_news/tensorboard --host 0.0.0.0 --port 6006
```

## 網頁推論

模型訓練完成後，啟動網頁介面：

```bash
uv run python src/app.py --host 0.0.0.0 --port 7860 --model-dir models/macbert_news_classifier
```

在瀏覽器開啟：

```text
http://localhost:7860
```

網頁可輸入簡體或繁體中文。推論前會使用 OpenCC 將輸入轉成簡體，讓文字形式與 THUCNews 訓練資料一致；預測結果會以繁體中文類別顯示。

## 主要輸出

```text
outputs/
├── eda_summary.json
├── label_counts.csv
├── metrics.json
├── predictions.csv
├── training_summary.json
├── tensorboard/
└── figures/
    ├── confusion_matrix.png
    ├── label_distribution.png
    ├── text_length_by_label.png
    └── text_length_distribution.png
```

模型會儲存在：

```text
models/macbert_news_classifier/
```

訓練完成後，Trainer 狀態會儲存在：

```text
models/macbert_news_classifier/trainer_state.json
```

`outputs/training_summary.json` 會記錄資料筆數、label 分布、訓練參數、CUDA 資訊、完成 epoch 與主要 metrics。

## 程式結構

```text
src/
├── main.py             # 主流程入口
├── app.py              # Gradio 網頁推論
├── dataset_loader.py   # 載入本機 THUCNews 資料
├── preprocessing.py    # 解析類別、清理新聞正文、建立 label
├── news_dataset.py     # MacBERT lazy tokenization dataset
├── training.py         # MacBERT fine-tuning
├── eda.py              # EDA 圖表與摘要
└── evaluation.py       # metrics 與 confusion matrix
```
