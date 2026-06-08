# 中文新聞分類規格書

## 1. 專題名稱

中文新聞分類

## 2. 專題目標

本專題使用本機下載的 THUCNews 中文新聞資料夾，訓練 `hfl/chinese-macbert-base` 判斷一段中文新聞屬於哪一個新聞類別。

專題需要完成以下目標：

1. 載入並整理 THUCNews 中文新聞資料。
2. 從原始文字中解析新聞類別與新聞正文。
3. 分析各新聞類別數量與新聞文字長度分布。
4. 使用 `hfl/chinese-macbert-base` 建立中文新聞分類模型。
5. 輸出 accuracy、macro F1、weighted F1 與 confusion matrix。
6. 建立 Gradio 網頁，讓使用者輸入新聞內容並取得預測類別。
7. 網頁推論需接受簡體與繁體中文輸入，並以繁體中文顯示預測結果。

## 3. 明確假設

依照 `AGENTS.md`，本專題先列出實作假設，避免後續開發時隱含決策。

1. 主要資料來源只使用本機原始 THUCNews 分類資料夾。
2. 模型只使用 `hfl/chinese-macbert-base` 作為主要模型。
3. 正式流程必須全量載入資料，清理後使用所有有效資料。
4. 任務是多類別新聞主題分類。
5. Python 環境必須使用 `uv` 管理。
6. 訓練硬體預期為 NVIDIA RTX 5070 12GB。
7. 正式訓練只使用 CUDA；若沒有 CUDA，程式不得進行訓練。

## 4. 使用資料集

- Dataset: 本機 THUCNews 資料夾
- 官方來源：[THUCTC: An Efficient Chinese Text Classifier](http://thuctc.thunlp.org)
- 語言：中文
- 資料量：依本機下載內容為準
- 主要格式：每個分類一個資料夾，資料夾內為 `.txt` 新聞檔

本機資料夾格式為：

```text
THUCNews/
├── 体育/
├── 财经/
├── 科技/
└── ...
```

例如 `THUCNews/体育/112642.txt` 代表該新聞類別為 `体育`。

## 5. 問題定義

本專題將任務定義為 14 類文字分類：

> 給定一段中文新聞內容，預測新聞所屬類別。

### 5.1 輸入特徵

主要輸入文字：

```text
新聞正文
```

新聞文字長度只用於 EDA，不作為第一版額外數值特徵。第一版模型以 MacBERT tokenizer 處理新聞正文。

### 5.2 預測標籤

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

## 6. 資料處理

資料前處理流程：

1. 讀取本機 THUCNews 分類資料夾。
2. 將資料夾名稱作為 `label_name`，讀取 `.txt` 檔內容作為新聞正文。
3. 對新聞正文做基本空白清理。
4. 移除無法解析類別或新聞正文為空的資料。
5. 建立欄位：

| 欄位 | 說明 |
| --- | --- |
| `text` | 清理後新聞正文 |
| `label_name` | 新聞類別文字 |
| `label` | 類別編號 |
| `text_length` | 新聞正文長度 |

不使用 jieba 斷詞、不刪停用詞、不刪標點，因為 MacBERT 會使用自己的 tokenizer 處理中文文字，保留完整語境較適合模型學習。

## 7. EDA

需輸出下列圖表與摘要：

| 圖表 | 輸出位置 | 用途 |
| --- | --- | --- |
| 類別分布圖 | `outputs/figures/label_distribution.png` | 檢查 14 類新聞資料量 |
| 新聞長度分布圖 | `outputs/figures/text_length_distribution.png` | 分析新聞正文長度 |
| 類別 vs 新聞長度箱型圖 | `outputs/figures/text_length_by_label.png` | 比較不同類別文字長度差異 |

另需輸出：

```text
outputs/eda_summary.json
outputs/label_counts.csv
```

## 8. 模型訓練

使用 `hfl/chinese-macbert-base` 加上 sequence classification head。

建議訓練設定：

| 項目 | 建議值 |
| --- | --- |
| split | 80% train, 10% validation, 10% test |
| epochs | 3 到 5 |
| max_length | 256 |
| batch_size | 32；若 VRAM 不足改 16 或 8 |
| learning_rate | 2e-5 |
| weight_decay | 0.01 |
| fp16 | 開啟 |
| dataloader_pin_memory | 開啟 |
| dataloader_num_workers | 4 |
| TensorBoard log dir | `outputs_train_news/tensorboard` |

正式訓練只使用 CUDA。若沒有 CUDA，程式需停止訓練。

正式訓練指令：

```bash
uv run python src/main.py --data-path THUCNews --skip-eda --epochs 3 --batch-size 32 --max-length 256 --dataloader-num-workers 4 --output-dir outputs_train_news --logging-dir outputs_train_news/tensorboard --model-dir models/macbert_news_classifier
```

若 12GB VRAM 不足：

```bash
uv run python src/main.py --data-path THUCNews --skip-eda --epochs 3 --batch-size 16 --max-length 256 --dataloader-num-workers 4 --output-dir outputs_train_news --logging-dir outputs_train_news/tensorboard --model-dir models/macbert_news_classifier
```

## 9. 評估方式

資料切分：

| 資料 | 比例 |
| --- | --- |
| 訓練集 | 80% |
| 驗證集 | 10% |
| 測試集 | 10% |

評估指標：

1. Accuracy
2. Macro precision
3. Macro recall
4. Macro F1-score
5. Weighted F1-score
6. Confusion matrix

測試集評估輸出：

```text
outputs_train_news/metrics.json
outputs_train_news/predictions.csv
outputs_train_news/figures/confusion_matrix.png
```

## 10. 網頁推論

模型訓練完成後，使用 Gradio 建立網頁介面。

啟動指令：

```bash
uv run python src/app.py --host 0.0.0.0 --port 7860 --model-dir models/macbert_news_classifier
```

使用者輸入中文新聞內容後，網頁輸出：

1. 預測新聞類別。
2. 各類別機率。

網頁推論端需使用 OpenCC 將繁體輸入轉成簡體後再送入模型，使輸入形式與 THUCNews 訓練資料一致。模型內部類別仍使用資料集原始簡體 label，網頁顯示層則轉為繁體中文類別。

## 11. 預期專案結構

```text
nlp/
├── AGENTS.md
├── PROJECT_SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── main.py
│   ├── app.py
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── news_dataset.py
│   ├── training.py
│   ├── eda.py
│   └── evaluation.py
├── outputs_train_news/
└── models/
    └── macbert_news_classifier/
```

## 12. 成功標準

專題完成時需達成以下條件：

1. 能用 `uv run python src/main.py --dev-sample 5000 --skip-train` 完成資料處理與 EDA。
2. 能成功載入並處理本機 THUCNews 分類資料夾。
3. 能輸出 EDA 圖表與 `label_counts.csv`。
4. 能微調 `hfl/chinese-macbert-base` 新聞分類模型。
5. 能輸出 accuracy、macro precision、macro recall、macro F1-score、weighted F1-score。
6. 能輸出 confusion matrix。
7. 能輸出 `training_summary.json`、`metrics.json`、`predictions.csv` 與 TensorBoard logs。
8. 能使用 `uv run python src/app.py --host 0.0.0.0 --port 7860 --model-dir models/macbert_news_classifier` 啟動網頁推論介面。
9. 網頁推論能接受繁體與簡體新聞內容，並以繁體中文顯示預測類別。
10. GitHub 上包含完整程式碼與 README。
11. 投影片中包含程式執行畫面、資料分析圖、MacBERT 模型結果與網頁推論畫面。

## 13. 報告重點

1. 說明 THUCNews 是中文新聞分類資料集。
2. 展示 14 類新聞類別分布。
3. 展示新聞文字長度分布。
4. 說明 MacBERT 如何將新聞正文分類成新聞主題。
5. 呈現 classification report 與 confusion matrix。
6. 展示 Gradio 網頁輸入新聞內容並輸出類別。
