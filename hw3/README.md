# HW3 搜尋引擎實作說明

本資料夾提供 `search_engine.py`，對應 `hw3.md` 的 9 點要求：

1. 搜尋引擎實作：`CampusSearchEngine`
2. 校網資料爬蟲：`crawl(seed_urls, max_docs=10000)`
3. 一頁一文件：每個 URL 對應一個 `Document`
4. 中文分詞 + 倒排索引：`tokenize()` + `build_inverted_index()`
5. PageRank 排序：`compute_pagerank()`
6. 任意關鍵字查詢：`query("長庚資工")`
7. 輸出格式：`print_query_results()` 會輸出 `<doc_id, PageRank, title>`
8. Precision / Recall：`evaluate_query()`
9. JSON 儲存格式：`save_json()` 產出 `{doc_id,page_name,pagerank,page_content,links}`（另含 `url` 欄位方便追蹤來源）

## 安裝

```bash
pip install requests beautifulsoup4 jieba
```

> `jieba` 為可選；若未安裝，程式會使用內建 fallback tokenizer。

## 最小可執行範例

```bash
python hw3/search_engine.py
```

會跑一個 3 篇文件的 demo，示範：
- 建索引
- 計算 PageRank
- 查詢「長庚資工」
- 計算 precision / recall

## 真實爬蟲範例

```python
from hw3.search_engine import CampusSearchEngine

engine = CampusSearchEngine(allowed_domains=["cgu.edu.tw"])
engine.crawl([
    "https://www.cgu.edu.tw/",
    "https://cs.cgu.edu.tw/",
], max_docs=10000)
engine.build_inverted_index()
engine.compute_pagerank()
engine.save_json("hw3/cgu_pages.json")

engine.print_query_results("長庚資工", top_k=20)
metrics = engine.evaluate_query("長庚資工", relevant_doc_ids={"00001", "00032"}, top_k=20)
print("Precision:", f"{metrics['precision']:.2%}")
print("Recall:", f"{metrics['recall']:.2%}")
```

## 注意事項

- 大量爬蟲前請先確認目標網站 robots 規範與課堂規定。
- 若要爬滿 10,000 頁，建議提高 timeout 並視情況調整 `sleep_sec`。
