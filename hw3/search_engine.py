from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from html import unescape
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

import jieba  # type: ignore
import requests
from bs4 import BeautifulSoup


@dataclass
class Document:
    doc_id: str
    url: str
    page_name: str
    pagerank: float
    page_content: str
    links: List[str]


class CampusSearchEngine:
    """Simple Chinese search engine for CGU campus websites.

    Features:
    - Crawl pages from seed URLs (>= 10k supported by design)
    - Build inverted index after Chinese tokenization
    - Compute PageRank on page-link graph
    - Query with keyword(s), sorted by PageRank
    - Precision / Recall evaluation for query results
    - Save / load JSON dataset with required schema
    """

    def __init__(
        self,
        allowed_domains: Optional[Iterable[str]] = None,
        user_agent: str = "NLP2026-HW3-SearchEngine/1.0",
        timeout: int = 8,
    ) -> None:
        self.allowed_domains = set(allowed_domains or ["cgu.edu.tw"])
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

        self.documents: Dict[str, Document] = {}
        self.url_to_doc_id: Dict[str, str] = {}
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.graph_outgoing: Dict[str, Set[str]] = defaultdict(set)
        self.pagerank: Dict[str, float] = {}

    # --------------------- Crawl ---------------------
    def crawl(
        self,
        seed_urls: List[str],
        max_docs: int = 10_000,
        sleep_sec: float = 0.05,
    ) -> None:
        queue = deque(seed_urls)
        visited: Set[str] = set()
        graph_outgoing_urls: Dict[str, Set[str]] = defaultdict(set)
        self.graph_outgoing.clear()

        while queue and len(self.documents) < max_docs:
            url = self._normalize_url(queue.popleft())
            if not url or url in visited:
                continue
            visited.add(url)
            if not self._is_allowed(url):
                continue

            html = self._safe_get(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            title = self._extract_title(soup)
            content = self._extract_text(soup)
            if not content.strip():
                continue

            doc_id = self._ensure_doc(url, title, content)
            outgoing_urls = self._extract_links(url, soup)

            for linked_url in outgoing_urls:
                if self._is_allowed(linked_url) and linked_url not in visited:
                    queue.append(linked_url)

            graph_outgoing_urls.setdefault(doc_id, set())
            for linked_url in outgoing_urls:
                linked_url = self._normalize_url(linked_url)
                if not linked_url or not self._is_allowed(linked_url):
                    continue
                if linked_url != url:
                    graph_outgoing_urls[doc_id].add(linked_url)

            if sleep_sec > 0:
                time.sleep(sleep_sec)

        # Convert URL graph into doc_id graph (only crawled pages are valid docs)
        for doc_id, doc in self.documents.items():
            mapped_links: Set[str] = set()
            for linked_url in graph_outgoing_urls.get(doc_id, set()):
                linked_doc_id = self.url_to_doc_id.get(linked_url)
                if linked_doc_id and linked_doc_id != doc_id:
                    mapped_links.add(linked_doc_id)
            self.graph_outgoing[doc_id] = mapped_links
            doc.links = sorted(mapped_links)

    # --------------------- Index ---------------------
    def build_inverted_index(self, min_token_len: int = 1) -> None:
        self.inverted_index.clear()
        for doc_id, doc in self.documents.items():
            if not doc.page_content:
                continue
            tokens = set(self.tokenize(doc.page_content))
            for token in tokens:
                if len(token) >= min_token_len:
                    self.inverted_index[token].add(doc_id)

    def tokenize(self, text: str) -> List[str]:
        text = unescape(text)
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        return tokens

    # --------------------- PageRank ---------------------
    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> Dict[str, float]:
        docs = list(self.documents.keys())
        n = len(docs)
        if n == 0:
            self.pagerank = {}
            return {}

        pr = {doc_id: 1.0 / n for doc_id in docs}

        for _ in range(max_iter):
            new_pr = {doc_id: (1.0 - damping) / n for doc_id in docs}

            dangling_mass = 0.0
            for src in docs:
                outs = self.graph_outgoing.get(src, set())
                if not outs:
                    dangling_mass += pr[src]
                    continue
                share = pr[src] / len(outs)
                for dst in outs:
                    if dst in new_pr:
                        new_pr[dst] += damping * share

            if dangling_mass > 0:
                extra = damping * dangling_mass / n
                for doc_id in docs:
                    new_pr[doc_id] += extra

            delta = sum(abs(new_pr[d] - pr[d]) for d in docs)
            pr = new_pr
            if delta < tol:
                break

        self.pagerank = pr
        for doc_id, score in pr.items():
            self.documents[doc_id].pagerank = score
        return pr

    # --------------------- Query ---------------------
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        query_tokens = [t for t in self.tokenize(query_text) if t.strip()]
        if not query_tokens:
            return []

        # "任何字都可以" -> OR semantics by union
        matched_docs: Set[str] = set()
        for token in query_tokens:
            matched_docs |= self.inverted_index.get(token, set())

        if not matched_docs:
            return []

        token_counts = Counter(query_tokens)

        def sort_key(doc_id: str) -> Tuple[float, float, str]:
            pr = self.pagerank.get(doc_id, 0.0)
            tf_overlap = 0
            for tok, cnt in token_counts.items():
                tf_overlap += self.documents[doc_id].page_content.count(tok) * cnt
            return (pr, math.log1p(tf_overlap), doc_id)

        ranked = sorted(matched_docs, key=sort_key, reverse=True)[:top_k]
        return [
            (doc_id, self.pagerank.get(doc_id, 0.0), self.documents[doc_id].page_name or self.documents[doc_id].url)
            for doc_id in ranked
        ]

    def print_query_results(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        results = self.query(query_text, top_k=top_k)
        print(
            f"您的搜尋結果 (Sorting by PageRank Value)：共 {len(results)} 筆，符合「{query_text}」--- 共 indexing {len(self.documents)} 筆資料"
        )
        for doc_id, score, title in results:
            print(f"{doc_id} ({score:.5f}): {title}")
        return results

    def evaluate_query(self, query_text: str, relevant_doc_ids: Set[str], top_k: int = 10) -> Dict[str, float]:
        results = self.query(query_text, top_k=top_k)
        returned_ids = {doc_id for doc_id, _, _ in results}

        tp = len(returned_ids & relevant_doc_ids)
        precision = tp / len(returned_ids) if returned_ids else 0.0
        recall = tp / len(relevant_doc_ids) if relevant_doc_ids else 0.0

        return {"precision": precision, "recall": recall}

    # --------------------- JSON IO ---------------------
    def save_json(self, output_path: str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(doc) for doc in sorted(self.documents.values(), key=lambda d: d.doc_id)]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_json(self, input_path: str) -> None:
        with open(input_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.documents.clear()
        self.url_to_doc_id.clear()
        self.graph_outgoing.clear()
        self.inverted_index.clear()
        self.pagerank.clear()

        for row in payload:
            doc = Document(
                doc_id=row["doc_id"],
                url=row.get("url", ""),
                page_name=row.get("page_name", ""),
                pagerank=float(row.get("pagerank", 0.0)),
                page_content=row.get("page_content", ""),
                links=list(row.get("links", [])),
            )
            self.documents[doc.doc_id] = doc
            if doc.url:
                self.url_to_doc_id[doc.url] = doc.doc_id
            self.graph_outgoing[doc.doc_id] = set(doc.links)
            self.pagerank[doc.doc_id] = doc.pagerank

    # --------------------- Internal utils ---------------------
    def _safe_get(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype and "application/xhtml" not in ctype:
                return ""
            if resp.status_code != 200:
                return ""
            resp.encoding = resp.apparent_encoding or resp.encoding
            return resp.text
        except Exception:
            return ""

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        url, _ = urldefrag(url.strip())
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        return url

    def _is_allowed(self, url: str) -> bool:
        netloc = urlparse(url).netloc.lower()
        return any(netloc == d or netloc.endswith(f".{d}") for d in self.allowed_domains)

    def _extract_title(self, soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(" ", strip=True)
        return "(no title)"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.extract()
        text = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text)

    def _extract_links(self, base_url: str, soup: BeautifulSoup) -> Set[str]:
        links: Set[str] = set()
        for a in soup.find_all("a", href=True):
            href_raw = a.get("href")
            if isinstance(href_raw, list):
                if not href_raw:
                    continue
                href = str(href_raw[0]).strip()
            elif isinstance(href_raw, str):
                href = href_raw.strip()
            else:
                continue

            if not href:
                continue
            if href.startswith(("mailto:", "javascript:", "tel:")):
                continue
            full_url = self._normalize_url(urljoin(base_url, href))
            if full_url:
                links.add(full_url)
        return links

    def _ensure_doc(self, url: str, title: str, content: str) -> str:
        if url in self.url_to_doc_id:
            doc_id = self.url_to_doc_id[url]
            doc = self.documents[doc_id]
            if title and not doc.page_name:
                doc.page_name = title
            if content and not doc.page_content:
                doc.page_content = content
            return doc_id

        doc_id = f"{len(self.documents) + 1:05d}"
        self.url_to_doc_id[url] = doc_id
        self.documents[doc_id] = Document(
            doc_id=doc_id,
            url=url,
            page_name=title,
            pagerank=0.0,
            page_content=content,
            links=[],
        )
        return doc_id


def _main() -> None:
    # ===========================
    # HW3 參數區（直接在 Python 內修改）
    # ===========================
    # 1) seed_urls:
    #    - 搜尋引擎爬蟲的起始網址清單。
    #    - 可以放 1 個或多個 URL，系統會從這些入口頁面向外擴展抓取。
    #    - 建議至少包含學校首頁與系所首頁，可提高覆蓋率。
    seed_urls: List[str] = [
        "https://www.cgu.edu.tw/",
        "https://cs.cgu.edu.tw/",
    ]

    # 2) max_docs:
    #    - 最多要收集的「文件數」（一個頁面 = 一個文件）。
    #    - 作業要求至少 10000 筆時，請改成 10000 或以上。
    #    - 開發測試時可先設小一點（如 200）加快流程。
    max_docs: int = 200

    # 3) query_text:
    #    - 查詢關鍵字，任何字都可以（符合作業第 6 點）。
    #    - 例如：「長庚資工」、「獎學金」、「招生」等。
    query_text: str = "長庚資工"

    # 4) top_k:
    #    - 查詢結果要顯示前幾筆。
    #    - 系統會依 PageRank 由高到低排序後輸出前 K 筆。
    top_k: int = 10

    # 5) output_json_path:
    #    - 最終資料集輸出 JSON 檔案路徑。
    #    - 會儲存每篇文件的 doc_id/page_name/pagerank/page_content/links/url。
    #    - 如果資料夾不存在，程式會自動建立。
    output_json_path: str = "cgu_pages.json"

    # 6) sleep_sec:
    #    - 每次 HTTP 請求之間的等待秒數（禮貌爬蟲節流）。
    #    - 數值越大越不容易對網站造成負載，但整體速度會較慢。
    sleep_sec: float = 0.05

    # 7) relevant_doc_ids:
    #    - (可選) 用於計算 Precision / Recall 的「相關文件」標註集合。
    #    - 若留空，程式只會輸出查詢結果，不計算 Precision/Recall。
    #    - 若有標註資料，填入對應 doc_id（例如 {"08021", "00109"}）。
    relevant_doc_ids: Set[str] = set()

    engine = CampusSearchEngine(allowed_domains=["cgu.edu.tw"])
    print(f"[1/4] 開始爬蟲，目標 {max_docs} 筆...")
    engine.crawl(seed_urls=seed_urls, max_docs=max_docs, sleep_sec=sleep_sec)
    print(f"[完成] 已收集文件數: {len(engine.documents)}")

    print("[2/4] 建立倒排索引...")
    engine.build_inverted_index()
    print(f"[完成] 詞項數: {len(engine.inverted_index)}")

    print("[3/4] 計算 PageRank...")
    engine.compute_pagerank()
    print("[完成] PageRank 計算完成")

    print("[4/4] 儲存 JSON + 執行查詢...")
    engine.save_json(output_json_path)
    print(f"[完成] JSON 已輸出到: {output_json_path}")
    engine.print_query_results(query_text, top_k=top_k)
    if relevant_doc_ids:
        metrics = engine.evaluate_query(query_text, relevant_doc_ids=relevant_doc_ids, top_k=top_k)
        print(f"Precision : {metrics['precision']:.2%}")
        print(f"Recall : {metrics['recall']:.2%}")


if __name__ == "__main__":
    _main()
