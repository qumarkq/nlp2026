# 搜尋引擎實作

1. 搜尋引擎實作
2. 長庚校網資料 (⾄少爬10000筆)
3. ⼀個⾴⾯算⼀個⽂件
4. 中⽂分詞後，建立 Inverted Index
5. 利⽤ PageRank 演算法來排序
6. 輸入搜尋關鍵字，任何字都可以
   1. ex: search_engine.query(”長庚資⼯“)
7. 範例輸出搜尋結果呈現
   1. 格式為<doc_id, PageRank Value, 網頁title>
   2. 範例結果
      1. 您的搜尋結果 (Sorting by PageRank Value)：共 3 筆，符合”長庚資⼯“ - - - 共 indexing 999 筆電影資料
        8021 (0.34934): 長庚⼤學資訊⼯程學系
        0109 (0.14233): 資訊⼯程學系 最新消息
        1023 (0.09176): 資訊⼯程學系 專任教師
8. 計算查詢後的Precision and recall值
    ex: Precision : 85%
9. 校網資料檔請以JSON形式儲存
   1.  {doc_id, page_name, pagerank, page_content, links[doc_id]}