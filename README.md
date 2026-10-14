# agentic-rag

從最基本的 RAG 一步步做到 Agentic RAG 的個人學習專案。資料模擬公司內部繁雜的規章手冊與客戶訂單資料庫，所有元件都在地端執行。

完整規劃見 [企業級 Agentic RAG 實作規劃](https://claude.ai/code/artifact/ffd79fe3-ef92-4875-b4ea-c402192df463)。

## 目前進度

| 階段 | 內容 | 狀態 |
| --- | --- | --- |
| Phase 0 | 評估集 `eval/golden.jsonl`（29 題，含無答案題、版本題、SQL 題），檢索與答案正確性指標 | 完成 |
| Phase 1 | Naive RAG：Markdown 匯入、標題分節 + 固定長度切塊、pgvector 向量檢索、帶引用的生成、FastAPI、CLI、評估腳本 | 完成 |
| Phase 2 | Hybrid（BM25 + RRF）、Reranker、版本過濾 | 完成 |
| Phase 2 | Query Rewriting、Semantic / Parent-Child Chunking、RAGAS | 待做 |
| Phase 3 | LangGraph Agent：路由、文件檢索工具、Text-to-SQL（`sales` schema）、API 工具 | 待做 |
| Phase 4 | Langfuse 觀測、權限、Prompt Injection 防護、CI 評估回歸 | 待做 |

## 技術選型（全地端）

| 元件 | 選擇 | 理由 |
| --- | --- | --- |
| 資料庫 | PostgreSQL 17 + pgvector | 向量、全文檢索、業務資料、權限（RLS）都在同一套系統，學一套就夠；百萬級 chunk 內效能足夠 |
| LLM | Ollama + `qwen3.5:0.8b` | 約 1 GB，沒有顯示卡的筆電也跑得動；有 GPU 時可換 `qwen3.5:4b` 或 `9b` 提升品質 |
| Embedding | Ollama + `bge-m3` | 多語、1024 維，中文檢索效果好 |
| 關鍵字檢索 | BM25（Python，中文以相鄰兩字為詞） | 補足向量檢索對編號、金額、專有名詞不敏感的弱點 |
| Reranker | TEI + `BAAI/bge-reranker-v2-m3` | 多語 cross-encoder，CPU 可跑；以 Docker 服務提供，選用 |
| API | FastAPI | — |
| 套件管理 | uv | — |

## 快速開始

需要 Docker 與 [uv](https://docs.astral.sh/uv/)。

```bash
cp .env.example .env
docker compose up -d                       # 啟動 Postgres(pgvector) 與 Ollama
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen3.5:0.8b

uv sync                                    # 安裝相依套件
uv run rag ingest data/handbook            # 匯入規章手冊
uv run rag ask "到職滿三年半有幾天特休？"
uv run rag ask "特休可以遞延嗎？" --all-versions   # 連已廢止的舊版也查
uv run rag eval --name baseline            # 跑評估，結果寫到 eval/results/

uv run uvicorn agentic_rag.api.main:app --reload   # http://localhost:8000/docs
```

### 比較不同檢索方式

`ask` 與 `eval` 都可以用參數臨時覆蓋 `.env`，同一份資料比較三種設定：

```bash
uv run rag eval --phase 2 --mode vector --no-rerank --name vector
uv run rag eval --phase 2 --mode hybrid --no-rerank --name hybrid

docker compose --profile rerank up -d reranker   # 第一次會下載約 2 GB 模型
docker compose logs -f reranker                  # 看到 Ready 後再繼續
uv run rag eval --phase 2 --mode hybrid --rerank --name hybrid-rerank
```

一般問題預設只查現行版本；問題提到年份（例如「2023 年」）或「舊版、以前」時，才會連已廢止的版本一起查（`src/agentic_rag/retrieval/query.py`）。

模型大小在 `.env` 的 `RAG_LLM_MODEL` 調整。0.8B 模型速度快但答案品質有限，評估分數偏低時，可以先換大一點的模型比較看看，再判斷是檢索還是生成的問題。

## 測試

```bash
uv run pytest                  # 單元測試，不需要 Docker 與模型
uv run pytest -m postgres      # 需要 docker compose up -d postgres
uv run pytest -m ollama        # 端到端，需要 Ollama 與已下載的模型
uv run ruff check .
```

單元測試用 `tests/conftest.py` 裡的假 embedding（中文字 bigram 雜湊）與假 LLM，因此在 CI 上不用下載任何模型。

## 專案結構

```
src/agentic_rag/
  config.py          所有可調參數（RAG_ 開頭的環境變數）
  llm.py             ChatModel / Embedder 介面與 Ollama 實作
  ingestion/         讀檔（含 front matter metadata）、切塊、匯入管線
  store/             VectorStore 介面、記憶體版與 pgvector 版
  retrieval/         向量、BM25、Hybrid（RRF）、Reranker、版本過濾
  generation/        Prompt 與引用格式
  rag.py             檢索 + 生成主流程
  evaluation/        指標與評估執行器
  api/               FastAPI
  cli.py             rag ingest / ask / eval
data/handbook/       模擬規章（含一份已廢止的舊版，用來練習版本過濾）
db/init/             pgvector 資料表與 sales 訂單資料庫（含種子資料）
eval/golden.jsonl    評估集
```

## 模擬資料

- **規章手冊**：請假（含 2023 舊版）、差旅報支、資訊安全、客戶退換貨、採購，每份有 `doc_id`、版本、生效日、狀態、權限群組等 front matter。
- **干擾文件**：加班補休、福利補助、帳號權限、公務車。內容和上面的規章用詞相近（補休與特休、喪葬補助與喪假、帳號與密碼、公務車油資與自行開車油資），用來測試檢索會不會抓錯文件。
- **訂單資料庫**：`sales` schema 下的客戶、商品、訂單、訂單明細。退換貨規範中的退貨期限依客戶等級而定，Phase 3 的 Agent 需要同時查文件與 SQL 才能回答「這張訂單還能不能退」。

## 評估集題型

| 題型 | 說明 | 階段 |
| --- | --- | --- |
| fact / multi / reasoning | 單一事實、多個條件、需要推論 | 1 |
| unanswerable | 文件中沒有答案，應回覆「查無相關規定」 | 1 |
| version | 需區分現行版與廢止版 | 2 |
| sql / sql+doc | 需要查訂單資料庫，或同時查文件與資料庫 | 3 |

## 評估指標

| 指標 | 意義 |
| --- | --- |
| hit@k / recall@k | 正確文件有沒有出現在前 k 筆 |
| mrr | 正確文件排第幾名（第 1 名得 1，第 2 名得 0.5…），Reranker 主要改善這項 |
| obsolete@k | 前 k 筆中來自已廢止文件的比例（版本題不計），越低越好 |
| refusal_accuracy | 無答案題有拒答、有答案題沒有亂拒答 |
| fact_recall | 答案涵蓋了幾成關鍵事實（`must_include`，「\|」分隔同義寫法，比對時忽略空白、逗號與全半形） |
| citation_rate | 有答案的題目中，答案附上 `[n]` 引用的比例 |

`fact_recall` 是關鍵字比對，便宜又穩定，但可能漏判換句話說的正確答案。Phase 2 會再加上 LLM 評分（RAGAS）互相對照。
