"""集中管理所有可調參數；改參數就跑一次評估，看分數有沒有變好。"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    ollama_url: str = "http://localhost:11434"

    llm_model: str = "qwen3.5:0.8b"
    llm_temperature: float = 0.1
    # 支援思考模式的模型（如 Qwen3.5）要不要先思考；None 表示不送這個參數，交給模型預設
    llm_think: bool | None = False
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    # 切塊以「字元」為單位：中文一個字約等於一個 token，簡單又好理解
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 5

    # 檢索方式：vector 只用向量；hybrid 是向量 + BM25，用 RRF 合併
    retrieval_mode: Literal["vector", "hybrid"] = "hybrid"
    # Hybrid 與 Reranker 先各取回多少筆候選，再合併或精排成 top_k
    candidate_k: int = 20
    rrf_k: int = 60
    # Reranker 服務網址（docker compose --profile rerank up -d reranker）；留空表示不使用
    reranker_url: str = ""
    # 一般問題只查現行版本；問題提到年份、舊版時才查歷史版本
    version_filter: bool = True

    # 等待 Ollama 回應的秒數；CPU 執行 7B 模型時第一次載入可能要數分鐘
    request_timeout: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
