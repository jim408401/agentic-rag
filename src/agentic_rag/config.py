"""集中管理所有可調參數；改參數就跑一次評估，看分數有沒有變好。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    ollama_url: str = "http://localhost:11434"

    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.1
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    # 切塊以「字元」為單位：中文一個字約等於一個 token，簡單又好理解
    chunk_size: int = 400
    chunk_overlap: int = 60
    top_k: int = 5

    request_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
