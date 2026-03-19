from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal, Optional, Any
import yaml
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # backend/
ENV_PATH = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH, #".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ✅ 忽略 env 里多余旧字段
        case_sensitive=False,
    )

    # ---- Security ----
    secret_key: str = Field(default="09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7", alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # ---- Site Config (Must be loaded first to determine YAML path) ----
    site_id: str = Field(default="site-jwl", alias="SITE_ID")

    # ---- Database ----
    db_type: Literal["sqlite", "postgres"] = Field(default="sqlite", alias="DB_TYPE")
    # Postgres Connection String (e.g., postgresql://user:pass@host:5432/dbname)
    postgres_url: Optional[str] = Field(default=None, alias="POSTGRES_URL")
    inquiries_db_file: Optional[str] = Field(default="main.db", alias="INQUIRIES_DB_FILE")

    # ---- Redis ----
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # ---- Storage ----
    storage_backend: Literal["local", "minio", "s3"] = Field(default="local", alias="STORAGE_BACKEND")
    local_storage_dir: str = Field(default="uploads", alias="LOCAL_STORAGE_DIR")

    # ---- MinIO/S3 ----
    minio_endpoint: Optional[str] = Field(default=None, alias="MINIO_ENDPOINT")
    minio_access_key: Optional[str] = Field(default=None, alias="MINIO_ACCESS_KEY")
    minio_secret_key: Optional[str] = Field(default=None, alias="MINIO_SECRET_KEY")
    minio_bucket: Optional[str] = Field(default="site-health", alias="MINIO_BUCKET")
    minio_region: Optional[str] = Field(default="us-east-1", alias="MINIO_REGION")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")

    # ---- backend switch (给默认值可以) ----
    llm_backend: Literal["openai", "litellm", "deepseek"] = Field(default="openai", alias="LLM_BACKEND")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    # Selection for model prompt style: "default", "deepseek", "qwen", etc.
    model_type: str = Field(default="default", alias="MODEL_TYPE")
    volc_appid: Optional[str] = Field(default=None, alias="VOLC_APPID")
    volc_access_token: Optional[str] = Field(default=None, alias="VOLC_ACCESS_TOKEN")
    volc_cluster_id: str = Field(default="volc.service_type.10029", alias="VOLC_CLUSTER_ID")
    tts_default_voice: str = Field(default="zh_female_vv_uranus_bigtts", alias="TTS_DEFAULT_VOICE")
    tts_queue_maxsize: int = Field(default=64, alias="TTS_QUEUE_MAXSIZE")
    tts_gpu_or_external_workers: int = Field(default=2, alias="TTS_GPU_OR_EXTERNAL_WORKERS")

    # ---- OpenAI (API KEY 建议必填) ----
    openai_api_key: str = Field(alias="OPENAI_API_KEY")               # ✅ 无默认值：必须从 env 来
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    # ---- DeepSeek ----
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: Optional[str] = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: Optional[str] = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")

    # ---- LiteLLM (只有切换时才用，所以可以 Optional) ----
    litellm_api_key: Optional[str] = Field(default=None, alias="LITELLM_API_KEY")
    litellm_api_base: Optional[str] = Field(default=None, alias="LITELLM_API_BASE")
    litellm_model: Optional[str] = Field(default=None, alias="LITELLM_MODEL")

    # ---- Data (建议也从 env 来，但可给默认) ----
    data_dir: str = Field(default="apps/site-health/src/data", alias="DATA_DIR")
    chat_config_path: Optional[str] = Field(default=None, alias="CHAT_CONFIG_PATH")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    #new add embeddings
    embeddings_backend: Literal["openai", "litellm"] = Field(default="openai", alias="EMBEDDINGS_BACKEND")
    embeddings_model: str = Field(default="text-embedding-3-small", alias="EMBEDDINGS_MODEL")

    # OpenAI embeddings
    # openai_api_key: str = Field(alias="OPENAI_API_KEY") # Already defined above
    # openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL") # Already defined above

    # LiteLLM embeddings（走本地/多provider）
    # litellm_api_key: Optional[str] = Field(default=None, alias="LITELLM_API_KEY") # Already defined
    # litellm_api_base: Optional[str] = Field(default=None, alias="LITELLM_API_BASE") # Already defined

    # ---- AWS SES（生产建议必填；本地可 Optional）----
    aws_region: str = Field(default="us-west-2", alias="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    ses_from_email: Optional[str] = Field(default=None, alias="SES_FROM_EMAIL")
    ses_to_email: Optional[str] = Field(default=None, alias="SES_TO_EMAIL")
    ses_configuration_set: Optional[str] = Field(default=None, alias="SES_CONFIGURATION_SET")

    # ---- Features ----
    enable_semantic_search: bool = Field(default=True, alias="ENABLE_SEMANTIC_SEARCH")
    
    # Thresholds
    # Lexical: count of matched keywords + boost. Default 1.0 means at least one match.
    lexical_min_score_threshold: float = Field(default=1.0, alias="LEXICAL_MIN_SCORE_THRESHOLD")
    
    # Semantic: Cosine similarity (0.0 to 1.0). 
    # 0.25 is a reasonable default for "somewhat relevant". 0.5 is "very relevant".
    semantic_min_score_threshold: float = Field(default=0.25, alias="SEMANTIC_MIN_SCORE_THRESHOLD")
    
    # High Relevance Semantic Threshold: Cosine similarity (0.0 to 1.0).
    # Items with semantic score above this are considered "High Relevance" (shown initially).
    # Default 0.45 corresponds to a strong semantic match.
    semantic_high_relevance_threshold: float = Field(default=0.45, alias="SEMANTIC_HIGH_RELEVANCE_THRESHOLD")

    # Relevance threshold for UI "Show More" feature
    # Items with LEXICAL score >= this value will be marked as "high" relevance
    search_relevance_threshold: float = Field(default=4.0, alias="SEARCH_RELEVANCE_THRESHOLD")

    # Vector Index Backend
    vector_index_type: Literal["numpy", "faiss"] = Field(default="numpy", alias="VECTOR_INDEX_TYPE")

    # Knowledge Base
    kb_data_dir: Optional[str] = Field(default=None, alias="KB_DATA_DIR")
    kb_context_file: Optional[str] = Field(default=None, alias="KB_CONTEXT_FILE")

settings = Settings()