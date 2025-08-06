# app/config/settings.py - Configuration Management
from pydantic import BaseSettings, Field
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Query-Retrieval System"
    VERSION: str = "1.0.0"
    
    # Authentication
    HACKRX_TOKEN: str = Field(
        default="2a91272cc18b579a54b1281caf08e0c883f1daf70cbafa6418ca3778fbc17df3",
        description="HackRX API authentication token"
    )
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/llm_retrieval",
        description="PostgreSQL database URL"
    )
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(
        ...,
        description="OpenAI API key for GPT-4 and embeddings"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model for LLM processing"
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model"
    )
    
    # Vector Database Configuration
    VECTOR_DB_TYPE: str = Field(
        default="faiss",
        description="Vector database type: 'faiss' or 'pinecone'"
    )
    
    # Pinecone Configuration (optional)
    PINECONE_API_KEY: Optional[str] = Field(
        default=None,
        description="Pinecone API key"
    )
    PINECONE_ENVIRONMENT: Optional[str] = Field(
        default=None,
        description="Pinecone environment"
    )
    PINECONE_INDEX_NAME: str = Field(
        default="llm-retrieval-index",
        description="Pinecone index name"
    )
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = Field(
        default="./data/faiss_index",
        description="Path to store FAISS index"
    )
    
    # Document Processing Configuration
    MAX_FILE_SIZE_MB: int = Field(
        default=50,
        description="Maximum file size in MB"
    )
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Document chunk size for processing"
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between document chunks"
    )
    
    # LLM Configuration
    MAX_TOKENS: int = Field(
        default=4000,
        description="Maximum tokens for LLM responses"
    )
    TEMPERATURE: float = Field(
        default=0.1,
        description="LLM temperature for response generation"
    )
    
    # Performance Configuration
    BATCH_SIZE: int = Field(
        default=10,
        description="Batch size for embedding generation"
    )
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=5,
        description="Maximum concurrent API requests"
    )
    
    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FILE: str = Field(
        default="./logs/app.log",
        description="Log file path"
    )
    
    # Redis Configuration (for caching)
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for caching"
    )
    CACHE_TTL: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Development Configuration
    DEBUG: bool = Field(
        default=False,
        description="Debug mode"
    )
    RELOAD: bool = Field(
        default=False,
        description="Auto-reload on code changes"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Global settings instance
settings = get_settings()

# Ensure required directories exist
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)