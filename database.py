# app/models/database.py - Database Models
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any
from settings import settings

Base = declarative_base()

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG
)

# Session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

class Document(Base):
    """Document storage model"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    document_type = Column(String, nullable=False)  # pdf, docx, email
    source_url = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    queries = relationship("QueryLog", back_populates="document")

class DocumentChunk(Base):
    """Document chunk model for semantic search"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    embedding_vector = Column(JSON, nullable=True)  # Store as JSON for FAISS
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class QueryLog(Base):
    """Query logging and analytics"""
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    query_text = Column(Text, nullable=False)
    parsed_query = Column(JSON, nullable=True)
    answer_text = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    relevant_chunks = Column(JSON, nullable=True)
    reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="queries")

class EmbeddingCache(Base):
    """Cache for embeddings to avoid recomputation"""
    __tablename__ = "embedding_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    content_hash = Column(String, unique=True, index=True, nullable=False)
    content = Column(Text, nullable=False)
    embedding_vector = Column(JSON, nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Dependency to get database session
async def get_db_session() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Initialize database
async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Database utilities
class DatabaseManager:
    """Database operation manager"""
    
    @staticmethod
    async def create_document(
        content: str,
        title: Optional[str] = None,
        document_type: str = "pdf",
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a new document record"""
        async with AsyncSessionLocal() as session:
            document = Document(
                title=title,
                content=content,
                document_type=document_type,
                source_url=source_url,
                metadata=metadata,
                file_size=len(content)
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)
            return document
    
    @staticmethod
    async def create_document_chunks(
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Create document chunks"""
        async with AsyncSessionLocal() as session:
            chunk_objects = []
            for i, chunk_data in enumerate(chunks):
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=chunk_data["content"],
                    chunk_index=i,
                    start_char=chunk_data.get("start_char"),
                    end_char=chunk_data.get("end_char"),
                    metadata=chunk_data.get("metadata"),
                    embedding_vector=chunk_data.get("embedding_vector")
                )
                chunk_objects.append(chunk)
                session.add(chunk)
            
            await session.commit()
            for chunk in chunk_objects:
                await session.refresh(chunk)
            return chunk_objects
    
    @staticmethod
    async def log_query(
        query_text: str,
        document_id: Optional[int] = None,
        parsed_query: Optional[Dict[str, Any]] = None,
        answer_text: Optional[str] = None,
        confidence_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None,
        relevant_chunks: Optional[List[Dict[str, Any]]] = None,
        reasoning: Optional[str] = None
    ) -> QueryLog:
        """Log a query and its results"""
        async with AsyncSessionLocal() as session:
            query_log = QueryLog(
                document_id=document_id,
                query_text=query_text,
                parsed_query=parsed_query,
                answer_text=answer_text,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                relevant_chunks=relevant_chunks,
                reasoning=reasoning
            )
            session.add(query_log)
            await session.commit()
            await session.refresh(query_log)
            return query_log
    
    @staticmethod
    async def get_or_create_embedding_cache(
        content: str,
        embedding_vector: List[float],
        model_name: str
    ) -> EmbeddingCache:
        """Get or create embedding cache entry"""
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        async with AsyncSessionLocal() as session:
            # Try to get existing cache entry
            existing = await session.get(EmbeddingCache, content_hash)
            if existing:
                return existing
            
            # Create new cache entry
            cache_entry = EmbeddingCache(
                content_hash=content_hash,
                content=content,
                embedding_vector=embedding_vector,
                model_name=model_name
            )
            session.add(cache_entry)
            await session.commit()
            await session.refresh(cache_entry)
            return cache_entry