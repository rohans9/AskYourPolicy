# app/schemas/requests.py - Pydantic Request Schemas
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class HackRXRequest(BaseModel):
    """HackRX API request schema"""
    documents: str = Field(
        ...,
        description="Document blob URL to process",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf"
    )
    questions: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of questions to answer about the document"
    )
    
    @validator('documents')
    def validate_document_url(cls, v):
        """Validate document URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Document URL must be a valid HTTP/HTTPS URL')
        return v
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list"""
        if not v:
            raise ValueError('At least one question is required')
        
        for question in v:
            if not question.strip():
                raise ValueError('Questions cannot be empty')
            if len(question) > 500:
                raise ValueError('Questions must be less than 500 characters')
        
        return v

class DocumentProcessRequest(BaseModel):
    """Document processing request schema"""
    document_url: str = Field(
        ...,
        description="URL of the document to process"
    )
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract document metadata"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=2000,
        description="Custom chunk size for document processing"
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        le=500,
        description="Custom chunk overlap for document processing"
    )

class QueryRequest(BaseModel):
    """Single query request schema"""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question to ask about the document"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Specific document ID to query (optional)"
    )
    include_reasoning: bool = Field(
        default=True,
        description="Whether to include reasoning in the response"
    )
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of document chunks to consider"
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for answers"
    )

class BatchQueryRequest(BaseModel):
    """Batch query request schema"""
    questions: List[QueryRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of queries to process"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Default document ID for all queries"
    )

class EmbeddingRequest(BaseModel):
    """Embedding generation request schema"""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to generate embeddings for"
    )
    model: Optional[str] = Field(
        default=None,
        description="Embedding model to use (optional)"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate texts list"""
        for text in v:
            if not text.strip():
                raise ValueError('Texts cannot be empty')
            if len(text) > 8000:
                raise ValueError('Text length must be less than 8000 characters')
        return v

class VectorSearchRequest(BaseModel):
    """Vector search request schema"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to return"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Filter by specific document ID"
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )

class ClauseMatchingRequest(BaseModel):
    """Clause matching request schema"""
    query: str = Field(
        ...,
        description="Query to match against clauses"
    )
    document_content: str = Field(
        ...,
        description="Document content to search for clauses"
    )
    clause_types: Optional[List[str]] = Field(
        default=None,
        description="Specific clause types to look for (e.g., 'coverage', 'exclusion', 'waiting_period')"
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores"
    )

class LLMProcessingRequest(BaseModel):
    """LLM processing request schema"""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Prompt for LLM processing"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context for the prompt"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8000,
        description="Maximum tokens for response"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    include_reasoning: bool = Field(
        default=True,
        description="Whether to include reasoning steps"
    )

class DocumentAnalysisRequest(BaseModel):
    """Document analysis request schema"""
    document_url: str = Field(
        ...,
        description="URL of the document to analyze"
    )
    analysis_type: str = Field(
        default="comprehensive",
        description="Type of analysis to perform"
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific areas to focus analysis on"
    )
    include_summary: bool = Field(
        default=True,
        description="Whether to include document summary"
    )
    include_key_terms: bool = Field(
        default=True,
        description="Whether to extract key terms"
    )

class FeedbackRequest(BaseModel):
    """User feedback request schema"""
    query_id: str = Field(
        ...,
        description="ID of the query being rated"
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1-5"
    )
    feedback_text: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional feedback text"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Feedback categories (e.g., 'accuracy', 'relevance', 'completeness')"
    )

class SystemConfigRequest(BaseModel):
    """System configuration request schema"""
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model to use"
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model to use"
    )
    chunk_size: Optional[int] = Field(
        default=None,
        ge=100,
        le=2000,
        description="Default chunk size"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        le=8000,
        description="Default max tokens"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Default temperature"
    )