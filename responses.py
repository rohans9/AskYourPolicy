# app/schemas/responses.py - Pydantic Response Schemas
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class HackRXResponse(BaseModel):
    """HackRX API response schema"""
    answers: List[str] = Field(
        ...,
        description="List of answers corresponding to the input questions"
    )

class DocumentProcessResponse(BaseModel):
    """Document processing response schema"""
    document_id: str = Field(
        ...,
        description="Unique document identifier"
    )
    title: Optional[str] = Field(
        default=None,
        description="Extracted document title"
    )
    content_preview: str = Field(
        ...,
        description="Preview of document content"
    )
    chunk_count: int = Field(
        ...,
        description="Number of chunks created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    processing_status: str = Field(
        ...,
        description="Processing status (completed, failed, processing)"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Document creation timestamp"
    )

class ParsedQuery(BaseModel):
    """Parsed query structure"""
    original_query: str = Field(
        ...,
        description="Original query text"
    )
    intent: str = Field(
        ...,
        description="Detected query intent"
    )
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Extracted entities"
    )
    query_type: str = Field(
        ...,
        description="Type of query (coverage, exclusion, condition, etc.)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Parsing confidence score"
    )

class RelevantChunk(BaseModel):
    """Relevant document chunk"""
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier"
    )
    content: str = Field(
        ...,
        description="Chunk content"
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score to query"
    )
    chunk_index: int = Field(
        ...,
        description="Index of chunk in original document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata"
    )

class DecisionReasoning(BaseModel):
    """Decision reasoning structure"""
    reasoning_steps: List[str] = Field(
        ...,
        description="Step-by-step reasoning process"
    )
    evidence_chunks: List[str] = Field(
        ...,
        description="Chunks used as evidence"
    )
    confidence_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Factors contributing to confidence score"
    )
    contradictions: Optional[List[str]] = Field(
        default=None,
        description="Any contradictory information found"
    )
    assumptions: Optional[List[str]] = Field(
        default=None,
        description="Assumptions made in reasoning"
    )

class AnswerResponse(BaseModel):
    """Detailed answer response"""
    query_id: str = Field(
        ...,
        description="Unique query identifier"
    )
    question: str = Field(
        ...,
        description="Original question"
    )
    answer_text: str = Field(
        ...,
        description="Generated answer"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Answer confidence score"
    )
    parsed_query: ParsedQuery = Field(
        ...,
        description="Parsed query information"
    )
    relevant_chunks: List[RelevantChunk] = Field(
        ...,
        description="Relevant document chunks"
    )
    reasoning: DecisionReasoning = Field(
        ...,
        description="Decision reasoning"
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response creation timestamp"
    )

class BatchQueryResponse(BaseModel):
    """Batch query response"""
    results: List[AnswerResponse] = Field(
        ...,
        description="List of query results"
    )
    total_queries: int = Field(
        ...,
        description="Total number of queries processed"
    )
    successful_queries: int = Field(
        ...,
        description="Number of successfully processed queries"
    )
    failed_queries: int = Field(
        ...,
        description="Number of failed queries"
    )
    total_processing_time_ms: int = Field(
        ...,
        description="Total processing time in milliseconds"
    )

class EmbeddingResponse(BaseModel):
    """Embedding generation response"""
    embeddings: List[List[float]] = Field(
        ...,
        description="Generated embeddings"
    )
    model_used: str = Field(
        ...,
        description="Embedding model used"
    )
    dimensions: int = Field(
        ...,
        description="Embedding dimensions"
    )
    token_count: int = Field(
        ...,
        description="Total tokens processed"
    )

class VectorSearchResponse(BaseModel):
    """Vector search response"""
    results: List[RelevantChunk] = Field(
        ...,
        description="Search results"
    )
    query: str = Field(
        ...,
        description="Original search query"
    )
    total_results: int = Field(
        ...,
        description="Total number of results found"
    )
    search_time_ms: int = Field(
        ...,
        description="Search time in milliseconds"
    )

class ClauseMatch(BaseModel):
    """Clause matching result"""
    clause_text: str = Field(
        ...,
        description="Matched clause text"
    )
    clause_type: str = Field(
        ...,
        description="Type of clause (coverage, exclusion, condition, etc.)"
    )
    match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Matching confidence score"
    )
    context: str = Field(
        ...,
        description="Surrounding context"
    )
    location: Dict[str, int] = Field(
        default_factory=dict,
        description="Location in document (page, paragraph, etc.)"
    )

class ClauseMatchingResponse(BaseModel):
    """Clause matching response"""
    matches: List[ClauseMatch] = Field(
        ...,
        description="Found clause matches"
    )
    query: str = Field(
        ...,
        description="Original query"
    )
    total_matches: int = Field(
        ...,
        description="Total number of matches found"
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing time in milliseconds"
    )

class DocumentAnalysisResponse(BaseModel):
    """Document analysis response"""
    document_id: str = Field(
        ...,
        description="Document identifier"
    )
    summary: str = Field(
        ...,
        description="Document summary"
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Extracted key terms"
    )
    document_type: str = Field(
        ...,
        description="Detected document type"
    )
    language: str = Field(
        default="en",
        description="Detected language"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis metadata"
    )
    analysis_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Analysis confidence score"
    )

class SystemHealthResponse(BaseModel):
    """System health response"""
    status: str = Field(
        ...,
        description="Overall system status"
    )
    services: Dict[str, str] = Field(
        ...,
        description="Individual service statuses"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(
        ...,
        description="System version"
    )
    uptime_seconds: int = Field(
        ...,
        description="System uptime in seconds"
    )

class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    total_documents: int = Field(
        ...,
        description="Total documents processed"
    )
    total_queries: int = Field(
        ...,
        description="Total queries processed"
    )
    average_response_time_ms: float = Field(
        ...,
        description="Average response time"
    )
    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Query success rate"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics"
    )
    error_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Error count by type"
    )

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request identifier for tracking"
    )