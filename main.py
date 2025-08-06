# main.py - FastAPI Application Entry Point
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Optional
import uvicorn
from loguru import logger
import os
from datetime import datetime

# Local imports
from database import init_db
from requests import HackRXRequest, DocumentProcessRequest
from responses import HackRXResponse, DocumentProcessResponse
from document_processor import DocumentProcessorService
from embedding_service import EmbeddingService
from llm_service import LLMService
from vector_search_service import VectorSearchService
from settings import settings
from exceptions import DocumentProcessingError, EmbeddingError, LLMError

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Process large documents and make contextual decisions for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessorService()
embedding_service = EmbeddingService()
llm_service = LLMService()
vector_search_service = VectorSearchService()

# Authentication dependency
async def verify_token(authorization: Optional[str] = Header(None)):
    """Verify Bearer token authentication"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        # Verify against expected token
        if token != settings.HACKRX_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    logger.info("Starting LLM Query-Retrieval System...")
    await init_db()
    logger.info("✅ Team token loaded successfully")
    logger.info("System ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down system...")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "LLM-Powered Intelligent Query-Retrieval System", "status": "running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "document_processor": "ready",
            "embedding_service": "ready",
            "llm_service": "ready",
            "vector_search": "ready"
        }
    }

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main HackRX endpoint for processing documents and answering questions
    
    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Generates embeddings for document chunks
    3. For each question, performs semantic search and LLM-based reasoning
    4. Returns structured answers with confidence scores and reasoning
    """
    try:
        logger.info(f"Processing HackRX request with {len(request.questions)} questions")
        
        # Step 1: Process the document
        logger.info("Step 1: Processing document...")
        document_content = await document_processor.process_document_from_url(request.documents)
        
        # Step 2: Generate embeddings for document chunks
        logger.info("Step 2: Generating embeddings...")
        chunks = await document_processor.chunk_document(document_content)
        embeddings = await embedding_service.generate_embeddings_batch([chunk.content for chunk in chunks])
        
        # Step 3: Store in vector database
        logger.info("Step 3: Storing in vector database...")
        await vector_search_service.store_embeddings(chunks, embeddings)
        
        # Step 4: Process each question
        logger.info("Step 4: Processing questions...")
        answers = []
        
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Parse query with LLM
            parsed_query = await llm_service.parse_query(question)
            
            # Generate query embedding
            query_embedding = await embedding_service.generate_embeddings([question])
            
            # Semantic search
            relevant_chunks = await vector_search_service.search_similar(
                query_embedding[0], 
                top_k=5
            )
            
            # Generate answer using LLM
            answer = await llm_service.generate_answer(
                question=question,
                parsed_query=parsed_query,
                relevant_chunks=relevant_chunks,
                document_metadata={"source": request.documents}
            )
            
            answers.append(answer.answer_text)
        
        logger.info(f"Successfully processed all {len(request.questions)} questions")
        
        return HackRXResponse(answers=answers)
        
    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Document processing failed: {str(e)}")
    
    except EmbeddingError as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
    
    except LLMError as e:
        logger.error(f"LLM error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-document", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    token: str = Depends(verify_token)
):
    """
    Process a single document and return structured information
    """
    try:
        logger.info(f"Processing document: {request.document_url}")
        
        # Process document
        document_content = await document_processor.process_document_from_url(request.document_url)
        
        # Generate chunks
        chunks = await document_processor.chunk_document(document_content)
        
        # Extract metadata
        metadata = await document_processor.extract_metadata(document_content)
        
        return DocumentProcessResponse(
            document_id=document_content.document_id,
            title=document_content.title,
            content_preview=document_content.content[:500] + "..." if len(document_content.content) > 500 else document_content.content,
            chunk_count=len(chunks),
            metadata=metadata,
            processing_status="completed"
        )
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )