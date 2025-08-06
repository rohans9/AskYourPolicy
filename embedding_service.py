# app/services/embedding_service.py - Embedding Generation Service
import asyncio
import hashlib
from typing import List, Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass
import time

# OpenAI and local model imports
import openai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import tiktoken

from loguru import logger
from settings import settings
from exceptions import EmbeddingError
from database import DatabaseManager

@dataclass
class EmbeddingResult:
    """Embedding result structure"""
    embedding: List[float]
    token_count: int
    model_used: str
    processing_time_ms: int

class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self):
        self.openai_client = None
        self.local_model = None
        self.tokenizer = None
        self._initialize_clients()
        
        # Embedding cache
        self.cache_enabled = True
        self.batch_size = settings.BATCH_SIZE
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _initialize_clients(self):
        """Initialize OpenAI and local model clients"""
        try:
            # Initialize OpenAI client
            if settings.OPENAI_API_KEY:
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI embedding client initialized")
            
            # Initialize local model as fallback
            try:
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Local embedding model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize local model: {str(e)}")
            
            # Initialize tokenizer for token counting
            try:
                self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise EmbeddingError(f"Embedding service initialization failed: {str(e)}")
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            model: Optional model name to use
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Check cache first
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []
            
            if self.cache_enabled:
                for i, text in enumerate(texts):
                    cached_embedding = await self._get_cached_embedding(text, model)
                    if cached_embedding:
                        cached_embeddings[i] = cached_embedding
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
            
            # Generate embeddings for uncached texts
            new_embeddings = []
            if uncached_texts:
                new_embeddings = await self._generate_embeddings_batch(uncached_texts, model)
                
                # Cache new embeddings
                if self.cache_enabled:
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        await self._cache_embedding(text, embedding, model or settings.OPENAI_EMBEDDING_MODEL)
            
            # Combine cached and new embeddings
            all_embeddings = [None] * len(texts)
            
            # Place cached embeddings
            for i, embedding in cached_embeddings.items():
                all_embeddings[i] = embedding
            
            # Place new embeddings
            for i, embedding in zip(uncached_indices, new_embeddings):
                all_embeddings[i] = embedding
            
            logger.info(f"Generated {len(new_embeddings)} new embeddings, used {len(cached_embeddings)} cached")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings in batches for efficiency
        
        Args:
            texts: List of text strings to embed
            model: Optional model name to use
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
                
                batch_embeddings = await self._generate_embeddings_batch(batch, model)
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                await self._rate_limit()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}")
    
    async def _generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for a single batch"""
        model_name = model or settings.OPENAI_EMBEDDING_MODEL
        
        try:
            # Try OpenAI first
            if self.openai_client and settings.OPENAI_API_KEY:
                return await self._generate_openai_embeddings(texts, model_name)
            
            # Fallback to local model
            elif self.local_model:
                return await self._generate_local_embeddings(texts)
            
            else:
                raise EmbeddingError("No embedding models available")
                
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {str(e)}")
            
            # Try fallback if OpenAI fails
            if self.local_model and "openai" in str(e).lower():
                logger.info("Falling back to local embedding model")
                return await self._generate_local_embeddings(texts)
            
            raise EmbeddingError(f"All embedding methods failed: {str(e)}")
    
    async def _generate_openai_embeddings(
        self,
        texts: List[str],
        model: str
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            start_time = time.time()
            
            # Clean and validate texts
            cleaned_texts = []
            for text in texts:
                # Remove excessive whitespace and limit length
                cleaned_text = ' '.join(text.split())
                if len(cleaned_text) > 8000:  # OpenAI limit
                    cleaned_text = cleaned_text[:8000]
                cleaned_texts.append(cleaned_text)
            
            # Make API request
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=cleaned_texts,
                model=model
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.debug(f"OpenAI embeddings generated in {processing_time}ms")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise EmbeddingError(f"OpenAI API error: {str(e)}")
    
    async def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        try:
            start_time = time.time()
            
            # Clean texts
            cleaned_texts = []
            for text in texts:
                cleaned_text = ' '.join(text.split())
                # Local model can handle longer texts
                if len(cleaned_text) > 15000:
                    cleaned_text = cleaned_text[:15000]
                cleaned_texts.append(cleaned_text)
            
            # Generate embeddings
            embeddings = await asyncio.to_thread(
                self.local_model.encode,
                cleaned_texts,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Convert to list format
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.debug(f"Local embeddings generated in {processing_time}ms")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Local embedding generation failed: {str(e)}")
            raise EmbeddingError(f"Local model error: {str(e)}")
    
    async def _get_cached_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Optional[List[float]]:
        """Get cached embedding if available"""
        try:
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            model_name = model or settings.OPENAI_EMBEDDING_MODEL
            
            # Query database for cached embedding
            # This would typically be implemented with actual database query
            # For now, return None to skip caching
            return None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached embedding: {str(e)}")
            return None
    
    async def _cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str
    ):
        """Cache embedding in database"""
        try:
            await DatabaseManager.get_or_create_embedding_cache(
                content=text,
                embedding_vector=embedding,
                model_name=model
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {str(e)}")
    
    async def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(text))
            else:
                # Rough approximation: 1 token ≈ 4 characters
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Token counting failed: {str(e)}")
            return len(text) // 4
    
    def get_embedding_dimensions(self, model: Optional[str] = None) -> int:
        """Get embedding dimensions for a model"""
        model_name = model or settings.OPENAI_EMBEDDING_MODEL
        
        dimensions_map = {
            'text-embedding-3-large': 3072,
            'text-embedding-3-small': 1536,
            'text-embedding-ada-002': 1536,
            'all-MiniLM-L6-v2': 384,  # Local model
        }
        
        return dimensions_map.get(model_name, 1536)  # Default
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Perform similarity search between query and candidates
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            if not candidate_embeddings:
                return []
            
            # Convert to numpy arrays for efficient computation
            query_vec = np.array(query_embedding)
            candidate_vecs = np.array(candidate_embeddings)
            
            # Normalize vectors
            query_norm = query_vec / np.linalg.norm(query_vec)
            candidate_norms = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
            
            # Compute cosine similarities
            similarities = np.dot(candidate_norms, query_norm)
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_similarities = similarities[top_indices]
            
            results = [(int(idx), float(sim)) for idx, sim in zip(top_indices, top_similarities)]
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise EmbeddingError(f"Similarity search error: {str(e)}")
    
    async def batch_similarity_search(
        self,
        query_embeddings: List[List[float]],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[List[tuple[int, float]]]:
        """
        Perform batch similarity search
        
        Args:
            query_embeddings: List of query embedding vectors
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results per query
            
        Returns:
            List of results for each query
        """
        try:
            results = []
            
            for query_embedding in query_embeddings:
                query_results = await self.similarity_search(
                    query_embedding, candidate_embeddings, top_k
                )
                results.append(query_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch similarity search failed: {str(e)}")
            raise EmbeddingError(f"Batch similarity search error: {str(e)}")
    
    async def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get information about embedding model"""
        model_name = model or settings.OPENAI_EMBEDDING_MODEL
        
        info = {
            'model_name': model_name,
            'dimensions': self.get_embedding_dimensions(model_name),
            'max_tokens': 8000 if 'openai' in model_name.lower() else 15000,
            'type': 'openai' if 'text-embedding' in model_name else 'local'
        }
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding service"""
        health_status = {
            'status': 'healthy',
            'openai_available': bool(self.openai_client and settings.OPENAI_API_KEY),
            'local_model_available': bool(self.local_model),
            'cache_enabled': self.cache_enabled
        }
        
        # Test embedding generation
        try:
            test_embeddings = await self.generate_embeddings(['test'])
            health_status['test_embedding_success'] = bool(test_embeddings)
            health_status['test_embedding_dimensions'] = len(test_embeddings[0]) if test_embeddings else 0
        except Exception as e:
            health_status['test_embedding_success'] = False
            health_status['test_error'] = str(e)
        
        return health_status