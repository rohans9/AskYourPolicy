# app/services/vector_search_service.py - Vector Search Service
import asyncio
import os
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
import time

# Vector database imports
import faiss
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available, using FAISS only")

from loguru import logger
from app.config.settings import settings
from app.utils.exceptions import VectorSearchError
from app.services.document_processor import DocumentChunk
from app.schemas.responses import RelevantChunk

@dataclass 
class SearchResult:
    """Vector search result structure"""
    chunk_id: str
    similarity_score: float
    content: str
    metadata: Dict[str, Any]

class VectorSearchService:
    """Service for vector-based semantic search using FAISS or Pinecone"""
    
    def __init__(self):
        self.vector_db_type = settings.VECTOR_DB_TYPE.lower()
        self.dimension = None
        
        # FAISS components
        self.faiss_index = None
        self.faiss_metadata = {}
        self.faiss_index_path = settings.FAISS_INDEX_PATH
        
        # Pinecone components
        self.pinecone_index = None
        
        # Initialize the selected vector database
        asyncio.create_task(self._initialize_vector_db())
    
    async def _initialize_vector_db(self):
        """Initialize the vector database based on configuration"""
        try:
            if self.vector_db_type == "pinecone" and PINECONE_AVAILABLE:
                await self._initialize_pinecone()
            else:
                await self._initialize_faiss()
                
            logger.info(f"Vector database initialized: {self.vector_db_type}")
            
        except Exception as e:
            logger.error(f"Vector database initialization failed: {str(e)}")
            # Fallback to FAISS if Pinecone fails
            if self.vector_db_type == "pinecone":
                logger.info("Falling back to FAISS")
                self.vector_db_type = "faiss"
                await self._initialize_faiss()
    
    async def _initialize_faiss(self):
        """Initialize FAISS vector database"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            
            # Try to load existing index
            index_file = f"{self.faiss_index_path}.index"
            metadata_file = f"{self.faiss_index_path}.metadata"
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                # Load existing index
                self.faiss_index = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    self.faiss_metadata = pickle.load(f)
                
                self.dimension = self.faiss_index.d
                logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                # Initialize empty index (dimension will be set when first vectors are added)
                self.faiss_metadata = {}
                logger.info("Initialized empty FAISS index")
                
        except Exception as e:
            logger.error(f"FAISS initialization failed: {str(e)}")
            raise VectorSearchError(f"Failed to initialize FAISS: {str(e)}")
    
    async def _initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            if not PINECONE_AVAILABLE:
                raise VectorSearchError("Pinecone library not available")
            
            if not settings.PINECONE_API_KEY:
                raise VectorSearchError("Pinecone API key not provided")
            
            # Initialize Pinecone
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )
            
            # Connect to or create index
            index_name = settings.PINECONE_INDEX_NAME
            
            if index_name not in pinecone.list_indexes():
                # Create index (dimension will be set when first vectors are added)
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,  # Default dimension, will be updated
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            self.pinecone_index = pinecone.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {str(e)}")
            raise VectorSearchError(f"Failed to initialize Pinecone: {str(e)}")
    
    async def store_embeddings(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ):
        """
        Store document chunks and their embeddings in vector database
        
        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors corresponding to chunks
        """
        try:
            if len(chunks) != len(embeddings):
                raise VectorSearchError("Number of chunks and embeddings must match")
            
            if not embeddings:
                logger.warning("No embeddings to store")
                return
            
            logger.info(f"Storing {len(embeddings)} embeddings in {self.vector_db_type}")
            
            if self.vector_db_type == "pinecone":
                await self._store_embeddings_pinecone(chunks, embeddings)
            else:
                await self._store_embeddings_faiss(chunks, embeddings)
                
            logger.info(f"Successfully stored {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise VectorSearchError(f"Embedding storage failed: {str(e)}")
    
    async def _store_embeddings_faiss(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ):
        """Store embeddings in FAISS index"""
        try:
            # Convert embeddings to numpy array
            embedding_array = np.array(embeddings, dtype=np.float32)
            
            # Initialize index if not exists
            if self.faiss_index is None:
                self.dimension = embedding_array.shape[1]
                self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                logger.info(f"Created FAISS index with dimension {self.dimension}")
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embedding_array)
            
            # Add vectors to index
            start_id = self.faiss_index.ntotal
            self.faiss_index.add(embedding_array)
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                vector_id = start_id + i
                self.faiss_metadata[vector_id] = {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata
                }
            
            # Save index and metadata to disk
            await self._save_faiss_index()
            
        except Exception as e:
            logger.error(f"FAISS storage failed: {str(e)}")
            raise VectorSearchError(f"FAISS storage error: {str(e)}")
    
    async def _store_embeddings_pinecone(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ):
        """Store embeddings in Pinecone index"""
        try:
            # Prepare upsert data
            vectors_to_upsert = []
            
            for chunk, embedding in zip(chunks, embeddings):
                vector_data = {
                    'id': chunk.chunk_id,
                    'values': embedding,
                    'metadata': {
                        'content': chunk.content[:1000],  # Pinecone metadata size limit
                        'chunk_index': chunk.chunk_index,
                        'document_id': chunk.metadata.get('document_id', ''),
                        'document_type': chunk.metadata.get('document_type', '')
                    }
                }
                vectors_to_upsert.append(vector_data)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                await asyncio.to_thread(self.pinecone_index.upsert, vectors=batch)
                
        except Exception as e:
            logger.error(f"Pinecone storage failed: {str(e)}")
            raise VectorSearchError(f"Pinecone storage error: {str(e)}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        document_filter: Optional[str] = None
    ) -> List[RelevantChunk]:
        """
        Search for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            document_filter: Optional document ID to filter by
            
        Returns:
            List[RelevantChunk]: List of relevant chunks
        """
        try:
            logger.debug(f"Searching for {top_k} similar chunks")
            
            if self.vector_db_type == "pinecone":
                results = await self._search_pinecone(
                    query_embedding, top_k, similarity_threshold, document_filter
                )
            else:
                results = await self._search_faiss(
                    query_embedding, top_k, similarity_threshold, document_filter
                )
            
            # Convert to RelevantChunk objects
            relevant_chunks = []
            for result in results:
                chunk = RelevantChunk(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    similarity_score=result.similarity_score,
                    chunk_index=result.metadata.get('chunk_index', 0),
                    metadata=result.metadata
                )
                relevant_chunks.append(chunk)
            
            logger.debug(f"Found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise VectorSearchError(f"Search failed: {str(e)}")
    
    async def _search_faiss(
        self,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float,
        document_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Search using FAISS index"""
        try:
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Convert query to numpy array and normalize
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.faiss_index.search(query_vector, min(top_k * 2, self.faiss_index.ntotal))
            
            # Convert results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS uses -1 for invalid results
                    continue
                
                if score < similarity_threshold:
                    continue
                
                metadata = self.faiss_metadata.get(int(idx), {})
                
                # Apply document filter if specified
                if document_filter and metadata.get('metadata', {}).get('document_id') != document_filter:
                    continue
                
                result = SearchResult(
                    chunk_id=metadata.get('chunk_id', f'chunk_{idx}'),
                    similarity_score=float(score),
                    content=metadata.get('content', ''),
                    metadata=metadata.get('metadata', {})
                )
                results.append(result)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"FAISS search failed: {str(e)}")
            raise VectorSearchError(f"FAISS search error: {str(e)}")
    
    async def _search_pinecone(
        self,
        query_embedding: List[float],
        top_k: int,
        similarity_threshold: float,
        document_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Search using Pinecone index"""
        try:
            # Prepare query
            query_params = {
                'vector': query_embedding,
                'top_k': top_k,
                'include_metadata': True
            }
            
            # Add filter if specified
            if document_filter:
                query_params['filter'] = {'document_id': document_filter}
            
            # Execute search
            search_response = await asyncio.to_thread(
                self.pinecone_index.query,
                **query_params
            )
            
            # Convert results
            results = []
            for match in search_response.matches:
                if match.score < similarity_threshold:
                    continue
                
                result = SearchResult(
                    chunk_id=match.id,
                    similarity_score=float(match.score),
                    content=match.metadata.get('content', ''),
                    metadata={
                        'chunk_index': match.metadata.get('chunk_index', 0),
                        'document_id': match.metadata.get('document_id', ''),
                        'document_type': match.metadata.get('document_type', '')
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Pinecone search failed: {str(e)}")
            raise VectorSearchError(f"Pinecone search error: {str(e)}")
    
    async def _save_faiss_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.faiss_index is not None:
                index_file = f"{self.faiss_index_path}.index"
                metadata_file = f"{self.faiss_index_path}.metadata"
                
                # Save index
                faiss.write_index(self.faiss_index, index_file)
                
                # Save metadata
                with open(metadata_file, 'wb') as f:
                    pickle.dump(self.faiss_metadata, f)
                
                logger.debug("FAISS index and metadata saved")
                
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            stats = {
                'vector_db_type': self.vector_db_type,
                'dimension': self.dimension
            }
            
            if self.vector_db_type == "faiss":
                stats.update({
                    'total_vectors': self.faiss_index.ntotal if self.faiss_index else 0,
                    'index_file_exists': os.path.exists(f"{self.faiss_index_path}.index"),
                    'metadata_entries': len(self.faiss_metadata)
                })
            elif self.vector_db_type == "pinecone":
                try:
                    index_stats = await asyncio.to_thread(self.pinecone_index.describe_index_stats)
                    stats.update({
                        'total_vectors': index_stats.total_vector_count,
                        'namespaces': index_stats.namespaces
                    })
                except Exception as e:
                    stats['error'] = str(e)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {'error': str(e)}
    
    async def delete_by_document(self, document_id: str):
        """Delete all vectors for a specific document"""
        try:
            logger.info(f"Deleting vectors for document: {document_id}")
            
            if self.vector_db_type == "pinecone":
                await self._delete_pinecone_by_document(document_id)
            else:
                await self._delete_faiss_by_document(document_id)
                
        except Exception as e:
            logger.error(f"Failed to delete document vectors: {str(e)}")
            raise VectorSearchError(f"Vector deletion failed: {str(e)}")
    
    async def _delete_faiss_by_document(self, document_id: str):
        """Delete FAISS vectors by document ID (requires rebuilding index)"""
        try:
            if not self.faiss_index or self.faiss_index.ntotal == 0:
                return
            
            # Find vectors to keep
            vectors_to_keep = []
            metadata_to_keep = {}
            
            for vector_id, metadata in self.faiss_metadata.items():
                if metadata.get('metadata', {}).get('document_id') != document_id:
                    vectors_to_keep.append(vector_id)
            
            if len(vectors_to_keep) == len(self.faiss_metadata):
                logger.info("No vectors found for document deletion")
                return
            
            # This is complex with FAISS - for now, just mark as deleted in metadata
            # In production, you might want to periodically rebuild the index
            deleted_count = 0
            for vector_id, metadata in list(self.faiss_metadata.items()):
                if metadata.get('metadata', {}).get('document_id') == document_id:
                    del self.faiss_metadata[vector_id]
                    deleted_count += 1
            
            await self._save_faiss_index()
            logger.info(f"Marked {deleted_count} vectors as deleted")
            
        except Exception as e:
            logger.error(f"FAISS deletion failed: {str(e)}")
            raise VectorSearchError(f"FAISS deletion error: {str(e)}")
    
    async def _delete_pinecone_by_document(self, document_id: str):
        """Delete Pinecone vectors by document ID"""
        try:
            delete_response = await asyncio.to_thread(
                self.pinecone_index.delete,
                filter={'document_id': document_id}
            )
            logger.info(f"Deleted Pinecone vectors for document: {document_id}")
            
        except Exception as e:
            logger.error(f"Pinecone deletion failed: {str(e)}")
            raise VectorSearchError(f"Pinecone deletion error: {str(e)}")
    
    async def clear_index(self):
        """Clear all vectors from the index"""
        try:
            logger.info("Clearing vector index")
            
            if self.vector_db_type == "pinecone":
                await asyncio.to_thread(self.pinecone_index.delete, delete_all=True)
            else:
                # Reset FAISS index
                if self.dimension:
                    self.faiss_index = faiss.IndexFlatIP(self.dimension)
                else:
                    self.faiss_index = None
                self.faiss_metadata = {}
                await self._save_faiss_index()
            
            logger.info("Vector index cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear index: {str(e)}")
            raise VectorSearchError(f"Index clearing failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of vector search service"""
        health_status = {
            'status': 'healthy',
            'vector_db_type': self.vector_db_type,
            'dimension': self.dimension
        }
        
        try:
            # Get index statistics
            stats = await self.get_index_stats()
            health_status.update(stats)
            
            # Test search functionality if index has vectors
            if stats.get('total_vectors', 0) > 0:
                # Create dummy query vector
                if self.dimension:
                    dummy_query = [0.1] * self.dimension
                    test_results = await self.search_similar(dummy_query, top_k=1)
                    health_status['search_test'] = len(test_results) > 0
                else:
                    health_status['search_test'] = False
            else:
                health_status['search_test'] = None  # Cannot test without vectors
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status