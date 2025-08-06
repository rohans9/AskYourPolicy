# app/services/llm_service.py - LLM Processing Service
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re

import openai
from loguru import logger

from settings import settings
from exceptions import LLMError
from database import DatabaseManager
from responses import ParsedQuery, DecisionReasoning, AnswerResponse, RelevantChunk

@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    model_used: str
    tokens_used: int
    processing_time_ms: int
    reasoning: Optional[str] = None

class LLMService:
    """Service for LLM-based query processing and answer generation"""
    
    def __init__(self):
        self.client = None
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        self._initialize_client()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if settings.OPENAI_API_KEY:
                openai.api_key = settings.OPENAI_API_KEY
                self.client = openai
                logger.info("OpenAI LLM client initialized")
            else:
                raise LLMError("OpenAI API key not provided")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise LLMError(f"LLM client initialization failed: {str(e)}")
    
    async def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse natural language query to extract intent and entities
        
        Args:
            query: Natural language query
            
        Returns:
            ParsedQuery: Parsed query structure
        """
        try:
            logger.debug(f"Parsing query: {query[:50]}...")
            
            prompt = self._build_query_parsing_prompt(query)
            
            response = await self._make_llm_request(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            try:
                parsed_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback parsing if JSON fails
                parsed_data = self._fallback_query_parsing(query)
            
            return ParsedQuery(
                original_query=query,
                intent=parsed_data.get('intent', 'unknown'),
                entities=parsed_data.get('entities', {}),
                query_type=parsed_data.get('query_type', 'general'),
                confidence=parsed_data.get('confidence', 0.5)
            )
            
        except Exception as e:
            logger.error(f"Query parsing failed: {str(e)}")
            # Return fallback parsed query
            return ParsedQuery(
                original_query=query,
                intent='unknown',
                entities={},
                query_type='general',
                confidence=0.1
            )
    
    def _build_query_parsing_prompt(self, query: str) -> str:
        """Build prompt for query parsing"""
        return f"""
Analyze the following insurance/legal query and extract structured information.

Query: "{query}"

Please extract:
1. Intent (what the user wants to know)
2. Entities (people, procedures, conditions, time periods, amounts, etc.)
3. Query type (coverage, exclusion, condition, waiting_period, claim, premium, etc.)
4. Confidence score (0.0-1.0)

Return response as JSON:
{{
    "intent": "brief description of what user wants to know",
    "entities": {{
        "medical_procedures": ["list of medical procedures mentioned"],
        "conditions": ["medical conditions or situations"],
        "time_periods": ["time periods mentioned"],
        "amounts": ["monetary amounts or limits"],
        "people": ["types of people mentioned"],
        "locations": ["places mentioned"]
    }},
    "query_type": "coverage|exclusion|condition|waiting_period|claim|premium|benefit|general",
    "confidence": 0.9
}}
"""
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Fallback query parsing using regex patterns"""
        entities = {
            "medical_procedures": [],
            "conditions": [],
            "time_periods": [],
            "amounts": [],
            "people": [],
            "locations": []
        }
        
        # Simple keyword-based classification
        query_lower = query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ['cover', 'coverage', 'covered']):
            query_type = 'coverage'
        elif any(word in query_lower for word in ['exclude', 'exclusion', 'not covered']):
            query_type = 'exclusion'
        elif any(word in query_lower for word in ['wait', 'waiting period']):
            query_type = 'waiting_period'
        elif any(word in query_lower for word in ['claim', 'claiming']):
            query_type = 'claim'
        elif any(word in query_lower for word in ['premium', 'payment']):
            query_type = 'premium'
        else:
            query_type = 'general'
        
        # Extract medical procedures (basic patterns)
        medical_patterns = [
            r'\b(surgery|operation|procedure|treatment)\b',
            r'\b(knee|hip|heart|brain|liver|kidney)\s+(surgery|operation|procedure)',
            r'\b(cataract|bypass|transplant|dialysis)\b'
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities["medical_procedures"].extend(matches)
        
        # Extract time periods
        time_patterns = [
            r'\b(\d+)\s+(day|week|month|year)s?\b',
            r'\b(thirty|forty|fifty|sixty)\s+(day|month)s?\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            entities["time_periods"].extend([' '.join(match) for match in matches])
        
        return {
            'intent': f"User wants to know about {query_type}",
            'entities': entities,
            'query_type': query_type,
            'confidence': 0.6
        }
    
    async def generate_answer(
        self,
        question: str,
        parsed_query: ParsedQuery,
        relevant_chunks: List[RelevantChunk],
        document_metadata: Dict[str, Any]
    ) -> AnswerResponse:
        """
        Generate comprehensive answer using LLM
        
        Args:
            question: Original question
            parsed_query: Parsed query information
            relevant_chunks: Relevant document chunks
            document_metadata: Document metadata
            
        Returns:
            AnswerResponse: Complete answer with reasoning
        """
        try:
            logger.debug(f"Generating answer for: {question[:50]}...")
            
            start_time = time.time()
            
            # Build context from relevant chunks
            context = self._build_context(relevant_chunks)
            
            # Build answer generation prompt
            prompt = self._build_answer_prompt(question, parsed_query, context, document_metadata)
            
            # Generate answer
            response = await self._make_llm_request(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse structured response
            answer_data = self._parse_answer_response(response.content)
            
            # Build reasoning
            reasoning = DecisionReasoning(
                reasoning_steps=answer_data.get('reasoning_steps', []),
                evidence_chunks=[chunk.chunk_id for chunk in relevant_chunks],
                confidence_factors=answer_data.get('confidence_factors', {}),
                contradictions=answer_data.get('contradictions'),
                assumptions=answer_data.get('assumptions')
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create answer response
            answer_response = AnswerResponse(
                query_id=f"query_{int(time.time() * 1000)}",
                question=question,
                answer_text=answer_data.get('answer', 'Unable to provide a definitive answer based on the available information.'),
                confidence_score=answer_data.get('confidence_score', 0.5),
                parsed_query=parsed_query,
                relevant_chunks=relevant_chunks,
                reasoning=reasoning,
                processing_time_ms=processing_time
            )
            
            # Log query
            await self._log_query(answer_response)
            
            return answer_response
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise LLMError(f"Failed to generate answer: {str(e)}")
    
    def _build_context(self, chunks: List[RelevantChunk]) -> str:
        """Build context string from relevant chunks"""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Context {i+1}] (Similarity: {chunk.similarity_score:.3f})")
            context_parts.append(chunk.content.strip())
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_answer_prompt(
        self,
        question: str,
        parsed_query: ParsedQuery,
        context: str,
        document_metadata: Dict[str, Any]
    ) -> str:
        """Build comprehensive answer generation prompt"""
        return f"""
You are an expert insurance policy analyst. Analyze the following question using the provided document context and generate a comprehensive, accurate answer.

QUESTION: {question}

PARSED QUERY INFORMATION:
- Intent: {parsed_query.intent}
- Query Type: {parsed_query.query_type}
- Key Entities: {json.dumps(parsed_query.entities, indent=2)}

DOCUMENT CONTEXT:
{context}

DOCUMENT METADATA:
- Source: {document_metadata.get('source', 'Unknown')}
- Type: {document_metadata.get('document_type', 'Unknown')}

INSTRUCTIONS:
1. Provide a direct, clear answer to the question
2. Base your answer strictly on the provided context
3. If information is not available, clearly state this
4. Include specific details like time periods, amounts, conditions when mentioned
5. Explain any conditions or limitations that apply
6. Cite relevant sections from the context

Please provide your response in the following JSON format:
{{
    "answer": "Clear, comprehensive answer to the question",
    "confidence_score": 0.95,
    "reasoning_steps": [
        "Step 1: Identified relevant policy sections",
        "Step 2: Found specific coverage information",
        "Step 3: Noted any conditions or limitations"
    ],
    "confidence_factors": {{
        "information_availability": 0.9,
        "context_relevance": 0.95,
        "answer_specificity": 0.9
    }},
    "contradictions": ["Any contradictory information found"],
    "assumptions": ["Any assumptions made in the answer"],
    "key_quotes": ["Direct quotes from the policy that support the answer"]
}}

IMPORTANT: 
- Only make claims supported by the provided context
- If the answer cannot be determined from the context, say so clearly
- Be precise about waiting periods, coverage limits, and conditions
- Use professional, clear language appropriate for insurance documentation
"""
    
    def _parse_answer_response(self, response_content: str) -> Dict[str, Any]:
        """Parse structured answer response"""
        try:
            # Try to parse as JSON
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Fallback: extract answer from text
            logger.warning("Failed to parse JSON response, using fallback")
            
            # Simple fallback extraction
            lines = response_content.strip().split('\n')
            answer_text = response_content.strip()
            
            # Try to find answer section
            for i, line in enumerate(lines):
                if 'answer' in line.lower() and ':' in line:
                    answer_text = line.split(':', 1)[1].strip()
                    break
            
            return {
                'answer': answer_text,
                'confidence_score': 0.5,
                'reasoning_steps': ['Response parsed using fallback method'],
                'confidence_factors': {'fallback_parsing': 0.5},
                'contradictions': None,
                'assumptions': ['Answer extracted using text parsing fallback']
            }
    
    async def _make_llm_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> LLMResponse:
        """Make request to LLM API"""
        try:
            # Rate limiting
            await self._rate_limit()
            
            start_time = time.time()
            
            # Prepare request parameters
            request_params = {
                'model': model or self.model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert insurance policy analyst with deep knowledge of policy terms, coverage conditions, and legal requirements.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': max_tokens or self.max_tokens,
                'temperature': temperature or self.temperature
            }
            
            # Make API call
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **request_params
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model_used=response.model,
                tokens_used=response.usage.total_tokens,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise LLMError(f"LLM API error: {str(e)}")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _log_query(self, answer_response: AnswerResponse):
        """Log query and response for analytics"""
        try:
            await DatabaseManager.log_query(
                query_text=answer_response.question,
                parsed_query=answer_response.parsed_query.dict(),
                answer_text=answer_response.answer_text,
                confidence_score=answer_response.confidence_score,
                processing_time_ms=answer_response.processing_time_ms,
                relevant_chunks=[chunk.dict() for chunk in answer_response.relevant_chunks],
                reasoning=answer_response.reasoning.dict()
            )
        except Exception as e:
            logger.warning(f"Failed to log query: {str(e)}")
    
    async def summarize_document(self, content: str, max_length: int = 500) -> str:
        """Generate document summary"""
        try:
            prompt = f"""
Summarize the following document content in {max_length} words or less. Focus on key information, coverage details, and important conditions.

Document Content:
{content[:8000]}  # Limit content for token efficiency

Provide a clear, comprehensive summary that captures the essential information.
"""
            
            response = await self._make_llm_request(
                prompt=prompt,
                max_tokens=max_length,
                temperature=0.1
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Document summarization failed: {str(e)}")
            return "Unable to generate document summary."
    
    async def extract_key_terms(self, content: str, max_terms: int = 20) -> List[str]:
        """Extract key terms from document"""
        try:
            prompt = f"""
Extract the {max_terms} most important terms and concepts from the following document. Focus on insurance-specific terms, coverage types, conditions, and key phrases.

Document Content:
{content[:8000]}

Return the terms as a simple list, one per line.
"""
            
            response = await self._make_llm_request(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse terms from response
            terms = []
            for line in response.content.strip().split('\n'):
                term = line.strip()
                if term and not term.startswith('-') and not term.startswith('*'):
                    terms.append(term)
            
            return terms[:max_terms]
            
        except Exception as e:
            logger.error(f"Key term extraction failed: {str(e)}")
            return []
    
    async def detect_document_type(self, content: str) -> str:
        """Detect document type using LLM"""
        try:
            prompt = f"""
Analyze the following document content and determine its type. Common types include:
- insurance_policy
- legal_contract
- hr_document
- compliance_document
- terms_of_service
- privacy_policy
- user_manual
- other

Document Content (first 2000 characters):
{content[:2000]}

Return only the document type as a single phrase.
"""
            
            response = await self._make_llm_request(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            return response.content.strip().lower().replace(' ', '_')
            
        except Exception as e:
            logger.error(f"Document type detection failed: {str(e)}")
            return "unknown"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM service"""
        health_status = {
            'status': 'healthy',
            'client_available': bool(self.client),
            'model': self.model
        }
        
        # Test LLM functionality
        try:
            test_response = await self._make_llm_request(
                prompt="Test prompt. Respond with 'OK'.",
                max_tokens=10,
                temperature=0.0
            )
            health_status['test_request_success'] = True
            health_status['test_response'] = test_response.content.strip()
        except Exception as e:
            health_status['test_request_success'] = False
            health_status['test_error'] = str(e)
        
        return health_status