# app/utils/exceptions.py - Custom Exception Classes
class BaseCustomException(Exception):
    """Base exception class for custom exceptions"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

class DocumentProcessingError(BaseCustomException):
    """Raised when document processing fails"""
    pass

class EmbeddingError(BaseCustomException):
    """Raised when embedding generation fails"""
    pass

class VectorSearchError(BaseCustomException):
    """Raised when vector search operations fail"""
    pass

class LLMError(BaseCustomException):
    """Raised when LLM operations fail"""
    pass

class DatabaseError(BaseCustomException):
    """Raised when database operations fail"""
    pass

class AuthenticationError(BaseCustomException):
    """Raised when authentication fails"""
    pass

class ValidationError(BaseCustomException):
    """Raised when input validation fails"""
    pass

class ConfigurationError(BaseCustomException):
    """Raised when configuration is invalid"""
    pass

class RateLimitError(BaseCustomException):
    """Raised when rate limits are exceeded"""
    pass

class ExternalServiceError(BaseCustomException):
    """Raised when external service calls fail"""
    pass