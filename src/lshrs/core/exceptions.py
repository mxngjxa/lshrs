# defines common exceptions for the lshrs package

"""
Custom exceptions for the LSH recommendation system.
"""


class RecommenderError(Exception):
    """Base exception for all recommender system errors."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(RecommenderError):
    """Raised when there are configuration-related errors."""
    pass


class DataProcessingError(RecommenderError):
    """Raised when data processing fails."""
    pass


class LSHError(RecommenderError):
    """Raised when LSH operations fail."""
    pass


class EncodingError(RecommenderError):
    """Raised when encoding operations fail."""
    pass


class ModelNotTrainedError(RecommenderError):
    """Raised when attempting to use an untrained model."""
    pass


class InvalidInputError(RecommenderError):
    """Raised when input data is invalid."""
    pass


class RecommendationError(RecommenderError):
    """Raised when recommendation generation fails."""
    pass


class PersistenceError(RecommenderError):
    """Raised when model saving/loading fails."""
    pass

class DataLoaderError(RecommenderError):
    """Raised when dataloader operations fail."""
    pass

class HashingError(RecommenderError):
    """Raised when hashing operations fail."""
    pass

class SimilarityError(RecommenderError):
    """Raised when similarity calculations fail."""
    pass
