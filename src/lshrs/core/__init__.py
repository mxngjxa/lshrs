"""
Core module for LSH-based recommendation system.
Provides configuration, interfaces, and main orchestration components.
"""

from .config import (
    RecommenderConfig, 
    LSHConfig, 
    EncodingConfig
)

from .exceptions import (
    RecommenderError,
    ConfigurationError,
    DataProcessingError,
    LSHError
)

from .interfaces import (
    BaseEncoder,
    BaseHasher,
    BaseRecommender,
    BaseSimilarity
)

from .main import RecommendationPipeline, LSHRecommender

__version__ = "0.0.1"
__author__ = "Y. Zhao, M. Guan"

__all__ = [
    "RecommenderConfig",
    "LSHConfig", 
    "EncodingConfig",
    "RecommenderError",
    "ConfigurationError",
    "DataProcessingError",
    "LSHError",
    "BaseEncoder",
    "BaseHasher", 
    "BaseRecommender",
    "BaseSimilarity",
    "RecommendationPipeline",
    "LSHRecommender"
]
