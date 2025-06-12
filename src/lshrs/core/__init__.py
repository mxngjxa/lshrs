"""
Core module for LSH-based recommendation system.
Provides configuration, interfaces, and main orchestration components.
"""

from .config import EncodingConfig, LSHConfig, RecommenderConfig
from .exceptions import (
    ConfigurationError,
    DataProcessingError,
    LSHError,
    RecommenderError,
)
from .interfaces import BaseEncoder, BaseHasher, BaseRecommender, BaseSimilarity
from .main import LSHRecommender, RecommendationPipeline

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
