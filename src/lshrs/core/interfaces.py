"""
Abstract interfaces for the LSH recommendation system components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix


class BaseEncoder(ABC):
    """Abstract base class for all encoders."""

    @abstractmethod
    def fit(self, data: Any) -> "BaseEncoder":
        """Fit the encoder to the data."""
        pass

    @abstractmethod
    def transform(self, data: Any) -> Union[np.ndarray, csr_matrix]:
        """Transform data using the fitted encoder."""
        pass

    @abstractmethod
    def fit_transform(self, data: Any) -> Union[np.ndarray, csr_matrix]:
        """Fit and transform data in one step."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get feature names if applicable."""
        pass


class BaseHasher(ABC):
    """Abstract base class for LSH hashers."""

    @abstractmethod
    def fit(self, data: Union[np.ndarray, csr_matrix]) -> "BaseHasher":
        """Fit the hasher to the data."""
        pass

    @abstractmethod
    def hash(self, data: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """Generate hash signatures for the data."""
        pass

    @abstractmethod
    def get_buckets(self, signatures: np.ndarray) -> Dict[str, List[int]]:
        """Group signatures into buckets."""
        pass

    @abstractmethod
    def find_candidates(self, query_signature: np.ndarray) -> List[int]:
        """Find candidate items for a query signature."""
        pass


class BaseSimilarity(ABC):
    """Abstract base class for similarity computation."""

    @abstractmethod
    def compute_similarity(
        self,
        item1: Union[np.ndarray, csr_matrix],
        item2: Union[np.ndarray, csr_matrix]
    ) -> float:
        """Compute similarity between two items."""
        pass

    @abstractmethod
    def compute_batch_similarity(
        self,
        query: Union[np.ndarray, csr_matrix],
        candidates: Union[np.ndarray, csr_matrix]
    ) -> np.ndarray:
        """Compute similarity between query and multiple candidates."""
        pass


class BaseRecommender(ABC):
    """Abstract base class for recommenders."""

    @abstractmethod
    def fit(self, data: Any, **kwargs) -> "BaseRecommender":
        """Train the recommender."""
        pass

    @abstractmethod
    def recommend(
        self,
        query: Any,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate recommendations."""
        pass

    @abstractmethod
    def get_similar_items(
        self,
        item_id: Any,
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Find similar items."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        pass


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self, source: str) -> Any:
        """Load data from source."""
        pass

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess loaded data."""
        pass
