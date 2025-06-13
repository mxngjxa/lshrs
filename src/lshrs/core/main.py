"""
Main orchestration module for the LSH recommendation system.
"""

import logging
from typing import Any, List, Optional, Tuple

from lshrs.core.config import RecommenderConfig
from lshrs.core.exceptions import (
    ModelNotTrainedError,
    RecommendationError,
    RecommenderError,
)
from lshrs.core.interfaces import (
    BaseEncoder,
    BaseHasher,
    BaseRecommender,
    BaseSimilarity,
)


class RecommendationPipeline:
    """Main pipeline orchestrating the recommendation process."""

    def __init__(self, config: RecommenderConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Components will be initialized during fit
        self.encoder: Optional[BaseEncoder] = None
        self.hasher: Optional[BaseHasher] = None
        self.similarity: Optional[BaseSimilarity] = None

        # Data storage
        self.item_data = None
        self.encoded_data = None
        self.signatures = None
        self.item_index = None

        self.is_fitted = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _validate_components(self):
        """Validate that all required components are available."""
        if not all([self.encoder, self.hasher, self.similarity]):
            raise RecommenderError("Pipeline components not properly initialized")

    def set_components(
        self,
        encoder: BaseEncoder,
        hasher: BaseHasher,
        similarity: BaseSimilarity
    ):
        """Set the pipeline components."""
        self.encoder = encoder
        self.hasher = hasher
        self.similarity = similarity
        self.logger.info("Pipeline components set successfully")

    def fit(self, data: Any, item_ids: Optional[List] = None):
        """Fit the entire pipeline to the data."""
        try:
            self._validate_components()
            self.logger.info("Starting pipeline fitting process")

            # Store original data and create index
            self.item_data = data
            self.item_index = item_ids or list(range(len(data)))

            # Encode data
            self.logger.info("Encoding data...")
            self.encoded_data = self.encoder.fit_transform(data)

            # Generate LSH signatures
            self.logger.info("Generating LSH signatures...")
            self.signatures = self.hasher.fit(self.encoded_data).hash(self.encoded_data)

            self.is_fitted = True
            self.logger.info("Pipeline fitting completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline fitting failed: {str(e)}")
            raise RecommenderError(f"Pipeline fitting failed: {str(e)}")

    def _check_fitted(self):
        """Check if the pipeline is fitted."""
        if not self.is_fitted:
            raise ModelNotTrainedError(
                "Pipeline must be fitted before making recommendations"
            )

    def recommend(
        self,
        query: Any,
        top_k: int = None,
        similarity_threshold: float = None
        ) -> List[Tuple[Any, float]]:

        """Generate recommendations for a query."""
        self._check_fitted()

        top_k = top_k or self.config.top_k
        similarity_threshold = similarity_threshold or self.config.similarity_threshold

        try:
            # Encode query
            query_encoded = self.encoder.transform([query])
            query_signature = self.hasher.hash(query_encoded)

            # Find candidates using LSH
            candidates = self.hasher.find_candidates(query_signature[0])

            if not candidates:
                self.logger.warning("No candidates found for query")
                return []

            # Compute similarities for candidates
            candidate_data = self.encoded_data[candidates]
            similarities = self.similarity.compute_batch_similarity(
                query_encoded[0], candidate_data
            )

            # Filter by threshold and sort
            valid_candidates = [
                (self.item_index[candidates[i]], sim)
                for i, sim in enumerate(similarities)
                if sim >= similarity_threshold
            ]

            # Sort by similarity and return top-k
            valid_candidates.sort(key=lambda x: x[1], reverse=True)

            return valid_candidates[:top_k]

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            raise RecommendationError(f"Recommendation generation failed: {str(e)}")


class LSHRecommender(BaseRecommender):
    """Main LSH-based recommender implementation."""

    def __init__(self, config: RecommenderConfig = None):
        self.config = config or RecommenderConfig()
        self.pipeline = RecommendationPipeline(self.config)
        self.logger = logging.getLogger(__name__)

    def fit(
        self, data: Any, item_ids: Optional[List] = None, **kwargs
    ) -> "LSHRecommender":
        """Fit the recommender to the data."""
        # Initialize components based on configuration
        from ..encoding import get_encoder
        from ..hashing import get_hasher
        from ..recommendation.similarity import get_similarity

        encoder = get_encoder(self.config.encoding_config)
        hasher = get_hasher(self.config.lsh_config)
        similarity = get_similarity()

        self.pipeline.set_components(encoder, hasher, similarity)
        self.pipeline.fit(data, item_ids)

        return self

    def recommend(
        self,
        query: Any,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[Any, float]]:
        """Generate recommendations for a query."""
        return self.pipeline.recommend(query, top_k)

    def get_similar_items(
        self,
        item_id: Any,
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Find items similar to a given item."""
        # Implementation for item-to-item similarity
        pass

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        # Implementation for model persistence
        pass

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        # Implementation for model loading
        pass
