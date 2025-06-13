"""
Configuration management for the LSH recommendation system.
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class EncodingType(str, Enum):
    TFIDF = "tfidf"
    EMBEDDING = "embedding"
    ONEHOT = "onehot"


class LSHType(str, Enum):
    MINHASH = "minhash"
    HYPERPLANE = "hyperplane"
    MULTIPROBE = "multiprobe"


class LSHConfig(BaseModel):
    """Configuration for LSH parameters."""

    hash_type: LSHType = LSHType.MINHASH
    num_bands: int = Field(default=20, ge=1, le=200)
    rows_per_band: int = Field(default=5, ge=1, le=50)
    num_hash_functions: int = Field(default=100, ge=10, le=1000)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @validator('num_bands', 'rows_per_band')
    def validate_lsh_params(cls, v, values):
        """Ensure LSH parameters are reasonable."""
        if 'num_bands' in values and 'rows_per_band' in values:
            total_hashes = values['num_bands'] * values['rows_per_band']
            if total_hashes > 1000:
                raise ValueError(
                    "Total hash functions (bands * rows) should not exceed 1000"
                )
        return v


class EncodingConfig(BaseModel):
    """Configuration for encoding methods."""

    encoding_type: EncodingType = EncodingType.TFIDF
    max_features: Optional[int] = Field(default=10000, ge=100)
    min_df: int = Field(default=2, ge=1)
    max_df: float = Field(default=0.95, ge=0.1, le=1.0)
    ngram_range: tuple = Field(default=(1, 2))
    embedding_dim: int = Field(default=128, ge=50, le=1024)

    class Config:
        arbitrary_types_allowed = True


class RecommenderConfig(BaseModel):
    """Main configuration for the recommendation system."""

    lsh_config: LSHConfig = Field(default_factory=LSHConfig)
    encoding_config: EncodingConfig = Field(default_factory=EncodingConfig)

    # Recommendation parameters
    top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

    # Performance parameters
    batch_size: int = Field(default=1000, ge=100)
    n_jobs: int = Field(default=-1)
    random_state: Optional[int] = Field(default=42)

    # Data paths
    data_path: Optional[str] = None
    model_save_path: Optional[str] = "./models/"

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RecommenderConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: str) -> "RecommenderConfig":
        """Load configuration from JSON/YAML file."""
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
