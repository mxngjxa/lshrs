"""
Utility helpers exposed for convenient imports.
"""

from .br import (
    PRECOMPUTED_CONFIGS,
    compute_collision_probability,
    compute_false_rates,
    compute_lsh_threshold,
    find_optimal_br,
    get_optimal_config,
    print_config_analysis,
)
from .norm import l2_norm
from .similarity import cosine_similarity, top_k_cosine

__all__ = [
    "PRECOMPUTED_CONFIGS",
    "compute_collision_probability",
    "compute_false_rates",
    "compute_lsh_threshold",
    "find_optimal_br",
    "get_optimal_config",
    "print_config_analysis",
    "l2_norm",
    "cosine_similarity",
    "top_k_cosine",
]
