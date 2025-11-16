"""
LSH Configuration Module

Precomputed optimal (b, r) configurations for common hash sizes and similarity thresholds.
Based on LSH theory where threshold ≈ (1/b)^(1/r) and probability P = 1 - (1 - s^r)^b
"""

import numpy as np
from scipy.integrate import quad
from typing import Dict, Tuple, Optional


# Pre-computed optimal configurations for common hash sizes
# Format: hash_size -> {threshold: (b, r)}
# where b = number of bands, r = rows per band
PRECOMPUTED_CONFIGS = {
    4096: {  # 2^12
        0.5: (512, 8),   # Actual threshold: 0.459, FP: 7.1%, FN: 0.3%
        0.7: (256, 16),  # Actual threshold: 0.707, FP: 2.9%, FN: 1.3%
        0.85: (128, 32), # Actual threshold: 0.859, FP: 1.5%, FN: 1.0%
        0.9: (64, 64),   # Actual threshold: 0.937, FP: 0.1%, FN: 3.0%
        0.95: (32, 128), # Actual threshold: 0.973, FP: 0.03%, FN: 1.9%
    },
    8192: {  # 2^13
        0.4: (1024, 8),  # Actual threshold: 0.420, FP: 2.5%, FN: 2.1%
        0.7: (512, 16),  # Actual threshold: 0.677, FP: 4.8%, FN: 0.3%
        0.8: (256, 32),  # Actual threshold: 0.841, FP: 0.5%, FN: 3.1%
        0.85: (256, 32), # Actual threshold: 0.841, FP: 2.7%, FN: 0.3%
        0.9: (128, 64),  # Actual threshold: 0.927, FP: 0.2%, FN: 2.1%
        0.95: (64, 128), # Actual threshold: 0.968, FP: 0.06%, FN: 1.4%
    },
    16384: {  # 2^14
        0.4: (2048, 8),  # Actual threshold: 0.386, FP: 4.4%, FN: 0.7%
        0.6: (1024, 16), # Actual threshold: 0.648, FP: 1.0%, FN: 3.7%
        0.8: (512, 32),  # Actual threshold: 0.823, FP: 0.9%, FN: 1.8%
        0.85: (512, 32), # Actual threshold: 0.823, FP: 4.2%, FN: 0.04%
        0.9: (256, 64),  # Actual threshold: 0.917, FP: 0.4%, FN: 1.3%
        0.95: (128, 128),# Actual threshold: 0.963, FP: 0.1%, FN: 1.0%
    },
    32768: {  # 2^15 - Production scale
        0.4: (4096, 8),   # Fuzzy matching - Actual: 0.354, FP: 6.8%, FN: 0.1%
        0.6: (2048, 16),  # High recall - Actual: 0.621, FP: 1.8%, FN: 1.9%
        0.8: (1024, 32),  # Standard dedup - Actual: 0.805, FP: 1.6%, FN: 0.8%
        0.85: (1024, 32), # High precision - Actual: 0.805, FP: 5.9%, FN: 0.0%
        0.9: (512, 64),   # Very high precision - Actual: 0.907, FP: 0.7%, FN: 0.6%
        0.95: (256, 128), # Near-exact - Actual: 0.958, FP: 0.2%, FN: 0.6%
    },
    65536: {  # 2^16 - Large scale
        0.3: (8192, 8),   # Very fuzzy - Actual: 0.324, FP: 1.6%, FN: 2.1%
        0.6: (4096, 16),  # Moderate recall - Actual: 0.595, FP: 3.1%, FN: 0.7%
        0.8: (2048, 32),  # Balanced - Actual: 0.788, FP: 2.8%, FN: 0.2%
        0.85: (1024, 64), # High precision - Actual: 0.897, FP: 0.04%, FN: 4.0%
        0.9: (1024, 64),  # Very high precision - Actual: 0.897, FP: 1.3%, FN: 0.2%
        0.95: (512, 128), # Near-exact - Actual: 0.952, FP: 0.5%, FN: 0.3%
    },
}


def compute_lsh_threshold(b: int, r: int) -> float:
    """
    Compute the approximate similarity threshold for an LSH configuration.

    The threshold is the Jaccard/cosine similarity where we have approximately
    50% probability of the item being detected as similar.

    Args:
        b: Number of bands
        r: Number of rows (hash functions) per band

    Returns:
        Approximate similarity threshold
    """
    return (1/b) ** (1/r)


def compute_collision_probability(similarity: float, b: int, r: int) -> float:
    """
    Compute the probability that two items will be detected as similar.

    Args:
        similarity: True similarity between items (0 to 1)
        b: Number of bands
        r: Number of rows per band

    Returns:
        Probability of collision (detection)
    """
    return 1 - (1 - similarity ** r) ** b


def compute_false_rates(b: int, r: int, threshold: float) -> Tuple[float, float]:
    """
    Compute false positive and false negative rates for given configuration.

    Args:
        b: Number of bands
        r: Number of rows per band
        threshold: Similarity threshold for classification

    Returns:
        (false_positive_rate, false_negative_rate)
    """
    # False positive: probability of detecting items with similarity < threshold
    def integrand_fp(s):
        return 1 - (1 - s**r)**b

    # False negative: probability of missing items with similarity >= threshold
    def integrand_fn(s):
        return (1 - s**r)**b

    fp_rate, _ = quad(integrand_fp, 0, threshold, limit=100)
    fn_rate, _ = quad(integrand_fn, threshold, 1, limit=100)

    return fp_rate, fn_rate


def find_optimal_br(num_perm: int, target_threshold: float, 
                   tolerance: float = 0.05) -> Optional[Tuple[int, int]]:
    """
    Find optimal b and r values for a given number of permutations and target threshold.

    This function searches through all valid factorizations of num_perm to find
    the configuration that minimizes the sum of false positive and false negative rates.

    Args:
        num_perm: Total number of hash functions (b * r must equal this)
        target_threshold: Desired similarity threshold (0 to 1)
        tolerance: Maximum acceptable deviation from target threshold

    Returns:
        (b, r) tuple that minimizes error rates, or None if no valid config found
    """
    best_config = None
    best_score = float('inf')

    # Try all valid factorizations
    for r in range(1, int(np.sqrt(num_perm)) + 1):
        if num_perm % r != 0:
            continue

        b = num_perm // r

        # Compute actual threshold for this configuration
        actual_threshold = compute_lsh_threshold(b, r)

        # Skip if threshold is too far from target
        if abs(actual_threshold - target_threshold) > tolerance:
            continue

        # Compute error rates
        fp_rate, fn_rate = compute_false_rates(b, r, target_threshold)

        # Combined score (equal weight to FP and FN)
        score = fp_rate + fn_rate

        if score < best_score:
            best_score = score
            best_config = (b, r)

    # Also try the reverse factorization for completeness
    for b in range(1, int(np.sqrt(num_perm)) + 1):
        if num_perm % b != 0:
            continue

        r = num_perm // b

        actual_threshold = compute_lsh_threshold(b, r)

        if abs(actual_threshold - target_threshold) > tolerance:
            continue

        fp_rate, fn_rate = compute_false_rates(b, r, target_threshold)
        score = fp_rate + fn_rate

        if score < best_score:
            best_score = score
            best_config = (b, r)

    return best_config


def get_optimal_config(num_perm: int, target_threshold: float = 0.5) -> Tuple[int, int]:
    """
    Get optimal LSH configuration for given parameters.

    First checks precomputed configurations, then falls back to computing
    optimal values if not found.

    Args:
        num_perm: Total number of hash functions
        target_threshold: Target similarity threshold (default 0.5)

    Returns:
        (b, r) tuple for number of bands and rows per band
    """
    # Check precomputed configurations
    if num_perm in PRECOMPUTED_CONFIGS:
        # Find closest threshold
        thresholds = list(PRECOMPUTED_CONFIGS[num_perm].keys())
        closest_threshold = min(thresholds, key=lambda x: abs(x - target_threshold))

        if abs(closest_threshold - target_threshold) <= 0.05:
            return PRECOMPUTED_CONFIGS[num_perm][closest_threshold]

    # Compute optimal configuration
    config = find_optimal_br(num_perm, target_threshold)

    if config:
        return config

    # Fallback to square root heuristic
    b = int(np.sqrt(num_perm))
    r = num_perm // b

    # Ensure b * r = num_perm
    while b * r != num_perm:
        b -= 1
        if num_perm % b == 0:
            r = num_perm // b

    return b, r


def print_config_analysis(num_perm: int, threshold: float = 0.5):
    """
    Print detailed analysis of LSH configuration.

    Args:
        num_perm: Number of hash functions
        threshold: Target similarity threshold
    """
    b, r = get_optimal_config(num_perm, threshold)
    actual_threshold = compute_lsh_threshold(b, r)
    fp_rate, fn_rate = compute_false_rates(b, r, threshold)

    print(f"LSH Configuration Analysis")
    print(f"{'='*50}")
    print(f"Number of permutations: {num_perm}")
    print(f"Target threshold: {threshold:.2f}")
    print(f"\nOptimal configuration:")
    print(f"  Bands (b): {b}")
    print(f"  Rows per band (r): {r}")
    print(f"\nPerformance metrics:")
    print(f"  Actual threshold: {actual_threshold:.4f}")
    print(f"  False positive rate: {fp_rate:.2%}")
    print(f"  False negative rate: {fn_rate:.2%}")
    print(f"  S-curve steepness: {b * r}")

    # Show probability curve at key points
    print(f"\nDetection probabilities:")
    for sim in [0.3, 0.5, 0.7, 0.9]:
        prob = compute_collision_probability(sim, b, r)
        print(f"  Similarity {sim:.1f}: {prob:.2%} chance of detection")


if __name__ == "__main__":
    # Example usage
    print("Example configurations for common hash sizes:\n")

    for size in [2**12, 2**13, 2**14, 2**15, 2**16]:
        print(f"\nHash size: {size}")
        for threshold in [0.5, 0.8, 0.9]:
            b, r = get_optimal_config(size, threshold)
            actual = compute_lsh_threshold(b, r)
            print(f"  Threshold {threshold:.1f}: b={b:4d}, r={r:3d} (actual: {actual:.3f})")