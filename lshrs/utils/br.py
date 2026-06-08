"""
LSH Configuration Module - Optimal Band/Row Selection for Sign-Random-Projection LSH

This module provides mathematical optimization for LSH parameters based on the
SimHash / sign-random-projection collision model for COSINE similarity. It helps
select the optimal number of bands (b) and rows per band (r) to achieve a desired
similarity threshold with minimal error rates.

Theory Background:
    This library uses SIGN RANDOM PROJECTION LSH (SimHash) for cosine similarity.
    Each hash bit is the sign of (random_hyperplane . vector). For two vectors with
    cosine similarity ``sim``, the probability that a single bit collides is:

        p(sim) = 1 - arccos(clip(sim, -1, 1)) / pi

    Hash signatures are split into b bands of r bits each. Two items collide in a
    band iff all r bits of that band match (probability p^r). They are detected as
    similar if they collide in AT LEAST ONE of the b bands, giving the S-curve:

        P(sim, b, r) = 1 - (1 - p(sim)^r)^b

    Key relationships:
        - Threshold: cos(pi * (1 - p_star)) where p_star = (1 - 0.5**(1/b))**(1/r)
        - More bands (b↑): Higher recall, more false positives
        - More rows (r↑): Higher precision, more false negatives
        - Total bits = b × r must stay constant

Typical usage:
    >>> b, r = get_optimal_config(num_perm=128, target_threshold=0.8)
    >>> # Use b bands with r rows each in your LSH implementation
"""

import warnings

import numpy as np
from scipy.integrate import quad


def _bit_collision_probability(sim: float) -> float:
    """
    Per-bit collision probability for sign-random-projection (SimHash) LSH.

    For two vectors with cosine similarity ``sim`` and a random hyperplane, the
    probability that both vectors fall on the same side of the hyperplane (i.e. the
    sign bit matches) is:

        p(sim) = 1 - arccos(clip(sim, -1, 1)) / pi

    Args:
        sim: Cosine similarity, clipped internally to [-1, 1].

    Returns:
        Per-bit collision probability in [0, 1].
    """
    sim_clipped = np.clip(sim, -1.0, 1.0)
    return 1.0 - np.arccos(sim_clipped) / np.pi


def compute_lsh_threshold(b: int, r: int) -> float:
    """
    Compute the cosine similarity threshold for an LSH configuration.

    The threshold is the inflection point of the S-curve where the detection
    probability P(sim, b, r) equals exactly 0.5.

    Mathematical basis (cosine / SimHash model):
        S-curve: P(sim) = 1 - (1 - p(sim)^r)^b, with p(sim) = 1 - arccos(sim)/pi
        Solving P == 0.5 for the per-bit probability gives a closed form:
            p_star = (1 - 0.5**(1/b))**(1/r)          (always in (0, 1))
        Inverting p(sim) = p_star (i.e. sim = cos(pi * (1 - p))):
            threshold = cos(pi * (1 - p_star))

    Args:
        b: Number of bands. Range: [1, num_perm].
        r: Number of rows (bits) per band. Range: [1, num_perm].

    Returns:
        Cosine similarity threshold in (-1, 1) where P == 0.5.

    Notes:
        - Larger b → lower threshold (more permissive)
        - Larger r → higher threshold (more strict)
        - The product b×r determines total hash bits used
    """
    p_star = (1 - 0.5 ** (1 / b)) ** (1 / r)
    return float(np.cos(np.pi * (1 - p_star)))


def compute_collision_probability(similarity: float, b: int, r: int) -> float:
    """
    Probability that two items are detected as similar (collide) under SimHash LSH.

    This is the fundamental LSH S-curve for the cosine / sign-random-projection
    model: low probability for dissimilar items, high probability for similar
    items, with a sharp transition near the threshold.

    Mathematical formula:
        p = 1 - arccos(clip(similarity, -1, 1)) / pi   (per-bit collision prob)
        P(collision) = 1 - (1 - p^r)^b

        Where:
        - p^r = probability of matching all r bits in a band
        - (1 - p^r) = probability of NOT matching a band
        - (1 - p^r)^b = probability of NOT matching ANY band
        - 1 - (1 - p^r)^b = probability of matching AT LEAST ONE band

    Args:
        similarity: True cosine similarity between items, range [-1, 1].
        b: Number of bands in the LSH scheme.
        r: Number of rows (bits) per band.

    Returns:
        Probability of collision (detection) in range [0, 1].

    S-curve characteristics:
        - Sigmoid shape with steepness proportional to b×r
        - Inflection point (P == 0.5) at the cosine threshold from
          compute_lsh_threshold(b, r)
        - Near 0 for sim << threshold, near 1 for sim >> threshold
    """
    p = _bit_collision_probability(similarity)
    return float(1 - (1 - p**r) ** b)


def compute_false_rates(b: int, r: int, threshold: float) -> tuple[float, float]:
    """
    Compute false positive and false negative rates for an LSH configuration.

    Uses numerical integration of the corrected cosine S-curve P(sim, b, r) to
    compute expected error rates on each side of the threshold.

    Definitions:
        - False Positive (FP): Items with sim < threshold detected as similar.
        - False Negative (FN): Items with sim >= threshold NOT detected.

    Domain:
        The integration assumes a uniform distribution of NON-NEGATIVE cosine
        similarities over [0, 1]. (Negative cosine similarities are not modelled
        here.) Each rate is normalized by the width of its integration interval so
        both are returned as rates in [0, 1].

    Args:
        b: Number of bands.
        r: Number of rows (bits) per band.
        threshold: Cosine similarity cutoff for classification in [0, 1].

    Returns:
        Tuple of (false_positive_rate, false_negative_rate), both in [0, 1].

    Mathematical approach:
        FP rate = ∫[0, threshold] P(sim) dsim / threshold
        FN rate = ∫[threshold, 1] (1 - P(sim)) dsim / (1 - threshold)

        Where P(sim) = 1 - (1 - p(sim)^r)^b and p(sim) = 1 - arccos(sim)/pi.

    Notes:
        - Zero-width intervals (threshold == 0 or threshold == 1) are guarded:
          the corresponding rate is 0.0 (no probability mass on that side).
        - High b reduces FN but increases FP; high r reduces FP but increases FN.
        - Optimal config minimizes FP + FN.
    """

    # Integrand for false positives: probability of detecting items below threshold.
    def integrand_fp(s):
        p = 1.0 - np.arccos(np.clip(s, -1.0, 1.0)) / np.pi
        return 1 - (1 - p**r) ** b

    # Integrand for false negatives: probability of NOT detecting items above threshold.
    def integrand_fn(s):
        p = 1.0 - np.arccos(np.clip(s, -1.0, 1.0)) / np.pi
        return (1 - p**r) ** b

    # Guard against zero-width intervals to avoid dividing by zero.
    if threshold <= 0.0:
        fp_rate = 0.0
    else:
        fp_integral, _ = quad(integrand_fp, 0, threshold, limit=100)
        fp_rate = fp_integral / threshold

    if threshold >= 1.0:
        fn_rate = 0.0
    else:
        fn_integral, _ = quad(integrand_fn, threshold, 1, limit=100)
        fn_rate = fn_integral / (1 - threshold)

    return float(fp_rate), float(fn_rate)


def _divisor_pairs(num_perm: int) -> list[tuple[int, int]]:
    """Enumerate all (b, r) pairs with b * r == num_perm."""
    pairs = []
    for b in range(1, num_perm + 1):
        if num_perm % b == 0:
            pairs.append((b, num_perm // b))
    return pairs


def find_optimal_br(num_perm: int, target_threshold: float, tolerance: float = 0.05) -> tuple[int, int]:
    """
    Find the (b, r) factorization of num_perm best matching a target threshold.

    Enumerates ALL divisor pairs (b, r) with b * r == num_perm. Among pairs whose
    cosine threshold (compute_lsh_threshold) is within ``tolerance`` of
    ``target_threshold``, returns the one minimizing the false-rate sum (fp + fn).
    If no pair is within tolerance, returns the pair whose threshold is closest to
    the target.

    Args:
        num_perm: Total number of hash bits (b × r). Must be positive.
        target_threshold: Desired cosine similarity threshold.
        tolerance: Maximum acceptable deviation from target threshold.

    Returns:
        Tuple of (b, r) with b × r == num_perm. Never None.
    """
    pairs = _divisor_pairs(num_perm)

    within = []
    for b, r in pairs:
        thr = compute_lsh_threshold(b, r)
        if abs(thr - target_threshold) <= tolerance:
            within.append((b, r, thr))

    if within:
        best = min(within, key=lambda x: sum(compute_false_rates(x[0], x[1], target_threshold)))
        return best[0], best[1]

    # Nothing within tolerance: pick threshold closest to target.
    closest = min(pairs, key=lambda br: abs(compute_lsh_threshold(*br) - target_threshold))
    return closest


def get_optimal_config(num_perm: int, target_threshold: float = 0.5, tolerance: float = 0.05) -> tuple[int, int]:
    """
    Get the optimal LSH configuration for the requested cosine threshold.

    Always returns a divisor pair (b, r) of num_perm (b * r == num_perm) whose
    behavior best matches ``target_threshold`` under the cosine / SimHash model:

        1. Enumerate ALL divisor pairs of num_perm.
        2. Compute each pair's cosine threshold via compute_lsh_threshold.
        3. Among pairs within ``tolerance`` of the target, pick the one minimizing
           the false-rate sum (fp + fn) from compute_false_rates.
        4. If none are within tolerance, pick the pair whose threshold is CLOSEST
           to the target and emit a warning that num_perm is too coarse to hit the
           target threshold.

    Args:
        num_perm: Total number of hash bits (b × r). Must be positive.
        target_threshold: Target cosine similarity threshold. Default 0.5.
        tolerance: Maximum acceptable deviation from target threshold. Default 0.05.

    Returns:
        Tuple of (b, r) with b × r == num_perm exactly. Never None.

    Examples:
        >>> get_optimal_config(128, 0.5)
        (16, 8)
    """
    pairs = _divisor_pairs(num_perm)

    within = []
    for b, r in pairs:
        thr = compute_lsh_threshold(b, r)
        if abs(thr - target_threshold) <= tolerance:
            within.append((b, r))

    if within:
        best = min(within, key=lambda br: sum(compute_false_rates(br[0], br[1], target_threshold)))
        return best

    # No factorization lands within tolerance: return the closest achievable.
    closest = min(pairs, key=lambda br: abs(compute_lsh_threshold(*br) - target_threshold))
    achieved = compute_lsh_threshold(*closest)
    warnings.warn(
        f"num_perm={num_perm} is too coarse to hit target_threshold={target_threshold:.3f} "
        f"within tolerance={tolerance:.3f}. Closest achievable threshold is {achieved:.3f} "
        f"with (b, r)={closest}. Consider a more composite num_perm (e.g. a power of 2 "
        f"times small factors) for finer control over the threshold.",
        stacklevel=2,
    )
    return closest


def print_config_analysis(num_perm: int, threshold: float = 0.5):
    """
    Print a detailed analysis of an LSH configuration for debugging and tuning.

    Args:
        num_perm: Number of hash bits.
        threshold: Target cosine similarity threshold.

    Output includes optimal b/r, actual vs target threshold, false positive /
    negative rates, the S-curve steepness, and detection probabilities at several
    cosine similarity levels.
    """
    b, r = get_optimal_config(num_perm, threshold)

    actual_threshold = compute_lsh_threshold(b, r)
    fp_rate, fn_rate = compute_false_rates(b, r, threshold)

    print("LSH Configuration Analysis")
    print(f"{'=' * 50}")
    print(f"Number of permutations: {num_perm}")
    print(f"Target threshold: {threshold:.2f}")
    print("\nOptimal configuration:")
    print(f"  Bands (b): {b}")
    print(f"  Rows per band (r): {r}")
    print("\nPerformance metrics:")
    print(f"  Actual threshold: {actual_threshold:.4f}")
    print(f"  False positive rate: {fp_rate:.2%}")
    print(f"  False negative rate: {fn_rate:.2%}")
    print(f"  S-curve steepness: {b * r}")  # Higher = sharper transition

    print("\nDetection probabilities:")
    for sim in [0.3, 0.5, 0.7, 0.9]:
        prob = compute_collision_probability(sim, b, r)
        print(f"  Similarity {sim:.1f}: {prob:.2%} chance of detection")


# Module testing and demonstration
if __name__ == "__main__":
    print("Example configurations for common hash sizes:\n")

    for size in [128, 256, 512, 1024, 2048]:
        print(f"\nHash size: {size}")

        for threshold in [0.5, 0.8, 0.9]:
            b, r = get_optimal_config(size, threshold)
            actual = compute_lsh_threshold(b, r)
            print(f"  Threshold {threshold:.1f}: b={b:4d}, r={r:3d} (actual: {actual:.3f})")
