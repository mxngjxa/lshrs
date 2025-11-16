"""
Optimized Band/Row Calculator for LSHRS
Combines simple heuristics with sophisticated optimization.
No scipy dependencies - uses efficient numpy operations only.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class broptimizer:
    """
    Advanced LSH band/row optimizer for the LSHRS system.

    Supports:
    - Hash sizes from small (128) to large-scale (2^15 = 32,768)
    - Multiple optimization strategies (balanced, fpr, fnr, threshold)
    - Memory-aware configuration for production deployments
    - No scipy dependencies - pure numpy implementation
    """

    # Pre-computed optimal configurations for common hash sizes
    # Carefully tuned based on LSH theory: threshold ≈ (1/b)^(1/r)
    PRECOMPUTED = {
        128: {
            0.5: (16, 8),
            0.7: (8, 16), 
            0.8: (8, 16),
            0.9: (4, 32),
        },
        256: {
            0.5: (32, 8),
            0.7: (16, 16),
            0.8: (16, 16),
            0.9: (8, 32),
        },
        1024: {
            0.5: (64, 16),
            0.7: (32, 32),
            0.8: (32, 32),
            0.9: (16, 64),
        },
        32768: {  # 2^15 - Production scale
            0.4: (4096, 8),    # Fuzzy matching
            0.5: (2048, 16),   # High recall
            0.6: (2048, 16),   # Moderate recall
            0.7: (1024, 32),   # Balanced
            0.8: (1024, 32),   # Standard dedup
            0.85: (512, 64),   # High precision
            0.9: (512, 64),    # Very high precision
            0.95: (256, 128),  # Near-exact
        }
    }

    @staticmethod
    def find_optimal_br(
        n: int,
        target_threshold: float = 0.8,
        optimize_for: str = 'balanced',
        prefer_power_of_2: bool = False,
        max_bands: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Find optimal b and r values for LSH.

        Args:
            n: Number of hash functions (permutations)
            target_threshold: Target similarity threshold (0.3 to 0.95)
            optimize_for: One of:
                - 'simple': sqrt(n) heuristic (fastest)
                - 'balanced': minimize FPR + FNR
                - 'fpr': minimize false positives
                - 'fnr': minimize false negatives  
                - 'threshold': match target threshold
                - 'memory': minimize memory (fewer bands)
            prefer_power_of_2: Use power-of-2 values for efficiency
            max_bands: Maximum bands allowed (memory constraint)

        Returns:
            (bands, rows) tuple
        """

        # Fast path for precomputed common cases
        if n in LSHOptimizer.PRECOMPUTED and optimize_for == 'threshold':
            configs = LSHOptimizer.PRECOMPUTED[n]
            best_threshold = min(configs.keys(), 
                               key=lambda x: abs(x - target_threshold))
            if abs(best_threshold - target_threshold) < 0.1:
                return configs[best_threshold]

        # Simple heuristic
        if optimize_for == 'simple':
            b = int(np.sqrt(n))
            # Ensure clean division
            while n % b != 0 and b > 1:
                b -= 1
            r = n // b
            return b, r

        # Get candidate divisors
        if n > 10000:
            divisors = LSHOptimizer._get_efficient_divisors(n, prefer_power_of_2)
        else:
            divisors = [(n//r, r) for r in range(1, n+1) if n % r == 0]

        # Apply constraints
        if max_bands:
            divisors = [(b, r) for b, r in divisors if b <= max_bands]

        if not divisors:
            return LSHOptimizer.find_optimal_br(n, optimize_for='simple')

        best_b, best_r = divisors[0]
        best_score = float('inf')

        for b, r in divisors:
            threshold = (1/b)**(1/r) if b > 0 else 0

            if optimize_for == 'threshold':
                score = abs(threshold - target_threshold)
            elif optimize_for == 'memory':
                score = b  # Fewer bands = less memory
            elif optimize_for in ['balanced', 'fpr', 'fnr']:
                # Estimate error rates using trapezoidal rule
                fpr, fnr = LSHOptimizer._estimate_rates(r, b, threshold)

                if optimize_for == 'fpr':
                    score = fpr + 0.1 * fnr
                elif optimize_for == 'fnr':
                    score = 0.1 * fpr + fnr
                else:  # balanced
                    score = fpr + fnr
            else:
                score = abs(threshold - 0.8)

            if score < best_score:
                best_score = score
                best_b, best_r = b, r

        return best_b, best_r

    @staticmethod
    def _estimate_rates(r: int, b: int, threshold: float, 
                       n_samples: int = 20) -> Tuple[float, float]:
        """Estimate false positive and false negative rates."""
        # S-curve: P = 1 - (1 - s^r)^b

        # FPR: probability when similarity < threshold
        if threshold > 0:
            x = np.linspace(0, threshold, n_samples)
            y = [1 - (1 - xi**r)**b for xi in x]
            fpr = np.trapz(y, x) / threshold
        else:
            fpr = 0

        # FNR: probability of missing when similarity >= threshold
        if threshold < 1:
            x = np.linspace(threshold, 1, n_samples)
            y = [(1 - xi**r)**b for xi in x]
            fnr = np.trapz(y, x) / (1 - threshold)
        else:
            fnr = 0

        return fpr, fnr

    @staticmethod
    def _get_efficient_divisors(n: int, prefer_power_of_2: bool) -> List[Tuple[int, int]]:
        """Get efficient subset of divisors for large n."""
        divisors = []

        if prefer_power_of_2:
            # Powers of 2 only
            power = 1
            while power <= n:
                if n % power == 0:
                    b = n // power
                    if b & (b - 1) == 0:  # b is also power of 2
                        divisors.append((b, power))
                power *= 2
        else:
            # Sample divisors logarithmically
            candidates = set()

            # Powers of 2
            r = 1
            while r <= n:
                candidates.add(r)
                r *= 2

            # Near sqrt(n)
            sqrt_n = int(np.sqrt(n))
            for offset in range(-5, 6):
                r = sqrt_n + offset
                if 0 < r <= n:
                    candidates.add(r)

            # Some fractions of n
            for divisor in [4, 8, 16, 32, 64, 128, 256, 512]:
                if n > divisor:
                    candidates.add(n // divisor)

            # Check valid divisors
            for r in sorted(candidates):
                if n % r == 0:
                    divisors.append((n // r, r))

        return divisors


class OptimalBR:
    """
    Legacy class for backward compatibility.
    Redirects to new LSHOptimizer implementation.
    """

    def __init__(self):
        self.t0 = 0.5  # Default threshold

    def br(self, n: int) -> Tuple[int, int]:
        """Find optimal b,r using the new optimizer."""
        return LSHOptimizer.find_optimal_br(
            n, 
            target_threshold=self.t0,
            optimize_for='balanced'
        )

    def false_positive(self, r: int, b: int) -> float:
        """Estimate false positive rate."""
        threshold = (1/b)**(1/r) if b > 0 else 0
        fpr, _ = LSHOptimizer._estimate_rates(r, b, threshold)
        return fpr

    def false_negative(self, r: int, b: int) -> float:
        """Estimate false negative rate."""
        threshold = (1/b)**(1/r) if b > 0 else 0
        _, fnr = LSHOptimizer._estimate_rates(r, b, threshold)
        return fnr


def br(num_permutations: int, optimize: bool = False, 
       threshold: float = 0.8) -> Tuple[int, int]:
    """
    Compute optimal b and r values for LSHRS algorithm.

    This is the main interface function, providing both simple
    and sophisticated optimization options.

    Args:
        num_permutations: Total number of MinHash permutations
        optimize: If True, use advanced optimization. If False, use sqrt heuristic.
        threshold: Target similarity threshold (only used if optimize=True)

    Returns:
        (bands, rows) tuple where bands * rows ≈ num_permutations

    Examples:
        >>> # Simple heuristic for quick setup
        >>> b, r = br(128)
        >>> print(f"b={b}, r={r}")  # b=11, r=11 or similar

        >>> # Optimized for specific threshold
        >>> b, r = br(128, optimize=True, threshold=0.8)

        >>> # Large-scale configuration
        >>> b, r = br(32768, optimize=True, threshold=0.8)
        >>> print(f"b={b}, r={r}")  # b=1024, r=32
    """

    if not optimize:
        # Simple sqrt heuristic - fast and reasonable
        b = int(np.sqrt(num_permutations))

        # Ensure we use all permutations
        while num_permutations % b != 0 and b > 1:
            b -= 1
        r = num_permutations // b

        return b, r
    else:
        # Use advanced optimizer
        return LSHOptimizer.find_optimal_br(
            num_permutations,
            target_threshold=threshold,
            optimize_for='threshold',
            prefer_power_of_2=(num_permutations >= 1024)
        )


# Export main interfaces
__all__ = ['br', 'LSHOptimizer', 'OptimalBR']
