import importlib.metadata
from typing import Final

from lshrs.core.main import LSHRS, lshrs


try:
    _version = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    _version = "0.0.0"  # Fallback for development mode
__version__: Final[str] = _version

# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("numpy", "redis")

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        raise ImportError(
            f"Unable to import required dependency {_dependency}. "
            "Please see the traceback for details."
        ) from _e

del _hard_dependencies, _dependency


__all__ = ["LSHRS", "lshrs"]
