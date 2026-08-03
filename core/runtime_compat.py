"""Small compatibility shims for the mixed deployment environments.

The deployment image uses a NumPy 1.x build because the installed
PyTorch/OpenCV wheels depend on it. Some newer SciPy releases still expose
code paths that refer to NumPy aliases removed in NumPy 1.24. Transformers
5.x imports SciPy/Scikit-learn while initializing its generation mixin, so
that otherwise unrelated mismatch can make Qwen classes look unavailable.

Keep this shim deliberately narrow: it only adds an alias when the running
NumPy build does not already provide it. On an environment where the aliases
still exist, this function is a no-op.
"""

import numpy as np


def patch_numpy_legacy_aliases():
    """Restore legacy NumPy names needed by older numerical dependencies."""

    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "long": np.int64,
        "ulong": np.uint64,
    }
    for name, value in aliases.items():
        # Checking __dict__ avoids NumPy's deprecation-warning __getattr__.
        if name not in np.__dict__:
            setattr(np, name, value)
