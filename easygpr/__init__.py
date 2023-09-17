# Import main classes and functions from modules
from .core import (
    GPRModel
)

from .kernels import (
    KernelWrapper
)


# from .likelihoods import (
#     LikelihoodWrapper
# )

# Import subpackages

__all__ = [
    "GPRModel", "KernelWrapper"
]

# Package metadata
__version__ = "0.1.0"  # Update the version number as necessary
