"""
EasyGPR: A User-Friendly Wrapper for GPyTorch

EasyGPR is a Python package that serves as a user-friendly wrapper around the GPyTorch library, simplifying Gaussian Process Regression (GPR) for users, particularly those with a background in R. It aims to provide a similar functionality and syntax to the DiceKriging package available in R, with an emphasis on ease of use and statistical rigor.

Features:
- Simple and intuitive API for Gaussian Process Regression.
- Supports various kernel functions, with a structure inspired by R programming language and DiceKriging package.
- Automatic data handling to accommodate torch tensors, numpy arrays, and pandas dataframes.
- Focus on statistical rigor with functionalities to compute likelihoods and Bayesian Information Criterion (BIC).
- Comprehensive documentation and examples to help users get started quickly.

Modules:
- core: Contains the core functionalities of the package.
- kernels: Contains various kernel functions for GPR.
- likelihoods: Contains likelihood functions for GPR.

Subpackages:
- utilities: Contains utility functions for data handling and metrics.
- examples: Contains example scripts demonstrating the usage of the package.
- tests: Contains test scripts for the package modules.

For more information, refer to the package documentation.
"""

# Import main classes and functions from modules
from .core import (
    # Import necessary classes and functions from core.py
    # Example: ClassName, function_name
)

from .kernels import (
    # Import necessary classes and functions from kernels.py
    # Example: KernelClassName, kernel_function_name
)

from .likelihoods import (
    # Import necessary classes and functions from likelihoods.py
    # Example: LikelihoodClassName, likelihood_function_name
)

# Import subpackages
from . import utils
from . import examples
from . import tests

__all__ = [
    # List all the classes and functions imported above
    # Example: "ClassName", "function_name", "KernelClassName", "kernel_function_name", "LikelihoodClassName", "likelihood_function_name"
]

# Package metadata
__version__ = "0.1.0"  # Update the version number as necessary
