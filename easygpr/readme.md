
# EasyGPR

EasyGPR is a Python package that serves as a user-friendly wrapper around the GPyTorch library, simplifying Gaussian Process Regression (GPR) for users, particularly those with a background in R. It aims to provide a similar functionality and syntax to the DiceKriging package available in R, with an emphasis on ease of use and statistical rigor.

## Features

- Simple and intuitive API for Gaussian Process Regression.
- Supports various kernel functions, with a structure inspired by R programming language and DiceKriging package.
- Automatic data handling to accommodate torch tensors, numpy arrays, and pandas dataframes.
- Focus on statistical rigor with functionalities to compute likelihoods and Bayesian Information Criterion (BIC).
- Comprehensive documentation and examples to help users get started quickly.

## Installation

To install EasyGPR, you can use the following command:

```bash
pip install easygpr
```

## Quick Start

Here's a quick example to get you started with EasyGPR:

```python
from easygpr import gp

# Generate some data (X: features, y: responses)
X = ...
y = ...

# Create a GP model with an RBF kernel
fit = gp(X, y, kernel="rbf")

# Make predictions on new data
X_test = ...
predictions = fit.predict(X_test)
```

## Documentation

TODO: Add documentation

## Examples

You can find examples demonstrating the usage of EasyGPR in the `examples/` directory of this repository.

## License

TODO: Add license

