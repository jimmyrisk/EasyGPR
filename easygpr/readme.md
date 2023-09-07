
# EasyGPR

EasyGPR is a Python package that serves as a user-friendly wrapper around the GPyTorch library, simplifying Gaussian Process Regression (GPR) for users, particularly those with a background in R. It aims to provide a similar functionality and syntax to the DiceKriging package available in R, with an emphasis on ease of use and statistical rigor.

## Features

- Simple and intuitive API for Gaussian Process Regression.
- Supports various kernel functions, with a structure inspired by R programming language and DiceKriging package.
- Automatic data handling to accommodate torch tensors, numpy arrays, and pandas dataframes.
- Focus on statistical rigor with functionalities to compute likelihoods and Bayesian Information Criterion (BIC).
- Comprehensive documentation and examples to help users get started quickly.

## First Time Python Users

Congratulations on taking the leap into python. This package is designed to bridge the gap from users with little programming experience or with other languages like R, into doing Gaussian Process regression with GPyTorch. If it is your first time using python, I recommend the following steps to get started.

### Install Anaconda

Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment. It is particularly useful for managing complex dependencies and allows for the creation of isolated environments to work on separate projects without interference.

[Anaconda Installation Guide](https://docs.anaconda.com/free/anaconda/install/index.html)

### Accessing the Anaconda Command Prompt

After installing Anaconda, you will need to access the Anaconda Command Prompt to manage your environments and packages. Here's how you can do it based on your operating system:

- **Windows**: Search for "Anaconda Prompt" in the start menu and open it. This prompt allows you to use conda commands directly.
  
- **macOS and Linux**: Open your terminal (you can search for "terminal" in the finder or applications menu).


### Install an IDE

An Integrated Development Environment (IDE) is a software application that provides comprehensive facilities to computer programmers for software development. An IDE normally consists of at least a source code editor, build automation tools, and a debugger.

I recommend using PyCharm professional, which is free to students and academics. [Download PyCharm](https://www.jetbrains.com/pycharm/download/?section=windows)

### Creating an Environment

In the context of Python, an environment is a folder that contains a specific collection of packages that you have installed. For instance, you might have one environment with a version of PyTorch for one project, and another environment with a different version for another project. It helps to avoid version conflicts between packages and allows you to work on various projects more efficiently.

Here are the conda commands to create and install an environment for EasyGPR:

```shell
conda create -n easygpr python=3.10
conda activate easygpr
```

After you create and activate the ``easygpr`` environment, make sure that it is your active environment in your IDE.

## Package Installation

The correct version of PyTorch needs to be installed first.

**Before installing PyTorch, it is recommended to check if your system supports CUDA. Here are the steps to do this based on your operating system:**

```shell
nvidia-smi
```

If your system supports CUDA, you will see a summary of your GPU and the installed CUDA version. If not, an error message will be displayed.

Then, use conda to install according to whether you have CUDA support or not.  Make sure you have the environment activated through ``conda activate easygpr`` 

#### If CUDA is supported

Installing with CUDA is recommended, as it takes advantage of speed improvements through the torch library.

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### If CUDA is not supported

```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Next Steps

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

