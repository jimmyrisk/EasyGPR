import torch
import gpytorch
import numpy as np
import pandas as pd
from kernels import KernelWrapper
from utils.settings import set_gpytorch_settings
import utils.data_handling as data_handling


set_gpytorch_settings()


class GPRModel(gpytorch.models.ExactGP):
    """
    Gaussian Process Regression (GPR) Model class.

    This class encapsulates the GPR model, providing a simple and intuitive API for fitting the model to data and making predictions. It interacts with the GPyTorch library to perform these operations.

    Attributes:
        - train_x (Tensor): Training data inputs.
        - train_y (Tensor): Training data outputs.
        - likelihood (Likelihood): The likelihood model to use for the GPR.
        - kernel (Kernel): The kernel function to use for the GPR.
    """

    def __init__(self, train_x, train_y, kernel="rbf", kernel_kwargs={}, likelihood=None):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPRModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Handling kernel creation using KernelWrapper
        self.kernel_wrapper = KernelWrapper()
        self.kernel_wrapper.create_kernel(kernel, **kernel_kwargs)
        self.covar_module = self.kernel_wrapper.get_kernel_instance()

        self.predictions = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit_model(self, train_x, train_y, training_iterations=50):
        """
        Fit the GPR model to the training data.

        Args:
            - train_x (Tensor): The training data inputs.
            - train_y (Tensor): The training data outputs.
            - training_iterations (int): The number of iterations for training the model.

        Returns:
            - self: The fitted GPR model.
        """
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return self

    def make_predictions(self, test_x, return_type="numpy"):
        """
        Make predictions using the fitted GPR model.

        Args:
            - test_x (Tensor): The test data inputs.

        Returns:
            - predictions (Tensor): The predictions for the test data.
        """
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self(test_x))
            self.predictions = Predictions(predictions.mean, predictions.variance)

        if return_type == "numpy":
            self.predictions.to_numpy()
        elif return_type == "torch":
            pass
        else:
            raise ValueError("Invalid return_type. Valid options are 'numpy' and 'torch'.")
        return self.predictions

    def compute_bic(self, data):
        """
        Compute the Bayesian Information Criterion (BIC) for the fitted GPR model.

        Args:
            - data (Tensor): The data to use for computing the BIC.

        Returns:
            - bic (float): The BIC value for the fitted model.
        """
        # Implement the BIC computation procedure here
        pass

    def print_kernel_hyperparameters(self):
        """
        Print the hyperparameter values of the kernel in a readable format.
        """
        self.kernel_wrapper.print_hyperparameters()


# Additional utility functions can be added here as necessary

class Predictions:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def to_numpy(self):
        self.mean = data_handling.to_numpy(self.mean)
        self.variance = data_handling.to_numpy(self.variance)

    def to_torch(self):
        self.mean = data_handling.to_torch(self.mean)
        self.variance = data_handling.to_torch(self.mean)


if __name__ == "__main__":
    # Example usage of the GPRModel class and utility functions
    pass


