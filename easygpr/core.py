# TODO: clean up imports
from easygpr.utils import set_gpytorch_settings
set_gpytorch_settings()

import torch
import gpytorch
import numpy as np
from easygpr.kernels import KernelWrapper

from easygpr.utils import MinMaxScaler, NoScale
import easygpr.utils as utils
from easygpr.likelihoods import ScaledHeteroNoiseGaussianLikelihood






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

    def __init__(self, train_x=None, train_y=None, kernel="rbf", scale_x=True, kernel_kwargs={}, likelihood=None, mean="constant", noise_x=None, heteroskedastic=False):

        self.scale_x = scale_x
        self.noise_x = noise_x

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()


        if scale_x is True:
            # Initialize and fit the MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = NoScale()

        # Case of no training data, e.g. for prior GPs
        if train_x is None and train_y is None:
            self.train_x_scaled = None

        # Scale and handle data
        else:
            if isinstance(train_x, np.ndarray):
                train_x = utils.to_torch(train_x)
            if isinstance(train_y, np.ndarray):
                train_y = utils.to_torch(train_y)
            self.scaler.fit(train_x)
            self.train_x_scaled = self.scaler.scale(train_x)

        self.train_x = train_x
        self.train_y = train_y

        # Correctly initialize according to gpytorch.models.ExactGP
        super(GPRModel, self).__init__(self.train_x_scaled, self.train_y, likelihood)

        # Initialize the mean module based on the specified type
        if mean == "none":
            self.mean_module = None
        elif mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean == "linear":
            self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[1])
        else:
            raise ValueError("Invalid mean_module_type. Valid options are 'none', 'constant', and 'linear'.")

        # Handling kernel creation using KernelWrapper
        self.kernel_wrapper = KernelWrapper()
        self.kernel_wrapper.create_kernel(kernel, **kernel_kwargs)
        self.covar_module = self.kernel_wrapper.get_kernel_instance()

        self.predictions = None

        # Add a trainable noise parameter
        if heteroskedastic:
            self.sigma = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            noise = self.sigma * noise_x  # Compute heteroskedastic noise
            self.likelihood = ScaledHeteroNoiseGaussianLikelihood(
                noise=noise
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        if self.noise_x is not None:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit_model(self, training_iterations=50, verbose = True, cosine_annealing = True):
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

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        if cosine_annealing == True:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            scheduler = scheduler(optimizer=optimizer,  T_0 = 5, T_mult = 1)
        else:
            scheduler = None

        # Use the adam optimizer


        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)


        # TODO: convergence tolerance
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self(self.train_x_scaled)

            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        self.compute_bic()

        if verbose is True:
            print("Fitting complete.")
            print(f"--- ")
            print(f"--- final mll: {-loss:.4f}")
            print(f"--- num_params: {self.num_param}")
            print(f"--- BIC: {self.bic:.4f}")


        return self


    def simulate(self, x_sim, method='prior', type='f', return_type = 'numpy', n_paths = 1):
        """
        This method generates samples from a multivariate normal distribution, either from the prior or the posterior.

        Parameters:
        x_sim (torch.Tensor): The x values for which the samples will be generated.
        method (str): Specifies whether to generate samples from the 'prior' or the 'posterior'. Default is 'prior'.
        type (str): Specifies the type of samples to generate - 'f' for the underlying GP (function values) and 'y' for predictions (observations). Default is 'f'.

        Returns:
        torch.Tensor: Generated samples.
        """
        if n_paths > 1:
            # todo: address this case
            raise NotImplementedError("Currently supports simulating one sample path.  You can rerun 'simulate' to get additional paths.")

        # Strangely, GPyTorch doesn't allow "train mode" (i.e. prior) if there is no training data.
        if self.train_x is None and self.train_y is None:
            self.eval()
        else:
            if method == 'prior':
                self.train()
            elif method == 'posterior':
                self.eval()

        if isinstance(x_sim, np.ndarray):
            x_sim = utils.to_torch(x_sim)

        with torch.no_grad():
            # Getting the predictive distribution
            predictive_dist = self(x_sim)

            if type == 'f':
                # Getting samples from the GP (prior or posterior)
                realizations = predictive_dist.rsample()
            elif type == 'y':
                # Getting samples from the likelihood (observations)
                realizations = self.likelihood(predictive_dist).rsample()

            if return_type == "numpy":
                realizations = utils.to_numpy(realizations)
            elif return_type == "torch":
                pass
            else:
                raise ValueError("Invalid return_type. Valid options are 'numpy' and 'torch'.")
            return realizations



    def make_predictions(self, test_x, type = "f", return_type="numpy"):
        """
        Make predictions using the fitted GPR model.

        Args:
            - test_x (Tensor): The test data inputs.

        Returns:
            - predictions (Tensor): The predictions for the test data.
        """
        self.eval()
        self.likelihood.eval()

        test_x_scaled = self.scaler.scale(test_x)



        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type == "f":
                predictions = self(test_x_scaled)
            elif type == "y":
                predictions = self.likelihood(self(test_x_scaled))
            else:
                raise ValueError("Invalid type.  Use 'f' for latent function or 'y' for noisy predictions.")
            self.predictions = Predictions(predictions.mean, predictions.variance)

        if return_type == "numpy":
            self.predictions.to_numpy()
        elif return_type == "torch":
            pass
        else:
            raise ValueError("Invalid return_type. Valid options are 'numpy' and 'torch'.")
        return self.predictions

    def compute_bic(self, data = None):
        """
        Compute the Bayesian Information Criterion (BIC) for the fitted GPR model.

        Args:
            - data (Tensor): The data to use for computing the BIC.  Uses training data by default.

        Returns:
            - bic (float): The BIC value for the fitted model.
        """
        # Implement the BIC computation procedure here

        self.train()

        if data is None:
            data = self.train_x_scaled

        # Get the number of data points
        n = data.shape[0]

        # Initialize the marginal log likelihood object
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self).to(data.device)

        # Get the model output for the data
        output = self(data)

        # Compute the log marginal likelihood
        log_marginal_likelihood = mll(output, self.train_y)

        # Get the number of hyperparameters
        self.num_param = sum(p[1].numel() for p in self.named_parameters())

        # Compute the BIC
        with torch.no_grad():
            self.bic = -2 * log_marginal_likelihood + self.num_param * np.log(n)

        return self.bic

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
        self.mean = utils.to_numpy(self.mean)
        self.variance = utils.to_numpy(self.variance)

    def to_torch(self):
        self.mean = utils.to_torch(self.mean)
        self.variance = utils.to_torch(self.mean)


if __name__ == "__main__":
    # Example usage of the GPRModel class and utility functions
    pass


