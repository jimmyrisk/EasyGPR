import math
import warnings
from copy import deepcopy
from typing import Any, Optional, Tuple, Union
from torch import Tensor
from linear_operator.operators import LinearOperator, ZeroLinearOperator

import gpytorch
import torch
from torch.distributions import StudentT

from gpytorch.lazy import ZeroLazyTensor
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise
from linear_operator.operators import ZeroLinearOperator

from gpytorch.likelihoods import _GaussianLikelihoodBase, _OneDimensionalLikelihood

from gpytorch.constraints import GreaterThan, Interval, Positive
from gpytorch.distributions import base_distributions
from gpytorch.priors import Prior

class ScaledHeteroNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = True,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
            )
        else:
            self.second_noise_covar = None

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise * self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Tensor:
        if self.second_noise_covar is None:
            return 0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs):
        if "noise" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res * self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res







class ScaledHeteroNoiseStudentTLikelihood(_OneDimensionalLikelihood):
    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = True,
            batch_shape: torch.Size = torch.Size([]),
            deg_free_prior: Optional[Prior] = None,
            deg_free_constraint: Optional[Interval] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.noise_covar = FixedGaussianNoise(noise=noise)
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
            )
        else:
            self.second_noise_covar = None

        if deg_free_constraint is None:
            deg_free_constraint = GreaterThan(2)


        self.raw_deg_free = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        if deg_free_prior is not None:
            self.register_prior("deg_free_prior", deg_free_prior, lambda m: m.deg_free, lambda m, v: m._set_deg_free(v))

        self.register_constraint("raw_deg_free", deg_free_constraint)

        # Rough initialization
        self.initialize(deg_free=7)

    @property
    def deg_free(self) -> Tensor:
        return self.raw_deg_free_constraint.transform(self.raw_deg_free)

    @deg_free.setter
    def deg_free(self, value: Tensor) -> None:
        self._set_deg_free(value)

    def _set_deg_free(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_deg_free)
        self.initialize(raw_deg_free=self.raw_deg_free_constraint.inverse_transform(value))

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise * self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Tensor:
        if self.second_noise_covar is None:
            return 0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)




    def get_fantasy_likelihood(self, **kwargs):
        if "noise" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedGaussianNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res * self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> StudentT:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return base_distributions.StudentT(df=self.deg_free, loc=function_samples, scale=noise.sqrt())