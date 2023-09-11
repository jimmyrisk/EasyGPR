import torch
import gpytorch
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Interval
from gpytorch.lazy import MatmulLazyTensor
from typing import Optional


class MinKernel(Kernel):

    def __init__(
        self,
        offset_prior: Optional[Prior] = None,
        offset_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super(MinKernel, self).__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Interval(0.001, 100, initial_value=0.01)
        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self) -> torch.Tensor:
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        self._set_offset(value)

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        x1_ = x1
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
        a = torch.ones(x1_.shape)
        # a.to(x1_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
        offset = self.offset.view(*self.batch_shape, 1, 1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            aa = MatmulLazyTensor(x1_, a.transpose(-2, -1))
            bb = aa.transpose(-2,-1)
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset

        else:
            x2_ = x2
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)
            b = torch.ones(x2_.shape)
            # b.to(x2_.device)  # for cuda vs cpu  (I don't think this works - may need fixing later)
            aa = MatmulLazyTensor(x1_, b.transpose(-2, -1))
            bb = MatmulLazyTensor(a, x2_.transpose(-2, -1))
            K = 0.5 * (aa + bb - torch.abs((aa - bb).evaluate())) + offset
        if diag:
            return K.diag()
        else:
            return K

class KernelWrapper:
    def __init__(self):
        self.kernel_instance = None

    def create_kernel(self, kernel_type, **kwargs):
        if kernel_type in ["rbf", "squared exponential", "gaussian"]:
            # The RBF kernel has many names
            self.kernel_instance = gpytorch.kernels.RBFKernel(**kwargs)
        elif kernel_type in ["exp", "exponential", "laplace", "mat12"]:
            # The exponential kernel has many names
            self.kernel_instance = gpytorch.kernels.MaternKernel(nu=0.5, **kwargs)
        elif kernel_type.startswith("mat"):
            if kernel_type == "mat32":
                nu_value = 1.5
            elif kernel_type == "mat52":
                nu_value = 2.5
            else:
                nu_value = float(kernel_type.split("_")[1])
            self.kernel_instance = gpytorch.kernels.MaternKernel(nu=nu_value, **kwargs)
        elif kernel_type == "linear":
            self.kernel_instance = gpytorch.kernels.LinearKernel(**kwargs)
        elif kernel_type == "minimum":
            self.kernel_instance = MinKernel(**kwargs)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        # todo: handle kernel addition and multiplication
        self.kernel_instance = gpytorch.kernels.ScaleKernel(self.kernel_instance)

    def get_kernel_instance(self):
        return self.kernel_instance

    def print_hyperparameters(self):
        if self.kernel_instance is not None:
            for name, param in self.kernel_instance.named_hyperparameters():
                print(f"{name}: {param.item()}")
        else:
            print("No kernel instance created yet.")