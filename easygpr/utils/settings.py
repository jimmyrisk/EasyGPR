import torch
import gpytorch


def set_gpytorch_settings():
    gpytorch.settings.fast_computations.covar_root_decomposition._set_state(False)
    gpytorch.settings.fast_computations.log_prob._set_state(False)
    gpytorch.settings.fast_computations.solves._set_state(False)
    gpytorch.settings.cholesky_max_tries._set_value(100)
    gpytorch.settings.debug._set_state(False)
    gpytorch.settings.min_fixed_noise._set_value(float_value=1e-7, double_value=1e-7, half_value=1e-7)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_dtype(torch.float64)
    else:
        import warnings
        warnings.warn("CUDA is not available. Falling back to CPU.")
        if 'LAPACK' not in torch.__config__.show():
            raise RuntimeError(
                "PyTorch was not installed with LAPACK support. Please reinstall PyTorch with LAPACK support.")
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_default_dtype(torch.float64)

    # For approximate likelihoods, like t or binomial
    gpytorch.settings.ciq_samples._set_state(False)
    gpytorch.settings.skip_logdet_forward._set_state(False)
    gpytorch.settings.num_trace_samples._set_value(0)
    gpytorch.settings.num_gauss_hermite_locs._set_value(300)
    gpytorch.settings.num_likelihood_samples._set_value(300)
    gpytorch.settings.deterministic_probes._set_state(True)
