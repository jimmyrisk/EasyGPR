import pandas as pd
import numpy as np
import torch

def to_numpy(tensor):
    if tensor.requires_grad:
        if torch.cuda.is_available():
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    else:
        if torch.cuda.is_available():
            return tensor.cpu().numpy()
        else:
            return tensor.numpy()

def to_torch(array, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.Tensor(array.astype(np.float32)).to(device)

class MinMaxScaler:
    def __init__(self, mins=None, maxs=None):
        self.reset()
        self.mins = mins
        self.maxs = maxs

    def reset(self):
        self.mins = None
        self.maxs = None

    def fit(self, X):
        self.reset()
        if isinstance(X, pd.DataFrame):
            X = to_torch(X.values)
        elif isinstance(X, np.ndarray):
            X = to_torch(X)
        elif isinstance(X, torch.Tensor):
            X = X.float()

        self.mins = torch.min(X, dim=0).values
        self.maxs = torch.max(X, dim=0).values

        return self.scale(X)

    def scale(self, X):
        if isinstance(X, np.ndarray):
            X = to_torch(X)
        X_scaled = (X - self.mins) / (self.maxs - self.mins)
        return X_scaled

    def unscale(self, X_scaled):
        if isinstance(X_scaled, np.ndarray):
            X_scaled = to_torch(X_scaled)

        X = X_scaled.unsqueeze(-1) * (self.maxs - self.mins) + self.mins
        return X.squeeze(-1)


class NoScale:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def scale(self, X):
        return X

    def unscale(self, X_scaled):
        return X_scaled
