import torch

def to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()

def to_torch(array, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.Tensor(array).to(device)



class MinMaxScaler:
    def __init__(self, mins=None, maxs=None):
        self.mins = mins
        self.maxs = maxs

    def fit(self, X):
        self.mins, _ = torch.min(X, dim=0)
        self.maxs, _ = torch.max(X, dim=0)

    def scale(self, X):
        X_scaled = (X - self.mins) / (self.maxs - self.mins)
        return X_scaled

    def unscale(self, X_scaled):
        X = X_scaled * (self.maxs - self.mins) + self.mins
        return X

class NoScale:
    def __init__(self):
        pass
    def fit(self, X):
        pass

    def scale(self, X):
        return X

    def unscale(self, X_scaled):
        return X_scaled
