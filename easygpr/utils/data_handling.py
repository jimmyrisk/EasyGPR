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
