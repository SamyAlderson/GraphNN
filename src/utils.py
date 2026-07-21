# src/utils.py

import numpy as np
import torch

def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix by dividing by the sum of its rows.
    This is a common operation in graph neural networks to prevent exploding gradients.
    """
    row_sums = torch.sum(adjacency, dim=1)
    return adjacency / row_sums.unsqueeze(1)

def to_tensor(adjacency: np.ndarray) -> torch.Tensor:
    """
    Convert numpy adjacency matrix to PyTorch tensor.
    This is necessary because PyTorch requires tensors for computations.
    """
    return torch.from_numpy(adjacency).float()

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    This is useful for converting tensor outputs to a more human-readable format.
    """
    return tensor.detach().numpy()

def check_symmetry(adjacency: torch.Tensor) -> None:
    """
    Check if the adjacency matrix is symmetric.
    This is an important property of graph data, and we want to ensure it's preserved.
    """
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("Adjacency matrix is not symmetric")

def check_nonzero(adjacency: torch.Tensor) -> None:
    """
    Check if the adjacency matrix contains any zero values.
    This is a common issue in graph data, and we want to catch it early.
    """
    if torch.any(torch.eq(adjacency, 0)):
        raise ValueError("Adjacency matrix contains zero values")