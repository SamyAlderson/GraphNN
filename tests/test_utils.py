import pytest
import numpy as np
from src.utils import normalize_adj, get_node_degrees

def test_normalize_adj():
    # Test with a simple adjacency matrix
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    normalized_adj = normalize_adj(adj)
    assert np.allclose(normalized_adj, [[0., 1., 0.], [0.5, 0., 0.5], [0., 0.5, 0.]])

    # Test with an empty adjacency matrix
    empty_adj = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        normalize_adj(empty_adj)

def test_get_node_degrees():
    # Test with a simple adjacency matrix
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    node_degrees = get_node_degrees(adj)
    assert node_degrees == [1, 2, 1]

    # Test with an empty adjacency matrix
    empty_adj = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        get_node_degrees(empty_adj)

def test_normalize_adj_symmetric():
    # Test with a symmetric adjacency matrix
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    normalized_adj = normalize_adj(adj)
    assert np.allclose(normalized_adj, [[0., 2./3, 2./3], [2./3, 0., 2./3], [2./3, 2./3, 0.]])

def test_get_node_degrees_symmetric():
    # Test with a symmetric adjacency matrix
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    node_degrees = get_node_degrees(adj)
    assert node_degrees == [2, 2, 2]