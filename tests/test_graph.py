import pytest
import torch
import torch_scatter
import torch_sparse
from src.graph import Graph, Node
from src.utils import normalize_adjacency

# Test that an empty graph is created correctly
def test_empty_graph():
    g = Graph()
    assert g.num_nodes() == 0
    assert g.num_edges() == 0

# Test that a graph with a single node is created correctly
def test_single_node_graph():
    g = Graph()
    g.add_node(0)
    assert g.num_nodes() == 1
    assert g.num_edges() == 0

# Test that a graph with multiple nodes is created correctly
def test_multiple_node_graph():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    assert g.num_nodes() == 3
    assert g.num_edges() == 0

# Test that adding an edge to a graph works correctly
def test_add_edge():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    assert g.num_nodes() == 2
    assert g.num_edges() == 1

# Test that adding an edge with a weight works correctly
def test_add_edge_with_weight():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1, 2.0)
    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    assert g.get_edge_attr('weight')[0] == 2.0

# Test that getting the adjacency matrix works correctly
def test_get_adjacency_matrix():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    adj_matrix = g.get_adjacency_matrix()
    assert adj_matrix.shape == (2, 2)
    assert torch.allclose(adj_matrix[0, 1], 1.0)
    assert torch.allclose(adj_matrix[1, 0], 1.0)

# Test that getting the degree matrix works correctly
def test_get_degree_matrix():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    degree_matrix = g.get_degree_matrix()
    assert degree_matrix.shape == (2, 2)
    assert torch.allclose(degree_matrix[0, 0], 1.0)
    assert torch.allclose(degree_matrix[1, 1], 2.0)

# Test that normalizing the adjacency matrix works correctly
def test_normalize_adjacency():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    adj_matrix = g.get_adjacency_matrix()
    normalized_adj_matrix = normalize_adjacency(adj_matrix)
    assert normalized_adj_matrix.shape == (2, 2)
    assert torch.allclose(normalized_adj_matrix[0, 0], 1.0)
    assert torch.allclose(normalized_adj_matrix[1, 1], 1.0)

# Test that getting the graph's nodes and edges works correctly
def test_get_nodes_and_edges():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    nodes = g.get_nodes()
    edges = g.get_edges()
    assert nodes == [0, 1, 2]
    assert edges == [(0, 1), (1, 2)]

# Test that a node's attributes can be accessed correctly
def test_node_attributes():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    node0 = Node(g, 0)
    node1 = Node(g, 1)
    node2 = Node(g, 2)
    node0.attr = 'value'
    assert node0.attr == 'value'
    assert node1.attr is None
    assert node2.attr is None

# Test that a node's attribute can be updated correctly
def test_update_node_attribute():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    node0 = Node(g, 0)
    node1 = Node(g, 1)
    node2 = Node(g, 2)
    node0.attr = 'value'
    node0.attr = 'new_value'
    assert node0.attr == 'new_value'

# Test that a node can be deleted correctly
def test_delete_node():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    node0 = Node(g, 0)
    node1 = Node(g, 1)
    node2 = Node(g, 2)
    node3 = Node(g, 3)
    g.delete_node(0)
    assert len(g.nodes()) == 3
    assert node1 in g.nodes()
    assert node2 in g.nodes()
    assert node3 in g.nodes()
    assert node0 not in g.nodes()

# Test that a graph can be cleared correctly
def test_clear_graph():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    assert len(g.nodes()) == 4
    g.clear()
    assert len(g.nodes()) == 0

# Test that the graph's edge list can be accessed correctly
def test_edge_list():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    edge_list = g.edge_list()
    assert edge_list == [(0, 1), (1, 0)]

# Test that the graph's edge indices can be accessed correctly
def test_edge_indices():
    g = Graph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 0)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 1)
    edge_indices = g.edge_indices()
    assert edge_indices == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]