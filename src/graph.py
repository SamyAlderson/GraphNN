import torch
from torch import nn
from torch_sparse import SparseTensor

class Node:
    """
    Node class representing a node in the graph.
    """
    def __init__(self, id, features):
        """
        Initialize a Node object.

        Args:
        id (int): Unique identifier for the node.
        features (torch.Tensor): Node features.
        """
        self.id = id
        self.features = features

class Graph:
    """
    Graph class representing a graph data structure.
    """
    def __init__(self, nodes, edges):
        """
        Initialize a Graph object.

        Args:
        nodes (list[Node]): List of Node objects representing the nodes in the graph.
        edges (list[tuple[int, int]]): List of edge tuples representing the edges in the graph.
        """
        self.nodes = nodes
        self.edges = edges
        self.adj_list = self.build_adjacency_list()

    def build_adjacency_list(self):
        """
        Build the adjacency list for the graph.

        Returns:
        dict[int, list[int]]: Adjacency list where each key is a node ID and the value is a list of neighboring node IDs.
        """
        adj_list = {}
        for u, v in self.edges:
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append(v)
            if v not in adj_list:
                adj_list[v] = []
            adj_list[v].append(u)
        return adj_list

    def to_sparse_tensor(self):
        """
        Convert the graph to a sparse tensor.

        Returns:
        SparseTensor: Sparse tensor representation of the graph.
        """
        row = []
        col = []
        val = []
        for u, v in self.edges:
            row.append(u)
            col.append(v)
            val.append(1)
        return SparseTensor(row=torch.tensor(row), col=torch.tensor(col), value=torch.tensor(val))

class GraphNN(nn.Module):
    """
    Graph neural network class.
    """
    def __init__(self, num_nodes, num_features, num_classes):
        """
        Initialize a GraphNN object.

        Args:
        num_nodes (int): Number of nodes in the graph.
        num_features (int): Number of node features.
        num_classes (int): Number of output classes.
        """
        super(GraphNN, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, num_features)
        self.message_passing = nn.Linear(num_features, num_features)
        self.aggregation = nn.Linear(num_features, num_features)
        self.output_layer = nn.Linear(num_features, num_classes)

    def forward(self, graph):
        """
        Forward pass for the graph neural network.

        Args:
        graph (Graph): Graph object.

        Returns:
        torch.Tensor: Output tensor for the graph neural network.
        """
        node_embeddings = self.node_embedding(graph.nodes.features)
        message = self.message_passing(node_embeddings)
        message = torch.sparse.mm(graph.adj_list, message)
        aggregated_message = self.aggregation(message)
        output = self.output_layer(aggregated_message)
        return output