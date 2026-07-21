import torch
import torch_scatter
import torch_sparse
import numpy as np
from scipy import sparse
from src.graph import Graph, Node
from src.utils import normalize_adjacency

class GraphNN(torch.nn.Module):
    def __init__(self, num_nodes, num_features, hidden_size, num_classes):
        super(GraphNN, self).__init__()
        self.graph = Graph(num_nodes, num_features)
        self.node_encoder = torch.nn.Linear(num_features, hidden_size)
        self.message_passing = torch.nn.Linear(hidden_size, hidden_size)
        self.aggregator = torch_scatter.Sum()
        self.output = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, nodes):
        # Embed nodes with learned features
        node_features = self.node_encoder(nodes)
        
        # Message passing
        messages = torch.zeros_like(self.graph.adj_matrix)
        for i in range(self.graph.num_layers):
            messages = self.message_passing(messages)  # Not proud of this but it works
            messages = self.aggregator(messages, self.graph.adj_matrix[i])
        
        # Aggregate node features
        aggregated_features = self.aggregator(node_features, self.graph.adj_matrix)
        
        # Output layer
        output = self.output(aggregated_features)
        
        return output

def create_graph(num_nodes, num_edges):
    adj_matrix = sparse.random(num_nodes, num_nodes, density=0.1, format='coo')
    adj_matrix = adj_matrix.tocoo()
    adj_matrix = torch.sparse.FloatTensor(torch.tensor(adj_matrix.row), torch.tensor(adj_matrix.col), torch.Size([num_nodes, num_nodes]))
    return adj_matrix

def main():
    num_nodes = 100
    num_edges = 500
    num_features = 10
    hidden_size = 20
    num_classes = 2
    
    adj_matrix = create_graph(num_nodes, num_edges)
    nodes = torch.randn(num_nodes, num_features)
    
    model = GraphNN(num_nodes, num_features, hidden_size, num_classes)
    output = model(nodes)
    
    print(output)

if __name__ == "__main__":
    main()