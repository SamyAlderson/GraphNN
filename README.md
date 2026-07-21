# GraphNN
A simple graph neural network implementation using PyTorch for scalable data analysis

## What it does
GraphNN is a lightweight PyTorch library for building graph neural networks (GNNs). It provides a basic implementation of graph data structures, node and edge representations, message passing and aggregation, and an efficient training loop. This library is useful for data scientists who want to explore GNNs in a simple and flexible way.

## Installation
```bash
pip install graphnn
```
## Usage
```python
from graphnn import Graph, Node, Edge
from torch import nn

# Create a sample graph
g = Graph()
n1 = Node(1)
n2 = Node(2)
e1 = Edge(n1, n2, weight=0.5)
g.add_node(n1)
g.add_node(n2)
g.add_edge(e1)

# Define a GNN model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

# Train the model
model = MyModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(n1, n2)
    loss = criterion(outputs, torch.tensor([1, 2]))
    loss.backward()
    optimizer.step()
```
## Building from source
```bash
git clone https://github.com/SamyAlderson/GraphNN.git
cd GraphNN
pip install -r requirements.txt
python setup.py install
```
## Running tests
```bash
python -m unittest discover
```
## Project structure
- `graphnn/__init__.py`: Module initialization and imports
- `graphnn/graph.py`: Graph data structure implementation
- `graphnn/node.py`: Node representation and utilities
- `graphnn/edge.py`: Edge representation and utilities
- `graphnn/model.py`: GNN model definition and training loop
- `graphnn/utils.py`: Various utility functions
- `tests/test_graphnn.py`: Unit tests for the library
- `setup.py`: Build and installation script

## License
Copyright (c) 2026 SamyAlderson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.