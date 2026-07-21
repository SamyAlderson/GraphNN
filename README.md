# GraphNN
A simple graph neural network implementation using PyTorch

## What and Why
GraphNN is a basic implementation of a graph neural network using PyTorch. It includes a graph data structure, node and edge representations, message passing and aggregation, and a scalable and efficient training loop.

## Install
To install GraphNN, run the following command:
```bash
pip install -r requirements.txt
```
Alternatively, you can build and install the project from source using the following command:
```bash
pip install .
```
## Usage
To use GraphNN, simply run the main entry point:
```bash
python src/main.py
```
This will launch a simple training loop that demonstrates the usage of the graph neural network.

## Build from Source
To build GraphNN from source, run the following command:
```bash
pip install -r requirements.txt
python setup.py install
```
This will build the project and install it on your system.

## Project Structure
The project is structured as follows:

* `src/`: contains the source code for the project
	+ `main.py`: main entry point
	+ `graph.py`: graph and node implementations
	+ `utils.py`: utility functions
* `tests/`: contains unit tests for the project
	+ `test_graph.py`: unit tests for graph module
	+ `test_utils.py`: unit tests for utility functions
* `setup.py`: build and installation script
* `requirements.txt`: list of dependencies required by the project

## License
GraphNN is released under the MIT License.

## Dependencies
GraphNN depends on the following packages:

* `torch`
* `torch-scatter`
* `torch-sparse`
* `numpy`
* `scipy`
* `pytest`
* `pytest-cov`

## Contributing
Contributions are welcome! Please see the CONTRIBUTING.md file for details.

## Architecture
See ARCHITECTURE.md for a high-level overview of the project architecture.

## Tests
GraphNN includes unit tests for the graph module and utility functions. To run the tests, use the following command:
```bash
pytest
```