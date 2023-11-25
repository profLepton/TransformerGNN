# Transformer GNN

Transfomer based GNN implementatoin. Given a graph, returns an n-dimensional vector for each of them to be used in downstream tasks.

# Key Idea

Uses a single attnetion mechanism to aggregate node vectors over r-degree neighborhood. Adjacency matrix is used as a mask.


# Advantages of other attention based GNNs

1. Drastically reduces number of attention based aggregations with respect to r.

2. Faster to compute, parallelize.

3. Achieves r-layer message passing by including multiple layers in the transformer.



# How to run

1. Setup all required packages by using pip install .

2. All of the initial settings and inputs can be configured at the start of train.py. You can put your traversable graph in here, and change inputs in the config object there.

3. After running the train.py script, the vector embeddings will be saved in output, along with the config file.
