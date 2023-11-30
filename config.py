from dataclasses import dataclass
import torch




####### SETTING UP THE CONFIGURATION FILE #######

# Counting number of nodes in the network

network_file_path = "data/network.txt"

with open(network_file_path, "r") as f:
    edges = f.readlines()

num_nodes = len(set([int(edge.split()[0]) for edge in edges] + [int(edge.split()[1]) for edge in edges])) 


@dataclass
class Config():

    vector_size: int = 768
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 32
    num_heads: int = 8
    bias: bool = False
    dropout: float = 0.0
    num_layers: int = 2
    num_nodes: int = num_nodes
    num_walks: int = 5
    walk_length: int = 80
    window_size: int = 2
    P : float = 1
    Q : float = 2   
    batch_size: int = 512
    num_neg_samples: int = 3 # Number of negative samples per positive sample
    neighborhood_size: int = 128
    neighborhood_radius: int = 2
    learning_rate: float = 1e-2
    num_epochs: int = 10

if __name__ == "__main__":
    print("Confguration file for TransformerGNN ")
    config = Config()
    print(config)
    



