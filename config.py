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

    vector_size: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    num_heads: int = 6
    bias: bool = False
    dropout: float = 0.0
    num_layers: int = 6
    num_nodes: int = num_nodes
    num_walks: int = 10
    walk_length: int = 80
    window_size: int = 10
    P : float = 1
    Q : float = 2   
    batch_size: int = 256
    num_neg_samples: int = 5 # Number of negative samples per positive sample

if __name__ == "__main__":
    print("Confguration file for TransformerGNN ")
    config = Config()
    print(config)
    



