import os
import random
from multiprocessing import Pool


import torch
import numpy as np
from tqdm import tqdm
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from config import Config


### Loads title word embeddings with correct size if available, else creates them.

if not os.path.exists("saved_files"):
    os.mkdir("saved_files")


config = Config()

file_name = f"saved_files/title_embeddings_{config.vector_size}.pt"



G = nx.read_edgelist("./data/network.txt", nodetype=int)

# Loading all titles
title_file_path = "data/titles.txt"

with open(title_file_path) as f:
    titles = f.readlines()

titles = [x.strip().split() for x in titles]

title_dict = {}

for node in titles:
    title_dict[node[0]] = node[1]



try:

    title_word_embeddings = torch.load(file_name)
    title_word_embeddings = title_word_embeddings.to(config.device)
    print("Loaded title word embeddings")

except:

    print("Getting title vectors...")
    def labelled_sentences(node_dict):

        sentences = []
        
        for node in tqdm(node_dict.keys()):
            sentences.append(TaggedDocument(node_dict[node].split(), [node]))

        return sentences

    
    sentences = labelled_sentences(title_dict)

    model_dbow = Doc2Vec(sentences, vector_size=config.vector_size, window=2, min_count=1, workers=config.num_workers)
    model_dbow.build_vocab(sentences)

    for epoch in range(100):
        model_dbow.train(sentences, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha


    def get_vectors(node_dict, model):
        vectors = []
        for node in tqdm(node_dict.keys()):
            vectors.append(model.dv[node])

        return vectors

    title_vectors = get_vectors(title_dict, model_dbow)

    # Save vectors

    title_word_embeddings = torch.tensor(title_vectors).to(config.device)
   
    torch.save(title_word_embeddings, file_name)

    print("Saved title word embeddings")



def get_walks_np(nodes, num_walks, walk_length, p, q):
    walks = np.empty((num_walks * len(nodes), walk_length), dtype=np.int16)
    random.shuffle(nodes)

    for n, node in enumerate(nodes):
        for i in range(num_walks):
            walk = np.empty(walk_length, dtype=np.int16)
            walk[0] = node

            for j in range(walk_length-1):
                curr = walk[j]
                curr_nbrs = list(G.neighbors(curr))
                if len(curr_nbrs) > 0:
                    if j == 0:
                        walk[j+1] = random.choice(curr_nbrs)
                    else:
                        prev = walk[j-1]
                        prob = []
                        for nbr in curr_nbrs:
                            if nbr == prev:
                                prob.append(1/p)
                            elif G.has_edge(nbr, prev):
                                prob.append(1)
                            else:
                                prob.append(1/q)
                        prob = [float(i)/sum(prob) for i in prob]
                        walk[j+1] = np.random.choice(curr_nbrs,p=prob)

                else:
                    break
            walks[n*config.num_walks + i] = walk

    return walks



def get_walks_parallel_np(G, num_walks, walk_length, p, q, num_workers):
    
    all_nodes = list(G.nodes())

    nodes_split = [all_nodes[i:i+len(all_nodes)//num_workers] for i in range(len(all_nodes) // num_workers) ]

    args = [(nodes_split[i], num_walks, walk_length, p, q) for i in range(num_workers)]

    with Pool(num_workers) as pool:
        walks = pool.starmap(get_walks_np, args)

    # Concat all np arrays
    walks = np.concatenate(walks, axis=0)

    return walks


walk_file_name = f"saved_files/walks_{config.num_walks}_{config.walk_length}_{config.P}_{config.Q}.pt"

try:
    walks = torch.load(walk_file_name)
    print("Loaded walks")

except:
    print("Generating walks")
    # Generate walks
    

    walks = get_walks_parallel_np(G, config.num_walks, config.walk_length, p=config.P, q=config.Q, num_workers=config.num_workers)

    torch.save(walks, walk_file_name)

    print("Saved walks")

def get_x_train( i, j, window_size=2, n_samples=5):

    x_train = []
    context = [walk[i:j] for walk in walks]
    x_train.extend(context)

    return x_train


def get_x_train_parallel(walks, window_size=config.window_size, num_workers=config.num_workers):
    x_train = []
    indices =  [(i-window_size, i+window_size) for i in range(window_size, config.walk_length-window_size)]
    with Pool(num_workers) as pool:
        x_train = pool.starmap(get_x_train, indices)

    x_train = [x for x_per_worker in x_train for x in x_per_worker]

    return np.array(x_train)

x_train = get_x_train_parallel(walks, window_size=config.window_size)


print("X train acquired")
print("x_train length: ", len(x_train))


def get_random_negative_samples(walks, num_samples, sample_size):
    negative_samples = []
    
    negative_samples = np.random.choice(list(G.nodes()), size=(num_samples, sample_size), replace=True)


    return negative_samples

negative_samples = get_random_negative_samples(None, x_train.shape[0], config.num_neg_samples)


def get_batches(x_train, negative_samples, batch_size=config.batch_size):
    
        #Concat all the tensors in x_train across batch dimension
        # Retain originals tructure of x_train, but each element is a batch of size batch_size
    
    x_train = torch.tensor(x_train, dtype=torch.int32,device=config.device)

    x_train = torch.split(x_train, batch_size, dim=0)

    negative_samples = torch.tensor(negative_samples, dtype=torch.int32, device=config.device)

    negative_samples = torch.split(negative_samples, batch_size, dim=0)

    return x_train, negative_samples


train_batches, negative_sample_batches = get_batches(x_train, negative_samples, batch_size=config.batch_size)

neighbor_dict = {}

def get_radius_i_neighbors(G, node, radius):
    neighbors = [node]
    for i in range(radius):
        neighbors.extend([nbr for n in neighbors for nbr in G.neighbors(n)])
    return list(set(neighbors))

maax = 0
for node in tqdm(G.nodes()):

    neighborhood_tensor = torch.tensor(get_radius_i_neighbors(G, node, config.neighborhood_radius)[:config.neighborhood_size], dtype=torch.int32, device=config.device, requires_grad=False)

    if len(neighborhood_tensor) < config.neighborhood_size:
        # Pad with config.num_nodes
        neighborhood_tensor = torch.cat([neighborhood_tensor, torch.ones(config.neighborhood_size-len(neighborhood_tensor), dtype=torch.int32, device=config.device, requires_grad=False)*config.num_nodes])

    neighbor_dict[node] = neighborhood_tensor.unsqueeze(0) 

    maax = max(maax, len(neighbor_dict[node]))

print(f"max neighbors: {maax}")

adj_dict = {}

for node in tqdm(G.nodes()):
    sg = G.subgraph(neighbor_dict[node].squeeze(0).cpu().numpy())
    adj = nx.adjacency_matrix(sg).todense()

    if adj.shape[0] < config.neighborhood_size:
        # Pad with only zeros
        adj = np.pad(adj, (0, config.neighborhood_size-adj.shape[0]),  'constant', constant_values=0)

    adj_dict[node] = torch.tensor(adj + np.eye(adj.shape[0]), dtype=torch.float32, device=config.device, requires_grad=False)

adj_matrices = torch.stack(list(adj_dict.values()), dim=0)

print(f"Number of batches: ", len(train_batches))

torch.save(train_batches, "saved_files/train_batches.pt")
torch.save(negative_sample_batches, "saved_files/negative_sample_batches.pt")

torch.save(neighbor_dict, "saved_files/neighbor_dict.pt")
torch.save(adj_matrices, "saved_files/adj_matrices.pt")