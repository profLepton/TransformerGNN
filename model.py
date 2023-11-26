import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Paraneter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vector_size % config.num_heads == 0, "Number of heads must evenly divide the dimension of the input"

        self.num_heads = config.num_heads
        self.dropout = config.dropout


        self.c_attn = nn.Linear(config.vector_size, 3 * config.vector_size, bias=config.bias)
        self.c_proj = nn.Linear(config.vector_size, config.vector_size, bias=config.bias)
        

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        


    def forward(self, x, adj_matrix):
        B, T, C = x.size()

        q, k , v = self.c_attn(x).chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, adj_matrix.unsqueeze(1).expand(B, self.num_heads, T, T), dropout_p=self.dropout if self.training else 0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.vector_size, config.hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size, config.vector_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        h = self.gelu(self.c_fc(x))
        h2 = self.c_proj(self.dropout(h))
        return h2
        

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.vector_size, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.vector_size, config.bias)
        self.mlp = MLP(config)

    def forward(self, x, adj_matrix):
        x = x + self.attn(self.ln_1(x), adj_matrix)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__("GPT")

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.num_nodes, config.vecotr_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            Ln_f = LayerNorm(config.vector_size, config.bias),
        ))


    def initialize(self, title_word_embeddings, adj_matrices, neighbor_dict):
        
        self.transformer.wte.weight.data.copy_(title_word_embeddings)
        self.transformer.wte.weight.requires_grad = False
        
        self.adj_matrices = adj_matrices
        self.adj_matrices.requires_grad = False
        self.adj_matrices.float()

        self.neighbor_dict = neighbor_dict

    def forward(self, context_idx, negatives=None):

        b, t = context_idx.size()
        
        context_emb = self.transformer.wte(context_idx)
        negatives_emb = self.transformer.wte(negatives)

        neighborhood = self.neighbor_dict[context_idx[:, 0]]     
        neighborhood_emb = self.transformer.wte(neighborhood)
     
        adj_matrix = self.adj_matrices[context_idx[:, 0]]

        for block in self.transformer.h:
            neighborhood_emb = block(neighborhood_emb, adj_matrix)

        ### Save the new embeddings
        self.transformer.wte.weight.data[neighborhood] = neighborhood_emb

        first_node = neighborhood_emb[:, 0]

        n_sim = first_node @ context_emb.transpose(-1, -2)
        n_sim = n_sim.squeeze(-1)

        n_sim = F.log_softmax(n_sim, dim=-1)
        

        if negatives is None:
            return -torch.mean(n_sim, dim=-1)
        
        neg_sim = first_node @ negatives_emb.transpose(-1, -2)
        neg_sim = neg_sim.squeeze(-1)

        neg_sim = F.log_softmax(neg_sim, dim=-1)

        loss = -torch.mean(n_sim, dim=-1) - torch.mean(neg_sim, dim=-1)

        return loss
    
    def get_embeddings(self, idx):
        return self.transformer.wte(idx)


