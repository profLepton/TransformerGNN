import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime

from config import Config
from model import GPT

config = Config()

batches = torch.load("saved_files/train_batches.pt")
negative_batches = torch.load("saved_files/negative_sample_batches.pt")
title_word_embeddings = torch.load(f"saved_files/title_embeddings_{config.vector_size}.pt")
adj_matrices = torch.load("saved_files/adj_matrices.pt")
neighbor_dict = torch.load("saved_files/neighbor_dict.pt")

model = GPT(config)
model.initialize(title_word_embeddings, adj_matrices, neighbor_dict)
model.to(config.device)


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print("Start training...")

loss_history = []

for epoch in range(config.num_epochs):
    running_loss = 0.0
    print("Epoch:", epoch+1)
    for i in tqdm(range(len(batches))):
            
            optimizer.zero_grad(set_to_none=True)

            context = batches[i]
            negatives = negative_batches[i]

            loss = model(context)

            running_loss += loss.item()

            loss.backward()

            optimizer.step()

    loss_history.append(running_loss)
    print(f"Epoch {epoch+1} loss: {running_loss}")


current_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

if not os.path.exists("train_logs"):
    os.mkdir("train_logs")

os.mkdir(f"saved_models/model_{current_time}")

torch.save(model.state_dict(), f"saved_models/model_{current_time}/model.pt")
torch.save(config, f"saved_models/model_{current_time}/config.pt")
torch.save(loss_history, f"train_logs/loss_history_{current_time}.pt")


device = config.device

embeddings = model.transformer.wte.weight.data

train_path = "data/train.txt"


with open(train_path, "r") as f:
    train = f.readlines()


train = [line.strip().split() for line in train]

train = [[int(i) for i in line] for line in train]


val_path = "data/val.txt"

with open(val_path, "r") as f:
    val = f.readlines()


val = [line.strip().split() for line in val]

val = [[int(i) for i in line] for line in val]

embeddings = embeddings.to(device)
train_x = [embeddings[i[0]] for i in train]

train_y = [F.one_hot(torch.tensor(i[1], device=device), num_classes=26) for i in train]

val_x = [embeddings[i[0]] for i in val]

val_y = [F.one_hot(torch.tensor(i[1], device=device), num_classes=26) for i in val]

neural_net = nn.Sequential(
nn.Linear(config.vector_size, 128),
nn.ReLU(),
nn.Linear(128, 26),
#Softmax
nn.Softmax(dim=-1)
).to(device)

optimizer = torch.optim.Adam(neural_net.parameters(), lr=1e-3)

loss_fn = F.cross_entropy

train_x = torch.stack(train_x, dim=0)

train_y = torch.stack(train_y, dim=0)

val_x = torch.stack(val_x, dim=0)

val_y = torch.stack(val_y, dim=0)


for epoch in range(10):
    for i in range(len(train_x)):
        optimizer.zero_grad()
        pred = neural_net(train_x[i])
        loss = loss_fn(pred.float(), train_y[i].float())
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")


pred = neural_net(val_x)

pred = torch.argmax(pred, dim=1)

val_y = torch.argmax(val_y, dim=1)

print(f"Accuracy {torch.sum(pred == val_y).item() / len(val_y)}")



print("Training complete!")
