import json
import os
import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import build_model
from prepare_dataset import get_dataset
from train import train


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activation = nn.ReLU()
    n_layers = 8
    n_neurons = 128
    n_epochs = 10
    batch_size = 512
    # weight_decay = 1e-5  # for l2-regularization
    weight_decays = [1e-3, 1e-2]
    stats = {}
    for weight_decay in weight_decays:
        model = build_model(n_layers, n_neurons, activation).to(device)
        optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay)

        train_dataset, test_dataset = get_dataset(dataset_path="/home/ivan/datasets")
        train_loader, test_loader = (
            DataLoader(train_dataset, batch_size=batch_size),
            DataLoader(test_dataset, batch_size),
        )

        model_stats = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=n_epochs,
            device=device,
        )

        stats[weight_decay] = model_stats

    with open("01/experiments/results/04_l2_stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()
