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
    n_neurons_list = [32, 128, 512]
    stats = {neurons: {} for neurons in n_neurons_list}
    for n_neurons in n_neurons_list:
        print("Neurons:", n_neurons)
        n_layers = 8
        activation = nn.ReLU()
        n_epochs = 10

        model = build_model(n_layers, n_neurons, activation)
        optimizer = Adam(model.parameters(), lr=3e-4)

        train_dataset, test_dataset = get_dataset(dataset_path="/home/ivan/datasets")
        train_loader, test_loader = (
            DataLoader(train_dataset),
            DataLoader(test_dataset),
        )
        model_stats = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=n_epochs,
            device=device,
        )

        stats[n_neurons] = model_stats
    with open("01/experiments/results/01_neurons_stats.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()
