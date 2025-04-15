import json
import os
import sys

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from early_stopping import EarlyStopping
from model import build_model
from prepare_dataset import get_dataset
from train import train


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    activation = nn.ReLU()
    n_layers = 2
    n_neurons = 32
    n_epochs = 100
    batch_size = 512
    patience_list = [10, 30, 50]
    stats = {}
    for patience in patience_list:
        print(f"Patience: {patience}")
        es = EarlyStopping(patience=patience, delta=1e-2, verbose=True)
        model = build_model(n_layers, n_neurons, activation).to(device)
        optimizer = Adam(model.parameters(), lr=3e-4)

        train_dataset, test_dataset = get_dataset(dataset_path="/home/ivan/datasets")
        train_loader, test_loader = (DataLoader(train_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size))

        model_stats = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=n_epochs,
            device=device,
            early_stopping=es,
        )

        stats[optimizer.__class__.__name__] = model_stats

    with open("01/experiments/results/05_early_stopping.json", "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    main()
