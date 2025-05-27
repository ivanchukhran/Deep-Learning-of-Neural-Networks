from typing import Callable

import torch
from dataset import ShakespeareDataset
from rnn import TextGeneratorLSTM
from tokenizer import TextTokenizer
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_perplexity(model, data_loader, device):
    """Calculate perplexity on the dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Test batches"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)

            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module | Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: TextTokenizer,
    epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
):
    """Training loop for the LSTM text generation model"""

    train_losses = []
    val_perplexities = []

    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0
        num_batches = 0

        train_bar = tqdm(train_loader, desc="Train batches")

        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs, _ = model(inputs)

            outputs = outputs.view(
                -1, outputs.size(-1)
            )  # (batch_size * seq_length, vocab_size)
            targets = targets.view(-1)  # (batch_size * seq_length)

            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Print progress every 100 batches
            # if batch_idx % 100 == 0:
            #     print(
            #         f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
            #     )
            train_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        val_perplexity = calculate_perplexity(model, val_loader, device)
        val_perplexities.append(val_perplexity)

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{epochs} Train Loss: {avg_train_loss:.4f}  Val Perplexity: {val_perplexity:.2f}"
        )

        if (epoch + 1) % 5 == 0:
            print("\nSample generation:")
            sample_text = model.generate_text(
                tokenizer,
                "first citizen:",
                max_length=50,
                temperature=0.8,
                strategy="temperature",
                device=device,
            )
            print(f"Generated: {sample_text}\n")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_perplexity": val_perplexity,
                },
                f"model_checkpoint_epoch_{epoch + 1}.pt",
            )

    return train_losses, val_perplexities


# Example usage and main training script
if __name__ == "__main__":
    shakespeare_text = ""
    with open("data/shakespeare_input.txt", "r") as f:
        shakespeare_text = f.read()

    # Hyperparameters
    seq_length = 30
    batch_size = 32
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    dropout = 0.1
    learning_rate = 0.003
    epochs = 10

    # Create tokenizer and build vocabulary
    tokenizer = TextTokenizer(min_freq=1)  # Low min_freq for small dataset
    tokenizer.build_vocab([shakespeare_text])

    # Create dataset
    dataset = ShakespeareDataset(
        text=shakespeare_text, tokenizer=tokenizer, seq_length=seq_length
    )

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create model
    model = TextGeneratorLSTM(
        vocab_size=tokenizer.vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Train the model
    train_losses, val_perplexities = train_model(
        model, optimizer, criterion, train_loader, val_loader, tokenizer, epochs=epochs
    )

    # Test different generation strategies
    print("\n" + "=" * 50)
    print("TESTING DIFFERENT GENERATION STRATEGIES")
    print("=" * 50)

    test_prompts = ["first citizen:", "all:", "second citizen:"]
    strategies = ["greedy", "temperature"]
    temperatures = [0.5, 1.0, 1.5]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 30)

        for strategy in strategies:
            if strategy == "greedy":
                generated = model.generate_text(
                    tokenizer, prompt, max_length=30, strategy="greedy", device="cuda"
                )
                print(f"Greedy: {generated}")
            else:
                for temp in temperatures:
                    generated = model.generate_text(
                        tokenizer,
                        prompt,
                        max_length=30,
                        temperature=temp,
                        strategy="temperature",
                        device="cuda",
                    )
                    print(f"Temp {temp}: {generated}")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer,
            "hyperparameters": {
                "vocab_size": tokenizer.vocab_size,
                "embed_size": embed_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "seq_length": seq_length,
            },
        },
        "shakespeare_lstm_final.pt",
    )

    print("\nTraining completed! Model saved as 'shakespeare_lstm_final.pt'")
