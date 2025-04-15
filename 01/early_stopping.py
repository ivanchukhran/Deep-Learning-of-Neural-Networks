from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 5
    delta: float = 1e-3
    verbose: bool = False
    best_loss: float = float("inf")
    counter: int = 0

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Model improved with the best loss: {self.best_loss}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                return True
        return False
