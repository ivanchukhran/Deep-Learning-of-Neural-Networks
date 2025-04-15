import json
import os

import matplotlib.pyplot as plt
import numpy as np

results_dir = os.path.join("01", "experiments", "results")

# Load JSON data files
with open(os.path.join(results_dir, "01_layers_stats.json"), "r") as f:
    layers_data = json.load(f)

with open(os.path.join(results_dir, "01_neurons_stats.json"), "r") as f:
    neurons_data = json.load(f)

with open(os.path.join(results_dir, "02_activation_stats.json"), "r") as f:
    activation_data = json.load(f)

with open(os.path.join(results_dir, "03_minibatches_stats.json"), "r") as f:
    minibatches_data = json.load(f)

with open(os.path.join(results_dir, "04_dropout_stats.json"), "r") as f:
    dropout_data = json.load(f)

with open(os.path.join(results_dir, "04_l2_stats.json"), "r") as f:
    l2_data = json.load(f)

with open(os.path.join(results_dir, "05_early_stopping.json"), "r") as f:
    early_stopping_data = json.load(f)

with open(os.path.join(results_dir, "05_optimizers.json"), "r") as f:
    optimizers_data = json.load(f)

# Set up style and figure size for better visualization
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


# Function to create a training-testing curve plot
def plot_train_test_curves(data, title, xlabel="Epochs", ylabel="Loss/Accuracy", legend_loc="best"):
    epochs = range(1, len(data[list(data.keys())[0]]["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set a colormap to differentiate between configurations
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))

    # Plot loss
    for i, (config, metrics) in enumerate(data.items()):
        ax1.plot(epochs, metrics["train_loss"], "o-", color=colors[i], label=f"{config} (Train)")
        ax1.plot(epochs, metrics["test_loss"], "s--", color=colors[i], alpha=0.7, label=f"{config} (Test)")

    # Plot accuracy
    for i, (config, metrics) in enumerate(data.items()):
        ax2.plot(epochs, metrics["train_accuracy"], "o-", color=colors[i], label=f"{config} (Train)")
        ax2.plot(epochs, metrics["test_accuracy"], "s--", color=colors[i], alpha=0.7, label=f"{config} (Test)")

    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Loss")
    ax1.legend(loc=legend_loc)
    ax1.grid(True)

    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc=legend_loc)
    ax2.grid(True)

    plt.tight_layout()
    return fig


# 1. Number of Layers Experiment
result_img_dir = os.path.join(results_dir, "img")

if not os.path.exists(result_img_dir):
    os.mkdir(result_img_dir)

layers_fig = plot_train_test_curves(layers_data, "Effect of Number of Layers")
plt.savefig(os.path.join(result_img_dir, "01_layers_comparison.png"))

# 2. Number of Neurons Experiment
neurons_fig = plot_train_test_curves(neurons_data, "Effect of Number of Neurons")
plt.savefig(os.path.join(result_img_dir, "01_neurons_comparison.png"))

# 3. Activation Functions Experiment
activation_fig = plot_train_test_curves(activation_data, "Effect of Activation Functions")
plt.savefig(os.path.join(result_img_dir, "02_activation_comparison.png"))

# 4. Mini-batch Size Experiment
minibatches_fig = plot_train_test_curves(minibatches_data, "Effect of Mini-batch Size")
plt.savefig(os.path.join(result_img_dir, "03_minibatch_comparison.png"))

# 5. Dropout Rate Experiment
dropout_fig = plot_train_test_curves(dropout_data, "Effect of Dropout Rate")
plt.savefig(os.path.join(result_img_dir, "04_dropout_comparison.png"))

# 6. L2 Regularization Experiment
l2_fig = plot_train_test_curves(l2_data, "Effect of L2 Regularization")
plt.savefig(os.path.join(result_img_dir, "04_l2_comparison.png"))

# 7. Optimizers Comparison
optimizers_fig = plot_train_test_curves(optimizers_data, "Comparison of Optimizers")
plt.savefig(os.path.join(result_img_dir, "05_optimizers_comparison.png"))


# 8. Early Stopping Analysis
# Since early_stopping_data has more epochs, let's create a special plot
def plot_early_stopping():
    data = early_stopping_data["Adam"]
    epochs = range(1, len(data["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot loss with potential stopping point
    ax1.plot(epochs, data["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, data["test_loss"], "s-", label="Validation Loss")

    # Find where validation loss starts to increase (potential early stopping point)
    val_losses = data["test_loss"]
    min_idx = np.argmin(val_losses)

    # Mark the potential early stopping point
    ax1.axvline(x=min_idx + 1, color="r", linestyle="--", label=f"Potential Early Stop (Epoch {min_idx + 1})")

    # Plot accuracy
    ax2.plot(epochs, data["train_accuracy"], "o-", label="Train Accuracy")
    ax2.plot(epochs, data["test_accuracy"], "s-", label="Validation Accuracy")
    ax2.axvline(x=min_idx + 1, color="r", linestyle="--", label=f"Potential Early Stop (Epoch {min_idx + 1})")

    ax1.set_title("Loss with Early Stopping Analysis")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Accuracy with Early Stopping Analysis")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig


early_stopping_fig = plot_early_stopping()
plt.savefig(os.path.join(result_img_dir, "05_early_stopping_analysis.png"))

# 9. Create a comprehensive final comparison of the best configurations


# First, determine the best configuration from each experiment based on test accuracy
def get_best_config(data):
    best_config = None
    best_acc = 0

    for config, metrics in data.items():
        final_acc = metrics["test_accuracy"][-1]
        if final_acc > best_acc:
            best_acc = final_acc
            best_config = config

    return best_config, data[best_config]


best_layer, best_layer_metrics = get_best_config(layers_data)
best_neuron, best_neuron_metrics = get_best_config(neurons_data)
best_activation, best_activation_metrics = get_best_config(activation_data)
best_batch, best_batch_metrics = get_best_config(minibatches_data)
best_dropout, best_dropout_metrics = get_best_config(dropout_data)
best_l2, best_l2_metrics = get_best_config(l2_data)
best_optimizer, best_optimizer_metrics = get_best_config(optimizers_data)

# Create a summary plot
plt.figure(figsize=(10, 8))
epochs = range(1, 11)  # Assuming all have 10 epochs

# Plot test accuracies of the best configurations
plt.plot(epochs, best_layer_metrics["test_accuracy"], "o-", label=f"Best Layers: {best_layer}")
plt.plot(epochs, best_neuron_metrics["test_accuracy"], "s-", label=f"Best Neurons: {best_neuron}")
plt.plot(epochs, best_activation_metrics["test_accuracy"], "^-", label=f"Best Activation: {best_activation}")
plt.plot(epochs, best_batch_metrics["test_accuracy"], "d-", label=f"Best Batch Size: {best_batch}")
plt.plot(epochs, best_dropout_metrics["test_accuracy"], "p-", label=f"Best Dropout: {best_dropout}")
plt.plot(epochs, best_l2_metrics["test_accuracy"], "h-", label=f"Best L2 Reg: {best_l2}")
plt.plot(epochs, best_optimizer_metrics["test_accuracy"], "*-", label=f"Best Optimizer: {best_optimizer}")

plt.title("Comparison of Best Configurations (Test Accuracy)")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_img_dir, "best_configurations_comparison.png"))

# 10. Create a summary table of final test accuracies and losses
summary_data = {
    "Configuration": [
        "Best Number of Layers",
        "Best Number of Neurons",
        "Best Activation Function",
        "Best Mini-batch Size",
        "Best Dropout Rate",
        "Best L2 Regularization",
        "Best Optimizer",
    ],
    "Best Value": [best_layer, best_neuron, best_activation, best_batch, best_dropout, best_l2, best_optimizer],
    "Final Test Accuracy": [
        best_layer_metrics["test_accuracy"][-1],
        best_neuron_metrics["test_accuracy"][-1],
        best_activation_metrics["test_accuracy"][-1],
        best_batch_metrics["test_accuracy"][-1],
        best_dropout_metrics["test_accuracy"][-1],
        best_l2_metrics["test_accuracy"][-1],
        best_optimizer_metrics["test_accuracy"][-1],
    ],
    "Final Test Loss": [
        best_layer_metrics["test_loss"][-1],
        best_neuron_metrics["test_loss"][-1],
        best_activation_metrics["test_loss"][-1],
        best_batch_metrics["test_loss"][-1],
        best_dropout_metrics["test_loss"][-1],
        best_l2_metrics["test_loss"][-1],
        best_optimizer_metrics["test_loss"][-1],
    ],
}

# Print the summary table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("tight")
ax.axis("off")
table = ax.table(
    cellText=[
        summary_data["Configuration"],
        summary_data["Best Value"],
        [f"{acc:.4f}" for acc in summary_data["Final Test Accuracy"]],
        [f"{loss:.4f}" for loss in summary_data["Final Test Loss"]],
    ],
    rowLabels=["Configuration", "Best Value", "Final Test Accuracy", "Final Test Loss"],
    colLabels=range(1, len(summary_data["Configuration"]) + 1),
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.title("Summary of Best Configurations", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(result_img_dir, "summary_table.png"))

print("All plots have been created and saved.")
