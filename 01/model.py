from torch import nn


def build_model(
    n_layers: int, n_neurons: int, activation: nn.Module, dropout: float | None = None
) -> nn.Module:
    model = nn.ModuleList()
    input_dim = 28 * 28  # flattened 28x28 images
    output_dim = 10  # 10 classes

    model.append(nn.Linear(input_dim, n_neurons))

    current_width = n_neurons
    for i in range(1, n_layers - 1):
        next_width = current_width // 2
        if next_width < 10:
            next_width = 10
        model.append(nn.Linear(current_width, next_width))
        model.append(activation)
        if dropout is not None:
            model.append(nn.Dropout(dropout))
        current_width = next_width

    model.append(nn.Linear(current_width, output_dim))
    return nn.Sequential(*model)
