import torch
from torch import nn

class TinyRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super(TinyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        first_layer = nn.ParameterDict({
            'W_xh': nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
            'W_hh': nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
            'b_h': nn.Parameter(torch.zeros(hidden_size))
        })
        self.layers.append(first_layer)
        for _ in range(num_layers - 1):
                    layer = nn.ParameterDict({
                        'W_xh': nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                        'W_hh': nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                        'b_h': nn.Parameter(torch.zeros(hidden_size))
                    })
                    self.layers.append(layer)

        # Output projection (from last hidden layer)
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size))


    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.size()
        if h_prev is None:
                    h_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        h_states_all_layers = []
        outputs = []
        h_current = h_prev.clone()

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t_layers = []

            for layer_idx in range(self.num_layers):
                W_xh = self.layers[layer_idx]['W_xh'] # pyright: ignore
                W_hh = self.layers[layer_idx]['W_hh'] # pyright: ignore
                b_h = self.layers[layer_idx]['b_h'] # pyright: ignore

                h_prev_layer = h_current[layer_idx]
                layer_input = x_t if layer_idx == 0 else h_t_layers[layer_idx - 1]
                h_t_layer = torch.tanh(layer_input @ W_xh + h_prev_layer @ W_hh + b_h)

                h_t_layers.append(h_t_layer)
                h_current[layer_idx] = h_t_layer

            y_t = h_t_layers[-1] @ self.W_hy + self.b_y

            h_states_all_layers.append(h_current.clone())
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        h_states = torch.stack(h_states_all_layers, dim=1)  # (num_layers, batch_size, seq_len, hidden_size)

        return outputs, h_states, h_current


class TinyLSTM(nn.Module):
    pass

class TinyGRU(nn.Module):
    pass
