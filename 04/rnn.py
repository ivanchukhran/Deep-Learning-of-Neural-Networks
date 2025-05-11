import torch
from torch import nn


class TinyRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ):
        super(TinyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        first_layer = nn.ParameterDict(
            {
                "W_xh": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_h": nn.Parameter(torch.zeros(hidden_size)),
            }
        )
        self.layers.append(first_layer)
        for _ in range(num_layers - 1):
            layer = nn.ParameterDict(
                {
                    "W_xh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_h": nn.Parameter(torch.zeros(hidden_size)),
                }
            )
            self.layers.append(layer)

        # Output projection (from last hidden layer)
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.size()
        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )
        h_states_all_layers = []
        outputs = []
        h_current = h_prev.clone()

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t_layers = []

            for layer_idx in range(self.num_layers):
                W_xh = self.layers[layer_idx]["W_xh"]  # pyright: ignore
                W_hh = self.layers[layer_idx]["W_hh"]  # pyright: ignore
                b_h = self.layers[layer_idx]["b_h"]  # pyright: ignore

                h_prev_layer = h_current[layer_idx]
                layer_input = x_t if layer_idx == 0 else h_t_layers[layer_idx - 1]
                h_t_layer = torch.tanh(layer_input @ W_xh + h_prev_layer @ W_hh + b_h)

                h_t_layers.append(h_t_layer)
                h_current[layer_idx] = h_t_layer

            y_t = h_t_layers[-1] @ self.W_hy + self.b_y

            h_states_all_layers.append(h_current.clone())
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        h_states = torch.stack(
            h_states_all_layers, dim=1
        )  # (num_layers, batch_size, seq_len, hidden_size)

        return outputs, h_states, h_current


class TinyLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ):
        super(TinyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        first_layer = nn.ParameterDict(
            {
                # Input gate parameters
                "W_xi": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hi": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_i": nn.Parameter(torch.zeros(hidden_size)),
                # Forget gate parameters
                "W_xf": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hf": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_f": nn.Parameter(torch.zeros(hidden_size)),
                # Output gate parameters
                "W_xo": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_ho": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_o": nn.Parameter(torch.zeros(hidden_size)),
                # Cell state candidate parameters
                "W_xc": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hc": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_c": nn.Parameter(torch.zeros(hidden_size)),
            }
        )
        self.layers.append(first_layer)

        for _ in range(self.num_layers - 1):
            layer = nn.ParameterDict(
                {
                    # Input gate parameters
                    "W_xi": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hi": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_i": nn.Parameter(torch.zeros(hidden_size)),
                    # Forget gate parameters
                    "W_xf": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hf": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_f": nn.Parameter(torch.zeros(hidden_size)),
                    # Output gate parameters
                    "W_xo": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_ho": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_o": nn.Parameter(torch.zeros(hidden_size)),
                    # Cell state candidate parameters
                    "W_xc": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hc": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_c": nn.Parameter(torch.zeros(hidden_size)),
                }
            )
            self.layers.append(layer)

        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor, states: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        if states is None:
            h_states = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
            c_states = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )
        else:
            h_states, c_states = states

        h_seq = []
        c_seq = []
        outputs = []

        h_current = h_states.clone()
        c_current = c_states.clone()

        for t in range(seq_len):
            x_t = x[:, t, :]  # Current input: (batch_size, input_size)

            for layer_idx in range(self.num_layers):
                layer = self.layers[layer_idx]

                layer_input = x_t if layer_idx == 0 else h_current[layer_idx - 1]

                h_prev = h_current[layer_idx]
                c_prev = c_current[layer_idx]

                i_t = torch.sigmoid(
                    layer_input @ layer["W_xi"] + h_prev @ layer["W_hi"] + layer["b_i"]  # pyright: ignore
                )
                f_t = torch.sigmoid(
                    layer_input @ layer["W_xf"] + h_prev @ layer["W_hf"] + layer["b_f"]  # pyright: ignore
                )
                o_t = torch.sigmoid(
                    layer_input @ layer["W_xo"] + h_prev @ layer["W_ho"] + layer["b_o"]  # pyright: ignore
                )
                g_t = torch.tanh(
                    layer_input @ layer["W_xc"] + h_prev @ layer["W_hc"] + layer["b_c"]  # pyright: ignore
                )

                c_current[layer_idx] = f_t * c_prev + i_t * g_t
                h_current[layer_idx] = o_t * torch.tanh(c_current[layer_idx])

            y_t = h_current[-1] @ self.W_hy + self.b_y

            h_seq.append(h_current.clone())
            c_seq.append(c_current.clone())
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        h_seq_stacked = torch.stack(
            h_seq, dim=1
        )  # (num_layers, batch_size, seq_len, hidden_size)
        c_seq_stacked = torch.stack(
            c_seq, dim=1
        )  # (num_layers, batch_size, seq_len, hidden_size)

        return outputs, (h_seq_stacked, c_seq_stacked), (h_current, c_current)


class TinyGRU(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ):
        super(TinyGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        first_layer = nn.ParameterDict(
            {
                "W_xz": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hz": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_z": nn.Parameter(torch.zeros(hidden_size)),
                # Reset gate parameters
                "W_xr": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hr": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_r": nn.Parameter(torch.zeros(hidden_size)),
                # Candidate hidden state parameters
                "W_xh": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
                "W_hh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                "b_h": nn.Parameter(torch.zeros(hidden_size)),
            }
        )

        self.layers.append(first_layer)

        for _ in range(num_layers - 1):
            next_layer = nn.ParameterDict(
                {
                    "W_xz": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hz": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_z": nn.Parameter(torch.zeros(hidden_size)),
                    # Reset gate parameters
                    "W_xr": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hr": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_r": nn.Parameter(torch.zeros(hidden_size)),
                    # Candidate hidden state parameters
                    "W_xh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "W_hh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
                    "b_h": nn.Parameter(torch.zeros(hidden_size)),
                }
            )
            self.layers.append(next_layer)

            self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
            self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None):
        batch_size, seq_len, _ = x.size()
        device = x.device
        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device
            )

        h_seq = []
        outputs = []

        h_current = h_prev.clone()

        for t in range(self.num_layers):
            x_t = x[:, t, :]

            for layer_idx in range(self.num_layers):
                layer = self.layers[layer_idx]
                layer_input = x_t if layer_idx == 0 else h_current[layer_idx - 1]
                h_prev_layer = h_current[layer_idx]
                z_t = torch.sigmoid(
                    layer_input @ layer["W_xz"] # pyright: ignore
                    + h_prev_layer @ layer["W_hz"] # pyright: ignore
                    + layer["b_z"] # pyright: ignore
                )
                r_t = torch.sigmoid(
                    layer_input @ layer["W_xr"] # pyright: ignore
                    + h_prev_layer @ layer["W_hr"] # pyright: ignore
                    + layer["b_r"] # pyright: ignore
                )
                h_tilde = torch.tanh(
                    layer_input @ layer["W_xh"] # pyright: ignore
                    + (r_t * h_prev_layer) @ layer["W_hh"] # pyright: ignore
                    + layer["b_h"] # pyright: ignore
                )
                h_current[layer_idx] = (1 - z_t) * h_prev_layer + z_t * h_tilde

            y_t = h_current[-1] @ self.W_hy + self.b_y
            h_seq.append(h_current.clone())
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
        h_seq_stacked = torch.stack(
            h_seq, dim=1
        )  # (num_layers, batch_size, seq_len, hidden_size)

        return outputs, h_seq_stacked, h_current
