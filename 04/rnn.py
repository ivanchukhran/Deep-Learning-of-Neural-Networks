from typing import Tuple

import torch
from torch import nn


class BaseRecurrent(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, gate_size: int = 1
    ):
        super(BaseRecurrent, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_size = gate_size
        self.num_layers = num_layers


class TinyRNN(BaseRecurrent):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            w_ih = nn.Parameter(
                torch.rand(layer_input_size, self.gate_size * hidden_size) * 0.1
            )
            w_hh = nn.Parameter(
                torch.rand(hidden_size, self.gate_size * hidden_size) * 0.1
            )
            b_ih = nn.Parameter(torch.rand(self.gate_size * hidden_size))
            b_hh = nn.Parameter(torch.rand(self.gate_size * hidden_size))

            setattr(self, f"w_ih_{layer}", w_ih)
            setattr(self, f"w_hh_{layer}", w_hh)
            setattr(self, f"b_ih_{layer}", b_ih)
            setattr(self, f"b_hh_{layer}", b_hh)

    def forward(
        self, x: torch.Tensor, hx: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = x.size()
        if hx is None:
            hx = torch.zeros(
                self.num_layers,
                batch_size,
                self.gate_size * self.hidden_size,
                device=x.device,
            )
        h_layers = torch.unbind(hx, dim=0)
        outputs = []
        layer_input = x

        for t in range(seq_length):
            x_t = layer_input[:, t, :]
            new_h_layers = []

            for layer in range(self.num_layers):
                w_ih = getattr(self, f"w_ih_{layer}")
                w_hh = getattr(self, f"w_hh_{layer}")
                b_ih = getattr(self, f"b_ih_{layer}")
                b_hh = getattr(self, f"b_hh_{layer}")

                h_prev = h_layers[layer]
                h_new = torch.tanh(x_t @ w_ih + h_prev @ w_hh + b_ih + b_hh)
                new_h_layers.append(h_new)
                x_t = h_new

            h_layers = new_h_layers
            outputs.append(h_layers[-1].unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        final_hidden = torch.stack(h_layers, dim=0)

        return output, final_hidden


class TinyLSTM(BaseRecurrent):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gate_size=4,
        )
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            w_ih = nn.Parameter(
                torch.rand(layer_input_size, self.gate_size * hidden_size) * 0.1
            )
            w_hh = nn.Parameter(
                torch.rand(hidden_size, self.gate_size * hidden_size) * 0.1
            )
            b_ih = nn.Parameter(torch.rand(self.gate_size * hidden_size))
            b_hh = nn.Parameter(torch.rand(self.gate_size * hidden_size))

            setattr(self, f"w_ih_{layer}", w_ih)
            setattr(self, f"w_hh_{layer}", w_hh)
            setattr(self, f"b_ih_{layer}", b_ih)
            setattr(self, f"b_hh_{layer}", b_hh)

    def forward(
        self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] | None = None
    ):
        batch_size, seq_length, _ = x.size()
        if hx is None:
            h_zeros = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )
            c_zeros = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )
        else:
            h_zeros, c_zeros = hx

        h_layers = torch.unbind(h_zeros, dim=0)
        c_layers = torch.unbind(c_zeros, dim=0)

        outputs = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            new_h_layers, new_c_layers = [], []

            for layer in range(self.num_layers):
                w_ih = getattr(self, f"w_ih_{layer}")
                w_hh = getattr(self, f"w_hh_{layer}")
                b_ih = getattr(self, f"b_ih_{layer}")
                b_hh = getattr(self, f"b_hh_{layer}")

                h_prev = h_layers[layer]
                c_prev = c_layers[layer]

                h_new, c_new = self.lstm_cell(
                    x_t, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh
                )

                new_h_layers.append(h_new)
                new_c_layers.append(c_new)

                x_t = h_new

            h_layers = new_h_layers
            c_layers = new_c_layers

            outputs.append(h_layers[-1].unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        final_hidden = torch.stack(h_layers, dim=0)
        final_cell = torch.stack(c_layers, dim=0)

        return output, (final_hidden, final_cell)

    def lstm_cell(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor,
        b_hh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gi = x_t @ w_ih + b_ih
        gh = h_prev @ w_hh + b_hh
        i_r, i_i, i_n, i_o = gi.chunk(4, 1)
        h_r, h_i, h_n, h_o = gh.chunk(4, 1)

        forget_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        candidate_gate = torch.sigmoid(i_n + h_n)
        output_gate = torch.sigmoid(i_o + h_o)

        c_new = forget_gate * c_prev + input_gate * candidate_gate
        h_new = output_gate * torch.tanh(c_new)

        return h_new, c_new


# class TinyGRU(BaseRecurrent):
#     def __init__(
#         self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
#     ):
#         super(TinyGRU, self).__init__(hidden_size=hidden_size, num_layers=num_layers)

#         first_layer = nn.ParameterDict(
#             {
#                 "W_xz": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
#                 "W_hz": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                 "b_z": nn.Parameter(torch.zeros(hidden_size)),
#                 # Reset gate parameters
#                 "W_xr": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
#                 "W_hr": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                 "b_r": nn.Parameter(torch.zeros(hidden_size)),
#                 # Candidate hidden state parameters
#                 "W_xh": nn.Parameter(torch.randn(input_size, hidden_size) * 0.01),
#                 "W_hh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                 "b_h": nn.Parameter(torch.zeros(hidden_size)),
#             }
#         )

#         self.layers.append(first_layer)

#         for _ in range(num_layers - 1):
#             next_layer = nn.ParameterDict(
#                 {
#                     "W_xz": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                     "W_hz": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                     "b_z": nn.Parameter(torch.zeros(hidden_size)),
#                     # Reset gate parameters
#                     "W_xr": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                     "W_hr": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                     "b_r": nn.Parameter(torch.zeros(hidden_size)),
#                     # Candidate hidden state parameters
#                     "W_xh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                     "W_hh": nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01),
#                     "b_h": nn.Parameter(torch.zeros(hidden_size)),
#                 }
#             )
#             self.layers.append(next_layer)

#             self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
#             self.b_y = nn.Parameter(torch.zeros(output_size))

#     def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None):
#         batch_size, seq_len, _ = x.size()
#         device = x.device
#         if h_prev is None:
#             h_prev = torch.zeros(
#                 self.num_layers, batch_size, self.hidden_size, device=device
#             )

#         h_seq = []
#         outputs = []

#         h_current = h_prev.clone()

#         for t in range(self.num_layers):
#             x_t = x[:, t, :]

#             for layer_idx in range(self.num_layers):
#                 layer = self.layers[layer_idx]
#                 layer_input = x_t if layer_idx == 0 else h_current[layer_idx - 1]
#                 h_prev_layer = h_current[layer_idx]
#                 z_t = torch.sigmoid(
#                     layer_input @ layer["W_xz"]  # pyright: ignore
#                     + h_prev_layer @ layer["W_hz"]  # pyright: ignore
#                     + layer["b_z"]  # pyright: ignore
#                 )
#                 r_t = torch.sigmoid(
#                     layer_input @ layer["W_xr"]  # pyright: ignore
#                     + h_prev_layer @ layer["W_hr"]  # pyright: ignore
#                     + layer["b_r"]  # pyright: ignore
#                 )
#                 h_tilde = torch.tanh(
#                     layer_input @ layer["W_xh"]  # pyright: ignore
#                     + (r_t * h_prev_layer) @ layer["W_hh"]  # pyright: ignore
#                     + layer["b_h"]  # pyright: ignore
#                 )
#                 h_current[layer_idx] = (1 - z_t) * h_prev_layer + z_t * h_tilde

#             y_t = h_current[-1] @ self.W_hy + self.b_y
#             h_seq.append(h_current.clone())
#             outputs.append(y_t)

#         outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, output_size)
#         h_seq_stacked = torch.stack(
#             h_seq, dim=1
#         )  # (num_layers, batch_size, seq_len, hidden_size)

#         return outputs, h_seq_stacked, h_current


if __name__ == "__main__":
    input_size = 3
    hidden_size = 12
    output = 10
    num_layers = 2
    rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    my_rnn = TinyRNN(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    )

    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
    )

    my_lstm = TinyLSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    )

    x = torch.randn(10, 3, 3)
    output, temp = lstm(x)
    my_output, my_temp = my_lstm(x)
    print(output.shape, my_output.shape)
    print("lstm", temp[0].shape, temp[1].shape)
    print("my_lstm", my_temp[0].shape, my_temp[1].shape)
