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


class TinyGRU(BaseRecurrent):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gate_size=3,
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

    def forward(self, x: torch.Tensor, hx: torch.Tensor | None = None):
        batch_size, seq_length, _ = x.size()

        if hx is None:
            h_zeros = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=x.device
            )
        else:
            h_zeros = hx

        h_layers = torch.unbind(h_zeros, dim=0)
        outputs = []

        for t in range(seq_length):
            x_t = x[:, t, :]
            new_h_layers = []

            for layer in range(self.num_layers):
                w_ih = getattr(self, f"w_ih_{layer}")
                w_hh = getattr(self, f"w_hh_{layer}")
                b_ih = getattr(self, f"b_ih_{layer}")
                b_hh = getattr(self, f"b_hh_{layer}")

                h_prev = h_layers[layer]

                h_new = self.gru_cell(x_t, h_prev, w_ih, w_hh, b_ih, b_hh)

                new_h_layers.append(h_new)
                x_t = h_new

            h_layers = new_h_layers
            outputs.append(h_layers[-1].unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        final_hidden = torch.stack(h_layers, dim=0)

        return output, final_hidden

    def gru_cell(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor,
        b_hh: torch.Tensor,
    ):
        gi = x_t @ w_ih + b_ih
        gh = h_prev @ w_hh + b_hh

        i_r, i_z, i_n = gi.chunk(self.gate_size, 1)
        h_r, h_z, h_n = gh.chunk(self.gate_size, 1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)

        new_gate = torch.tanh(i_n + reset_gate * h_n)
        h_new = (1 - update_gate) * new_gate + update_gate * h_prev

        return h_new
