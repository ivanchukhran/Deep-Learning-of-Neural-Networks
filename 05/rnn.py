import random
from typing import Tuple

import torch
from tokenizer import TextTokenizer
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


class TextGeneratorLSTM(nn.Module):
    """Complete text generation model with LSTM"""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = TinyLSTM(embed_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)

        lstm_out, hidden = self.lstm(embedded, hidden)
        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Output layer
        output = self.output(lstm_out)  # (batch_size, seq_length, vocab_size)

        return output, hidden

    def generate_text(
        self,
        tokenizer: TextTokenizer,
        start_text: str = "",
        max_length: int = 100,
        temperature: float = 1.0,
        strategy: str = "temperature",
        device="cpu",
    ):
        """Generate text using different strategies"""
        self.eval()

        if start_text:
            tokens = tokenizer.tokenize(start_text)
            input_seq = tokenizer.encode(tokens)
        else:
            # Start with a random token
            input_seq = [random.randint(0, tokenizer.vocab_size - 1)]

        generated = input_seq.copy()
        hidden = None

        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                x = (
                    torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
                )  # (1, seq_length)

                # Forward pass
                output, hidden = self.forward(x, hidden)

                # Get the last time step
                logits = output[0, -1, :]  # (vocab_size,)

                # Apply temperature
                logits = logits / temperature

                # Choose next token based on strategy
                if strategy == "greedy":
                    next_token = torch.argmax(logits).item()
                elif strategy == "temperature":
                    probs = torch.softmax(logits, dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                else:  # random
                    next_token = random.randint(0, len(logits) - 1)

                generated.append(next_token)  # pyright: ignore
                input_seq = [next_token]  # Use only the last token for next prediction

        # Decode back to text
        generated_tokens = tokenizer.decode(generated)
        return " ".join(generated_tokens)
