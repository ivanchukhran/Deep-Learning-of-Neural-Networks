import torch
from tokenizer import TextTokenizer
from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, text: str, tokenizer: TextTokenizer, seq_length: int = 50):
        if not text:
            raise ValueError(
                "The text value should not be empty! Please, provide a correct value for the text to be tokenized"
            )
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.tokens = self.tokenizer.tokenize(text)
        self.encoded_tokens = self.tokenizer.encode(self.tokens)

        print(f"Total tokens: {len(self.tokens)}")
        print(f"Total sequences: {len(self.encoded_tokens) - seq_length}")

    def __len__(self):
        return len(self.encoded_tokens) - self.seq_length

    def __getitem__(self, idx: int):
        input_seq = self.encoded_tokens[idx : idx + self.seq_length]
        target_seq = self.encoded_tokens[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(
            target_seq, dtype=torch.long
        )
