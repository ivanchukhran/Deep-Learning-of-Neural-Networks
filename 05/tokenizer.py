import re
from collections import Counter
from typing import List


class TextTokenizer:
    """Tokenizer for converting text to tokens and vice versa"""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation"""
        text = text.lower()
        text = re.sub(r"([.!?,:;])", r" \1 ", text)
        tokens = text.split()
        return [token for token in tokens if token.strip()]

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts"""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))

        token_counts = Counter(all_tokens)

        filtered_tokens = [
            token for token, count in token_counts.items() if count >= self.min_freq
        ]

        vocab = [self.pad_token, self.unk_token] + sorted(filtered_tokens)

        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(vocab)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common tokens: {token_counts.most_common(10)}")

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [
            self.word_to_idx.get(token, self.word_to_idx[self.unk_token])
            for token in tokens
        ]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens"""
        return [self.idx_to_word.get(idx, self.unk_token) for idx in indices]
