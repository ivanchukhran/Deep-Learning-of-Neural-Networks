import os
import re
import string
from typing import Literal

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.corpus.reader.twitter import TweetTokenizer
from nltk.tag.sequential import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.models.efficientnet import List

nltk.download("punkt")
nltk.download("stopwords")


def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()
    text = re.sub("\\[.*?\\]", "", text)
    text = re.sub("https?://\\S+|www\\.\\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\\w*\\d\\w*", "", text)
    return text


def remove_stopword(x):
    return [y for y in x if y not in stopwords.words("english")]


class TwitterSentimentDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        split: Literal["train", "test"] = "train",
        max_length: int = 50,
        vocab_size: int = 10000,
        validation_size: float = 0.2,
    ):
        self.data = pd.read_csv(
            os.path.join(root, f"{split}.csv"),
            dtype={"textID": str, "text": str, "selected_text": str, "sentiment": str},
        )
        self.tokenizer = TweetTokenizer()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.__prepare()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text.lower())

    def token_to_index(self, token: str) -> int | None:
        return self.token2idx.get(token)

    def index_to_token(self, index: int) -> str | None:
        return self.idx2token.get(index)

    def __prepare(self):
        all_tokens = []
        for text in self.data["text"]:
            text = clean_text(text)
            text = " ".join(remove_stopword(text.split()))
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)

        unique_tokens = sorted(list(set(all_tokens)))

        token_freq = {}
        for token in all_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1

        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        unique_tokens = [token for token, _ in sorted_tokens[: self.vocab_size - 2]]

        self.token2idx = {"<PAD>": 0, "<UNK>": 1}
        self.token2idx.update(
            {token: idx + 2 for idx, token in enumerate(unique_tokens)}
        )
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.tokenized_texts = []
        self.labels = []

        for _, row in self.data.iterrows():
            tokens = self.tokenize(row["text"])  # pyright: ignore
            tokens = tokens[: self.max_length]

            token_indices = [
                self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens
            ]

            token_indices = token_indices + [self.token2idx["<PAD>"]] * (
                self.max_length - len(token_indices)
            )

            self.tokenized_texts.append(token_indices)

            sentiment = row["sentiment"]
            sentiment_idx = self.sentiment_map.get(sentiment, 1)  # pyright: ignore
            self.labels.append(sentiment_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text_indices = self.tokenized_texts[idx]
        labels = self.labels[idx]

        input = torch.tensor(text_indices, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return input, labels

    def __len__(self):
        return len(self.tokenized_texts)


if __name__ == "__main__":
    path = "04/data"
    split = "train"
    dataset = TwitterSentimentDataset(root=path, split=split)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        inputs, labels = batch
        print(inputs.shape, labels.shape)
        break
