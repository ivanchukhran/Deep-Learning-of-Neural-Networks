# %% imports
import re
import string
from collections import Counter

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm

# %% Load clean dataset and start to remove some unnecessary staff:
train_ds, test_ds = pd.read_csv("./data/train.csv"), pd.read_csv("./data/test.csv")
print("Describe train dataset:")
train_ds.describe()

# %%
print("Describe test dataset:")
test_ds.describe()
# %%
train_ds.groupby(["sentiment"]).count()["text"].reset_index().sort_values(by="text")
# %%

# %%
plt.figure(figsize=(12, 6))
sns.countplot(x="sentiment", data=train_ds)


# %%
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# %%

results_jaccard = []

for ind, row in tqdm(train_ds.iterrows()):
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1, sentence2)
    results_jaccard.append([sentence1, sentence2, jaccard_score])

jaccard_df = pd.DataFrame(
    results_jaccard, columns=["text", "selected_text", "jaccard_score"]
)
train = train_ds.merge(jaccard_df, how="outer")
# %%
train["num_words_st"] = train["selected_text"].apply(lambda x: len(str(x).split()))
train["num_words_text"] = train["text"].apply(lambda x: len(str(x).split()))
train["difference_in_words"] = train["num_words_text"] - train["num_words_st"]
train.head()
# %%
hist_data = [train["Num_words_ST"], train["Num_word_text"]]
group_labels = ["selected_text", "text"]
plt.figure(figsize=(12, 6))
sns.histplot(data=train, x="Num_words_ST", hue="sentiment", multiple="stack")
sns.histplot(data=train, x="Num_word_text", hue="sentiment", multiple="stack")
plt.show()
# %%
plt.figure(figsize=(12, 6))
p1 = sns.kdeplot(train["num_words_st"], shade=True, color="r").set_title(
    "Kernel Distribution of Number Of words"
)
p1 = sns.kdeplot(train["num_words_text"], shade=True, color="b")
# %%
plt.figure(figsize=(12, 6))
p1 = sns.kdeplot(
    train[train["sentiment"] == "positive"]["difference_in_words"],
    shade=True,
    color="b",
).set_title("Kernel Distribution of Difference in Number Of words")
p2 = sns.kdeplot(
    train[train["sentiment"] == "negative"]["difference_in_words"],
    shade=True,
    color="r",
)
plt.legend()
plt.show()
# %%
plt.figure(figsize=(12, 6))
p1 = sns.kdeplot(
    train[train["sentiment"] == "positive"]["jaccard_score"], shade=True, color="b"
).set_title("KDE of Jaccard Scores across different Sentiments")
p2 = sns.kdeplot(
    train[train["sentiment"] == "negative"]["jaccard_score"], shade=True, color="r"
)
plt.legend(labels=["positive", "negative"])

# %% Cleaning the dataset


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


# %%
train["text"] = train["text"].apply(lambda x: clean_text(x))
train["selected_text"] = train["selected_text"].apply(lambda x: clean_text(x))
train["temp_list"] = train["selected_text"].apply(lambda x: str(x).split())
train["temp_list"] = train["temp_list"].apply(lambda x: remove_stopword(x))
train.head()

# %%
Positive_sent = train[train["sentiment"] == "positive"]
Negative_sent = train[train["sentiment"] == "negative"]
Neutral_sent = train[train["sentiment"] == "neutral"]
top = Counter([item for sublist in Positive_sent["temp_list"] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ["Common_words", "count"]
plt.bar(temp_positive["Common_words"], temp_positive["count"], color="blue")
plt.xticks(rotation=90)
plt.show()
# %%
top = Counter([item for sublist in Neutral_sent["temp_list"] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ["Common_words", "count"]
plt.bar(temp_positive["Common_words"], temp_positive["count"], color="grey")
plt.xticks(rotation=90)
plt.show()

# %%
top = Counter([item for sublist in Negative_sent["temp_list"] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(20))
temp_positive.columns = ["Common_words", "count"]
plt.bar(temp_positive["Common_words"], temp_positive["count"], color="red")
plt.xticks(rotation=90)
plt.show()

# %%
