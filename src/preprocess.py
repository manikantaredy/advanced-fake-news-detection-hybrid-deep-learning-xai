import re
import string
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import (
    FAKE_PATH,
    TRUE_PATH,
    MAX_WORDS,
    MAX_LEN,
    TEST_SIZE,
    RANDOM_STATE,
)

import pickle

# Download NLTK data (first run only)
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove links, punctuation, stopwords, lemmatize."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove urls
    text = re.sub(r"@\w+|#\w+", "", text)                # remove @, #
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", "", text)                      # remove digits
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def load_and_prepare_data():
    """Load Fake.csv & True.csv, create labels, clean, tokenize, split."""
    print(f"Loading FAKE news from: {FAKE_PATH}")
    fake_df = pd.read_csv(FAKE_PATH)
    fake_df["label"] = 1  # FAKE

    print(f"Loading TRUE news from: {TRUE_PATH}")
    true_df = pd.read_csv(TRUE_PATH)
    true_df["label"] = 0  # REAL

    print(f"Fake shape: {fake_df.shape}")
    print(f"True shape: {true_df.shape}")

    # Combine into one dataframe
    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Combined shape: {df.shape}")

    # We will use title + text together for better context
    if not {"title", "text"}.issubset(df.columns):
        raise ValueError(f"Expected 'title' and 'text' columns, but got: {df.columns}")

    print("Combining title and text...")
    df["full_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

    print("Cleaning text (this may take a bit)...")
    df["clean_text"] = df["full_text"].apply(clean_text)

    X = df["clean_text"].values
    y = df["label"].values

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    print(f"Train shape: {X_train_pad.shape}, Test shape: {X_test_pad.shape}")
    print("Class balance in y_train:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        label_name = "FAKE" if cls == 1 else "REAL"
        print(f"  {label_name} ({cls}): {cnt}")

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer


def save_tokenizer(tokenizer, path):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {path}")
