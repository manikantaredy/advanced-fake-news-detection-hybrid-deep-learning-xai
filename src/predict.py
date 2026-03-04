import sys
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import MODEL_PATH, TOKENIZER_PATH, MAX_LEN

def load_tokenizer(path):
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def predict_text(text):
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob_fake = float(model.predict(pad)[0][0])
    label = "FAKE" if prob_fake >= 0.5 else "REAL"

    print(f"\nText: {text}")
    print(f"Predicted label: {label}")
    print(f"Probability (FAKE): {prob_fake:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter news text: ")

    predict_text(text)
