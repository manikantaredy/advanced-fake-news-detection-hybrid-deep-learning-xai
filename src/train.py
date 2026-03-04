import os
from config import MODEL_PATH, TOKENIZER_PATH, EPOCHS, BATCH_SIZE
from preprocess import load_and_prepare_data, save_tokenizer
from model import build_hybrid_model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def main():
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, tokenizer = load_and_prepare_data()

    # 2. Save tokenizer for later prediction & explainability
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # 3. Build model
    model = build_hybrid_model()

    # 4. Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 5. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Save model
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
