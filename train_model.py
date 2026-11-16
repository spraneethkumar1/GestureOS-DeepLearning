"""
GestureOS - Train a Keras neural network from dataset/landmarks.csv
Run after collecting examples for every gesture: python train_model.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

CSV_PATH = "dataset/landmarks.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
SEED = 42

def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit("Dataset not found. Run collect_data.py first.")

    df = pd.read_csv(CSV_PATH)
    if "label" not in df.columns or len(df) < 50:
        raise SystemExit("Not enough data. Collect more samples first.")
    df = df.dropna()
    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != 63:
        raise SystemExit(f"Expected 63 features, found {len(feature_cols)}. Check the dataset file.")

    counts = df["label"].value_counts()
    print("\nSamples per class:\n", counts.to_string())
    if len(counts) < 2 or counts.min() < 10:
        raise SystemExit("Collect at least 10 samples for every class (ideally 100-300+ each).")

    X = df[feature_cols].astype("float32").to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"].astype(str))
    # Split once for test set, then split the remaining data into train/validation.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(len(encoder.classes_), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "gesture_model.keras"),
            monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=100, batch_size=32, callbacks=callbacks, verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    report = classification_report(
        y_test, preds, labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_, zero_division=0
    )
    print(f"\nHeld-out test accuracy: {test_acc:.4f}")
    print(f"Held-out test loss: {test_loss:.4f}\n")
    print(report)

    with open(os.path.join(MODEL_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)

    with open(os.path.join(REPORT_DIR, "evaluation.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n{report}\n")

    cm = confusion_matrix(y_test, preds, labels=np.arange(len(encoder.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, xticks_rotation=35, colorbar=False)
    plt.title("GestureOS - Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("GestureOS - Training History")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "training_history.png"), dpi=160)
    plt.close()

    print("\nSaved model: models/gesture_model.keras")
    print("Saved labels: models/labels.json")
    print("Saved evaluation and plots in reports/")
    print("Use results from your actual run in the case study; do not assume accuracy in advance.")

if __name__ == "__main__":
    main()
