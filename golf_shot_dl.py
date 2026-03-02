
"""
Golf Shot Classification (Synthetic Dataset) - Deep Learning Midterm Project

This script can:
1) Generate a synthetic golf-shot dataset (CSV)
2) Train/evaluate a simple deep learning model (MLP) on that dataset

Typical usage (run in a terminal):
    python golf_shot_dl.py --make-data --n_rows 1200
    python golf_shot_dl.py --train

Or do both in one command:
    python golf_shot_dl.py --make-data --train --n_rows 1200
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_golf_dataset(
    n_rows: int = 1200,
    filename: str = "synthetic_golf_shots.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a synthetic launch-monitor style dataset and saves to CSV.
    Label: good_shot (1=good, 0=poor)
    """
    rng = np.random.default_rng(seed)

    # --- Base swing parameters (roughly realistic ranges) ---
    club_speed = rng.normal(loc=100, scale=8, size=n_rows)              # mph
    club_speed = np.clip(club_speed, 75, 125)

    # Smash factor correlated with contact quality
    smash_factor = rng.normal(loc=1.45, scale=0.06, size=n_rows)
    smash_factor = np.clip(smash_factor, 1.20, 1.55)

    ball_speed = club_speed * smash_factor + rng.normal(0, 2.5, n_rows) # mph

    # Launch + spin (slight dependency on attack angle)
    attack_angle = rng.normal(loc=1.5, scale=3.0, size=n_rows)          # degrees
    attack_angle = np.clip(attack_angle, -6, 8)

    launch_angle = rng.normal(loc=13.0 + 0.35*attack_angle, scale=3.0, size=n_rows)
    launch_angle = np.clip(launch_angle, 4, 22)

    spin_rate = rng.normal(loc=2600 - 60*attack_angle, scale=450, size=n_rows)  # rpm
    spin_rate = np.clip(spin_rate, 1400, 4200)

    # Path/face relationships -> direction/dispersion
    club_path = rng.normal(loc=0.0, scale=3.0, size=n_rows)             # degrees (in-to-out positive)
    club_path = np.clip(club_path, -8, 8)

    face_to_path = rng.normal(loc=0.0, scale=2.5, size=n_rows)          # degrees
    face_to_path = np.clip(face_to_path, -7, 7)

    # Offline roughly scales with face-to-path and club speed
    offline_yards = (face_to_path * 3.5 + club_path * 1.2) + rng.normal(0, 6, n_rows)
    offline_yards = np.clip(offline_yards, -55, 55)

    # Carry distance: simple physics-ish approximation (consistent, not perfect physics)
    carry_yards = (
        2.3 * ball_speed
        + 1.6 * launch_angle
        - 0.0045 * spin_rate
        - 0.12 * np.abs(offline_yards)
        + rng.normal(0, 8, n_rows)
    )
    carry_yards = np.clip(carry_yards, 120, 330)

    # --- Probability of a "good shot" using a weighted scoring model ---
    z = (
        2.2 * (smash_factor - 1.40)                 # reward solid contact
        + 0.015 * (carry_yards - 210)               # reward distance
        - 0.035 * np.abs(offline_yards)             # penalize dispersion
        - 0.0012 * np.abs(spin_rate - 2600)         # penalize extreme spin
        - 0.08 * np.abs(launch_angle - 14)          # penalize extreme launch
        - 0.12 * np.abs(face_to_path)               # penalize face/path mismatch
    )

    p_good = sigmoid(z)
    good_shot = rng.binomial(1, p_good, size=n_rows)

    df = pd.DataFrame({
        "club_speed_mph": np.round(club_speed, 2),
        "ball_speed_mph": np.round(ball_speed, 2),
        "smash_factor": np.round(smash_factor, 3),
        "attack_angle_deg": np.round(attack_angle, 2),
        "launch_angle_deg": np.round(launch_angle, 2),
        "spin_rate_rpm": np.round(spin_rate, 0).astype(int),
        "club_path_deg": np.round(club_path, 2),
        "face_to_path_deg": np.round(face_to_path, 2),
        "offline_yards": np.round(offline_yards, 1),
        "carry_yards": np.round(carry_yards, 1),
        "good_shot": good_shot.astype(int)
    })

    df.to_csv(filename, index=False)
    return df


def train_and_evaluate(
    csv_path: str = "synthetic_golf_shots.csv",
    seed: int = 42,
    save_plots: bool = True,
    plot_prefix: str = "training_curve"
) -> dict:
    """
    Trains a simple MLP model and prints evaluation metrics.
    Optionally saves plots to PNG files.
    """
    # Reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = pd.read_csv(csv_path)

    X = df.drop(columns=["good_shot"])
    y = df["good_shot"].astype(int)

    # Split: train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Build MLP
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    y_prob = model.predict(X_test_s).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\nTEST ACCURACY:", round(float(acc), 4))
    print("\nCONFUSION MATRIX:\n", cm)
    print("\nCLASSIFICATION REPORT:\n", report)

    # Plot training curves
    if save_plots:
        # Accuracy
        plt.figure()
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"])
        plt.title("Accuracy vs Epochs")
        acc_path = f"{plot_prefix}_accuracy.png"
        plt.savefig(acc_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Loss
        plt.figure()
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"])
        plt.title("Loss vs Epochs")
        loss_path = f"{plot_prefix}_loss.png"
        plt.savefig(loss_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        acc_path, loss_path = None, None

    return {
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "accuracy_plot": acc_path,
        "loss_plot": loss_path
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic Golf Shot Deep Learning Project")
    parser.add_argument("--make-data", action="store_true", help="Generate synthetic dataset CSV")
    parser.add_argument("--train", action="store_true", help="Train and evaluate the model")
    parser.add_argument("--n_rows", type=int, default=1200, help="Number of synthetic rows to generate")
    parser.add_argument("--csv", type=str, default="synthetic_golf_shots.csv", help="CSV path to save/load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plots", action="store_true", help="Do not save training curve plots")
    args = parser.parse_args()

    if not args.make_data and not args.train:
        parser.print_help()
        return

    if args.make_data:
        df = generate_synthetic_golf_dataset(n_rows=args.n_rows, filename=args.csv, seed=args.seed)
        print(df.head())
        print(f"\nSaved: {args.csv}")
        print("Class balance (good_shot=1):", round(df["good_shot"].mean(), 3))

    if args.train:
        results = train_and_evaluate(
            csv_path=args.csv,
            seed=args.seed,
            save_plots=(not args.no_plots),
            plot_prefix="training_curve"
        )
        if results.get("accuracy_plot"):
            print("\nSaved plots:")
            print(" -", results["accuracy_plot"])
            print(" -", results["loss_plot"])


if __name__ == "__main__":
    main()
