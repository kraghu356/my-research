"""
Preprocessing for UNSW-NB15 to match the CNN-LSTM-DQN intrusion detection
pipeline used in "Class-Frequency-Aware Deep Reinforcement Learning for
Imbalanced Network Intrusion Detection".

Produces train/test feature matrices, integer multiclass labels, and the
class-frequency-adaptive penalty lambda_c = log(N / N_c) used by the DQN reward.

Expected raw files (from the official UNSW-NB15 partitioned release):
    UNSW_NB15_training-set.csv   (~175,341 rows)
    UNSW_NB15_testing-set.csv    (~82,332 rows)

Usage:
    python preprocess_unsw_nb15.py \
        --train UNSW_NB15_training-set.csv \
        --test  UNSW_NB15_testing-set.csv \
        --out   ./processed

Outputs (in --out):
    X_train.npy, y_train.npy, X_test.npy, y_test.npy
    label_map.json     (class name -> integer id)
    lambda_c.json      (integer id -> penalty weight, computed from TRAIN counts)
    class_counts.json  (integer id -> N_c on train)
    scaler.pkl, encoders.pkl   (fitted transforms, for reuse / inference)
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

# Categorical flow features in UNSW-NB15
CATEGORICAL = ["proto", "service", "state"]
# Columns to drop: 'id' is an index, 'label' is the binary target (we use the
# multiclass 'attack_cat' so the rare-class story is preserved).
DROP = ["id", "label"]
TARGET = "attack_cat"


def load(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # 'Normal' rows sometimes carry NaN in attack_cat depending on the release
    train[TARGET] = train[TARGET].fillna("Normal").astype(str).str.strip()
    test[TARGET] = test[TARGET].fillna("Normal").astype(str).str.strip()
    return train, test


def build(train, test, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    y_train_raw = train[TARGET].values
    y_test_raw = test[TARGET].values

    feat_cols = [c for c in train.columns if c not in DROP + [TARGET]]
    Xtr = train[feat_cols].copy()
    Xte = test[feat_cols].copy()

    # --- Categorical encoding -------------------------------------------------
    # Fit the encoder on the UNION of train+test categories so that categories
    # appearing only in the test split don't crash inference. This is a
    # preprocessing convenience and does NOT leak target information.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_cat = pd.concat([Xtr[CATEGORICAL], Xte[CATEGORICAL]], axis=0)
    enc.fit(combined_cat)
    Xtr[CATEGORICAL] = enc.transform(Xtr[CATEGORICAL])
    Xte[CATEGORICAL] = enc.transform(Xte[CATEGORICAL])

    # --- Numeric scaling ------------------------------------------------------
    # Scaler is fit on TRAIN ONLY (no test leakage).
    scaler = StandardScaler()
    Xtr_arr = scaler.fit_transform(Xtr.values.astype(np.float32))
    Xte_arr = scaler.transform(Xte.values.astype(np.float32))

    # --- Label encoding -------------------------------------------------------
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw).astype(np.int64)
    y_test = le.transform(y_test_raw).astype(np.int64)
    label_map = {cls: int(i) for i, cls in enumerate(le.classes_)}

    # --- Class-frequency-adaptive penalty  lambda_c = log(N / N_c) ------------
    # Computed from TRAIN counts only, exactly as the DQN reward expects.
    counts = np.bincount(y_train, minlength=len(le.classes_))
    N = counts.sum()
    lambda_c = {int(c): float(np.log(N / max(n, 1))) for c, n in enumerate(counts)}
    class_counts = {int(c): int(n) for c, n in enumerate(counts)}

    # --- Persist --------------------------------------------------------------
    np.save(os.path.join(out_dir, "X_train.npy"), Xtr_arr)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), Xte_arr)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    with open(os.path.join(out_dir, "lambda_c.json"), "w") as f:
        json.dump(lambda_c, f, indent=2)
    with open(os.path.join(out_dir, "class_counts.json"), "w") as f:
        json.dump(class_counts, f, indent=2)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "encoders.pkl"), "wb") as f:
        pickle.dump({"ordinal": enc, "label": le}, f)

    # --- Report ---------------------------------------------------------------
    print(f"Features: {Xtr_arr.shape[1]}  | train: {Xtr_arr.shape[0]}  test: {Xte_arr.shape[0]}")
    print("\nClass | name            |  N_c (train) | lambda_c")
    print("-" * 52)
    inv = {v: k for k, v in label_map.items()}
    for c in sorted(inv):
        print(f"{c:>5} | {inv[c]:<15} | {class_counts[c]:>11} | {lambda_c[c]:.4f}")
    print(
        "\nRare classes (high lambda_c) are the analogue of NSL-KDD's U2R/R2L:"
        "\n  typically Worms, Shellcode, Backdoor, Analysis."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="./processed")
    args = ap.parse_args()
    tr, te = load(args.train, args.test)
    build(tr, te, args.out)


if __name__ == "__main__":
    main()
