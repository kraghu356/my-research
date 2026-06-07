# UNSW-NB15 — Experimental Setup

Third evaluation benchmark for the CNN-LSTM-DQN intrusion detection pipeline
("Class-Frequency-Aware Deep Reinforcement Learning for Imbalanced Network
Intrusion Detection"). Added to address the reviewer concern that NSL-KDD (2009)
does not represent modern network traffic, while keeping the paper's rare-class
recall narrative intact.

## Why UNSW-NB15

UNSW-NB15 is the field's accepted modern substitute for KDD-99 / NSL-KDD. It has
naturally rare attack categories — **Worms, Shellcode, Backdoor, Analysis** —
that play the same role as NSL-KDD's U2R and R2L, so the class-frequency penalty
`lambda_c = log(N / N_c)` transfers directly. The flow-feature format (49 numeric
+ 3 categorical features, predefined train/test split) is structurally close to
NSL-KDD, so the existing pipeline migrates with minimal re-engineering.

## Data

Use the official **partitioned** release:

- `UNSW_NB15_training-set.csv`  (~175,341 rows)
- `UNSW_NB15_testing-set.csv`   (~82,332 rows)

Source: UNSW Canberra Cyber (ADFA). Do NOT commit the raw CSVs to the repo —
they are large and licence-restricted. Add them to `.gitignore` and document
the download location instead.

## Pipeline

1. **Preprocess**

   ```bash
   python preprocess_unsw_nb15.py \
       --train UNSW_NB15_training-set.csv \
       --test  UNSW_NB15_testing-set.csv \
       --out   ./processed
   ```

   Produces: `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`,
   `label_map.json`, `lambda_c.json`, `class_counts.json`, plus the fitted
   `scaler.pkl` / `encoders.pkl`.

   Key choices (defensible under review):
   - Categorical encoding fit on the train+test category union (prevents
     unseen-category crashes; no target leakage).
   - Numeric scaler fit on **train only**.
   - `lambda_c` computed from **train counts only**, matching the DQN reward.

2. **Feed into the existing pipeline.** `X_*` are `(n_samples, n_features)`.
   Apply the same windowing/reshape the NSL-KDD and CICIDS2017 runs use before
   the CNN-LSTM front end. Load `lambda_c.json` into the DQN reward exactly as
   for the other datasets — no code change beyond the dataset loader.

3. **Train / evaluate** with the same protocol as the other two datasets:
   same seeds, same number of runs, same metrics. This is essential — the
   results table is only comparable if the protocol is identical.

## Reporting (for the paper)

Report per the SAME convention you settle on for the integrity pass:
- Per-class recall with a single, consistent uncertainty measure (pick std OR
  bootstrap CI and use it across all three datasets).
- Macro-F1 and weighted-F1 reported as distinct, clearly labelled metrics.
- Paired significance test vs. the no-penalty ablation, across seeds.
- Highlight the rare-class recall lift on Worms / Shellcode / Backdoor as the
  cross-dataset confirmation of the NSL-KDD finding.

## Integrity note

These numbers go in the paper ONLY after the runs actually complete. Until then
UNSW-NB15 is described as the planned modern benchmark, not as reported results.
