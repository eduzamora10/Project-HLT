import pandas as pd

# === Load label file (semicolon-separated!) ===
labels_df = pd.read_csv("dataset-files/labels.csv", sep=";")
labels_df["ISO 369-3"] = labels_df["ISO 369-3"].str.strip().str.lower()
code_to_lang = dict(zip(labels_df["ISO 369-3"], labels_df["English"]))

# === Load training data ===
with open("dataset-files/x_train.txt", "r", encoding="utf-8") as f:
    x_train = [line.strip() for line in f.readlines()]

with open("dataset-files/y_train.txt", "r", encoding="utf-8") as f:
    y_train = [line.strip() for line in f.readlines()]

# === Load test data ===
with open("dataset-files/x_test.txt", "r", encoding="utf-8") as f:
    x_test = [line.strip() for line in f.readlines()]

with open("dataset-files/y_test.txt", "r", encoding="utf-8") as f:
    y_test = [line.strip() for line in f.readlines()]

# === Sanity check ===
print(f"Loaded {len(x_train)} training samples, {len(x_test)} test samples")
print(f"Example:")
print(f"Text: {x_train[0]}")
print(f"Label code: {y_train[0]} → {code_to_lang.get(y_train[0], 'Unknown')}")
