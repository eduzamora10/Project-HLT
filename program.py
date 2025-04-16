# code that checks if the dataset files are present and loads them

# import pandas as pd

# # === Load label file (semicolon-separated!) ===
# labels_df = pd.read_csv("dataset-files/labels.csv", sep=";")
# labels_df["ISO 369-3"] = labels_df["ISO 369-3"].str.strip().str.lower()
# code_to_lang = dict(zip(labels_df["ISO 369-3"], labels_df["English"]))

# # === Load training data ===
# with open("dataset-files/x_train.txt", "r", encoding="utf-8") as f:
#     x_train = [line.strip() for line in f.readlines()]

# with open("dataset-files/y_train.txt", "r", encoding="utf-8") as f:
#     y_train = [line.strip() for line in f.readlines()]

# # === Load test data ===
# with open("dataset-files/x_test.txt", "r", encoding="utf-8") as f:
#     x_test = [line.strip() for line in f.readlines()]

# with open("dataset-files/y_test.txt", "r", encoding="utf-8") as f:
#     y_test = [line.strip() for line in f.readlines()]

# # === Sanity check ===
# print(f"Loaded {len(x_train)} training samples, {len(x_test)} test samples")
# print(f"Example:")
# print(f"Text: {x_train[0]}")
# print(f"Label code: {y_train[0]} → {code_to_lang.get(y_train[0], 'Unknown')}")

# loading dataset files and preprocessing the text data
import re
import string
import pandas as pd

from sklearn.model_selection import train_test_split

# Load data from dataset-files (edit path if needed)
with open("dataset-files/x_train.txt", "r", encoding="utf-8") as f:
    x_train = [line.strip() for line in f]

with open("dataset-files/y_train.txt", "r", encoding="utf-8") as f:
    y_train = [line.strip() for line in f]

# === Preprocessing function ===
def preprocess(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation (optional: keep if useful for context)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply preprocessing
x_train_clean = [preprocess(text) for text in x_train]

# Example check
print("Original:", x_train[0])
print("Cleaned:", x_train_clean[0])
print("Label:", y_train[0])
