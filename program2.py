import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Path to the folder
data_dir = "dataset-files"

# --- Load x and y data ---
def load_txt(filename):
    with open(os.path.join(data_dir, filename), encoding='utf-8') as f:
        return [line.strip() for line in f]

x_train = load_txt("x_train.txt")
y_train = load_txt("y_train.txt")
x_test = load_txt("x_test.txt")
y_test = load_txt("y_test.txt")

# --- Load and parse labels.csv ---
# Semicolon-delimited CSV
label_df = pd.read_csv(os.path.join(data_dir, "labels.csv"), delimiter=";")

# Create a mapping from 'Label' to 'English'
label_map = dict(zip(label_df['Label'], label_df['English']))

# Convert short codes (e.g., 'eng') to full English names (e.g., 'English')
y_train_full = [label_map.get(code, code) for code in y_train]
y_test_full = [label_map.get(code, code) for code in y_test]

# --- Encode labels as integers ---
label_encoder = LabelEncoder()
label_encoder.fit(y_train_full + y_test_full)

y_train_encoded = label_encoder.transform(y_train_full)
y_test_encoded = label_encoder.transform(y_test_full)

# Create mappings
label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

# --- Sanity checks ---
print("Train sample:", x_train[0])
print("Label (short):", y_train[0], "→ Full:", y_train_full[0])
print("Encoded label:", y_train_encoded[0])
print("Train size:", len(x_train), "| Test size:", len(x_test))
print("Unique labels:", len(label2id))

print("Example label → id:")
print(f"{y_train_full[0]} → {y_train_encoded[0]}")