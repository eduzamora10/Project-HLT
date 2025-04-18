import joblib
import os
import sys
import re
import string
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Argument Check
if len(sys.argv) != 2:
    print("Usage: python evaluate_hard_dataset.py [logistic | svm | nb]")
    sys.exit(1)

chosen_model = sys.argv[1].lower()

# Load Saved Models
model_dir = "saved_models"
model = joblib.load(os.path.join(model_dir, f"{chosen_model}_model.joblib"))
vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

# Load Hard Dataset
with open("dataset-files/x_test_hard.txt", "r", encoding="utf-8") as f:
    x_test_hard = [line.strip() for line in f]

with open("dataset-files/y_test_hard.txt", "r", encoding="utf-8") as f:
    y_test_hard = [line.strip() for line in f]

# Preprocessing Data
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

# Cleaned data stored in x_test_hard as a list of strings
x_test_hard_clean = [preprocess(text) for text in x_test_hard]

# Vectorization
# Use the loaded vectorizer to transform the hard test data
X_test_hard_tfidf = vectorizer.transform(x_test_hard_clean)

# Label Encoding
# Use the loaded label encoder to transform the hard test labels
y_test_hard_enc = label_encoder.transform(y_test_hard)

# Predict 
y_pred_hard = model.predict(X_test_hard_tfidf)

unique_label_indices = np.unique(y_test_hard_enc)  # Numeric encoded labels
unique_label_names = label_encoder.inverse_transform(unique_label_indices)  # Their string names

# Evaluation
accuracy = accuracy_score(y_test_hard_enc, y_pred_hard)

# Fix: Specify the `labels` parameter to match the encoder's classes.
report = classification_report(
    y_test_hard_enc, 
    y_pred_hard, 
    target_names=unique_label_names,  # Ensure target_names are correct
    labels=unique_label_indices,  # Ensure labels are correct
    zero_division=0
)

# Create Results Directory
os.makedirs("results", exist_ok=True)

# Determine Output File Name
filename_map = {
    "logistic": "logreg_hard_results.txt",
    "svm": "svm_hard_results.txt",
    "nb": "nb_hard_results.txt"
}
output_path = os.path.join("results", filename_map[chosen_model])

# Write to File
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"=== {chosen_model.upper()} Model Evaluation on Hard Dataset ===\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print(f"\nEvaluation results on hard dataset written to {output_path}")
