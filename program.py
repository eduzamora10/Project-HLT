# loading dataset files and preprocessing the text data
import sys
import re
import os
import string
import pandas as pd
import numpy as np
import joblib # for saving models and vectorizers

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

# Argument Check
# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    print("Usage: python program.py [logistic | svm | nb]")
    sys.exit(1)

chosen_model = sys.argv[1].lower()

# Load Dataaset
with open("dataset-files/x_train.txt", "r", encoding="utf-8") as f:
    x_train = [line.strip() for line in f]

with open("dataset-files/y_train.txt", "r", encoding="utf-8") as f:
    y_train = [line.strip() for line in f]

with open("dataset-files/x_test.txt", "r", encoding="utf-8") as f:
    x_test = [line.strip() for line in f]

with open("dataset-files/y_test.txt", "r", encoding="utf-8") as f:
    y_test = [line.strip() for line in f]

# Preprocessing Data
def preprocess(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r"\d+", "", text) # Remove digits
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

# Cleaned data stored in x_train and x_test as a list of strings
x_train_clean = [preprocess(text) for text in x_train]
x_test_clean = [preprocess(text) for text in x_test]

# Vectorization
# Using TF-IDF Vectorizer to convert text data into numerical format
# max_features limits the number of features to 5000
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(x_train_clean)
X_test_tfidf = vectorizer.transform(x_test_clean)

# label encoding
# Convert string labels into numerical format
# This is necessary for classification algorithms
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Model Selection
# Choose the model based on the command line argument
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(), 
    "nb": MultinomialNB()
}

if chosen_model not in models:
    print("Invalid model choice. Choose from: logistic, svm, nb")
    sys.exit(1)

model = models[chosen_model]
print(f"\n=== Training {chosen_model.upper()} model ===")

# Training the model
model.fit(X_train_tfidf, y_train_enc)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluation
# Calculate accuracy and classification report
accuracy = accuracy_score(y_test_enc, y_pred)
report = classification_report(y_test_enc, y_pred, target_names=le.classes_, zero_division=0)

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Determine output file name based on the chosen model
filename_map = {
    "logistic": "logreg_results.txt",
    "svm": "svm_results.txt",
    "nb": "nb_results.txt"
}
output_path = os.path.join("results", filename_map[chosen_model])

# Writing results to a file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"=== {chosen_model.upper()} Model Evaluation ===\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print(f"\nEvaluation results written to {output_path}")

# Save Model, Vectorizer, and Label Encoder 
# That way we don't have to retrain the model when we evaulated on the hard test set
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, f"{chosen_model}_model.joblib"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))

print(f"Model, vectorizer, and label encoder saved to '{model_dir}/'")
