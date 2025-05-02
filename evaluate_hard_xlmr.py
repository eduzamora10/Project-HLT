import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# --- Paths ---
data_dir = "dataset-files"
model_dir = "model-files"

def load_txt(filename):
    with open(os.path.join(data_dir, filename), encoding='utf-8') as file:
        return [line.strip() for line in file]

# --- Load hard test set ---
x_test_hard = load_txt("x_test_hard.txt")
y_test_hard = load_txt("y_test_hard.txt")

# --- Label map ---
label_df = pd.read_csv(os.path.join(data_dir, "labels.csv"), delimiter=';')
label_map = dict(zip(label_df["Label"], label_df["English"]))

# Convert short codes to full English names
y_test_hard_full = [label_map.get(code, code) for code in y_test_hard]

# Encode labels using same label set as during training
label_encoder = LabelEncoder()
label_encoder.fit(label_df['English'].tolist())
y_test_hard_encoded = label_encoder.transform(y_test_hard_full)

label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(
    f'{model_dir}'
)
model = AutoModelForSequenceClassification.from_pretrained(
    f'{model_dir}',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# a checkpoint to see if the files are correctly recognized
# print(type(tokenizer)) --> XLMRobertaTokenizerFast
# print(type(model))    --> XLMRobertaForSequenceClassification

# --- Tokenization function ---
def token_function(sample):
    return tokenizer(sample["text"], padding="max_length",  truncation=True, max_length=128)

# --- Tokenize dataset ---
hard_dataset = Dataset.from_dict(
    {"text": x_test_hard, "label": y_test_hard_encoded}
)
tokenized = hard_dataset.map(token_function, batched=True)

# --- Evaluate ---
trainer = Trainer(model=model, tokenizer=tokenizer)
predictions = trainer.predict(tokenized)
preds = np.argmax(predictions.predictions, axis=1)

# another checkpoint just in case
# print(type(trainer)) --> transformers.trainer.Trainer

# --- Accuracy ---
accuracy = accuracy_score(y_test_hard_encoded, preds)

# --- Classification report ---
unique_label_indices = np.unique(y_test_hard_encoded)
unique_label_names = label_encoder.inverse_transform(unique_label_indices)

report = classification_report(
    y_test_hard_encoded,
    preds,
    target_names=unique_label_names,
    labels=unique_label_indices,
    zero_division=0
)

# --- Save report ---
os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", "xmlr_hard_results.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== XMLRoBERTa Model Evaluation on Hard Dataset ===\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

print(f"\nEvaluation results written to {output_path}")
print(f"Hard test set accuracy: {accuracy:.4f}")