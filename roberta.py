import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

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
# print("Train sample:", x_train[0])
# print("Label (short):", y_train[0], "→ Full:", y_train_full[0])
# print("Encoded label:", y_train_encoded[0])
# print("Train size:", len(x_train), "| Test size:", len(x_test))
# print("Unique labels:", len(label2id))

# print("Example label → id:")
# print(f"{y_train_full[0]} → {y_train_encoded[0]}")


# Construct train and test datasets from raw lists
train_dataset = Dataset.from_dict({
    "text": x_train,
    "label": y_train_encoded
})

test_dataset = Dataset.from_dict({
    "text": x_test,
    "label": y_test_encoded
})

# Wrap in a DatasetDict to match Hugging Face trainer format
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print(dataset['train'][0])

# initialize pretrained model and tokenizer
model_checkpoint = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Apply tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(tokenized_dataset["train"][0])

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels = len(label_encoder.classes_),
    id2label = id2label,
    label2id = label2id
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./xlmr-results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False
)

# Function to compute accuracy metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train() # Train the model

# save model
trainer.save_model("xlmr-trained-model")
tokenizer.save_pretrained("xlmr-trained-model")

# === Full Evaluation Report ===
predictions = trainer.predict(tokenized_dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
target_names = [id2label[i] for i in sorted(id2label.keys())]

report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

with open("xlmr_classification_report.txt", "w") as f:
    f.write("=== XLM-RoBERTa Model Evaluation ===\n")
    f.write(f"Accuracy: {predictions.metrics['test_accuracy']:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("=== XLM-RoBERTa Model Evaluation ===")
print(f"Accuracy: {predictions.metrics['test_accuracy']:.4f}")
print("\nClassification Report:\n")
print(report)