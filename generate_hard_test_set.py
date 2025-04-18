# generate_hard_test_set.py
import json
from collections import Counter

# Load hardest languages
with open("hardest_languages.json", "r", encoding="utf-8") as f:
    hardest = json.load(f)

# Combine all hardest languages across all models
combined_hardest = set()
for langs in hardest.values():
    combined_hardest.update(langs)

print(f"Total unique hard languages across all models: {len(combined_hardest)}")

# Load original test set
with open("dataset-files/x_test.txt", "r", encoding="utf-8") as f:
    x_test = [line.strip() for line in f]

with open("dataset-files/y_test.txt", "r", encoding="utf-8") as f:
    y_test = [line.strip() for line in f]

assert len(x_test) == len(y_test), "Mismatch between x_test and y_test lengths."

# Filter hard samples
x_hard, y_hard = [], []
for text, lang in zip(x_test, y_test):
    if lang in combined_hardest:
        x_hard.append(text)
        y_hard.append(lang)

# Write new hard test set to files
with open("dataset-files/x_test_hard.txt", "w", encoding="utf-8") as f:
    for line in x_hard:
        f.write(line + "\n")

with open("dataset-files/y_test_hard.txt", "w", encoding="utf-8") as f:
    for label in y_hard:
        f.write(label + "\n")

print(f"\nHard test set created with {len(x_hard)} samples.")
print(f"*Saved to 'dataset-files/x_test_hard.txt' and 'y_test_hard.txt'")
print(f"*Language distribution (optional):\n")

# print label distribution
lang_counts = Counter(y_hard)
for lang, count in lang_counts.most_common():
    print(f"   {lang}: {count} samples")
