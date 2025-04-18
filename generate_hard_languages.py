# generate_hard_languages.py
import os
import re

# Configurations
results_folder = "results"
models = {
    "logreg": "logreg_results.txt",
    "svm": "svm_results.txt",
    "nb": "nb_results.txt"
}
num_hardest = 20  # you can increase or decrease this

# Function to extract F1 scores from classification report
def extract_f1_scores(file_path):
    f1_scores = {}
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    in_report = False
    for line in lines:
        if "Classification Report" in line:
            in_report = True
            continue
        if in_report and re.match(r"^\s*\S+", line):  # check for any non-empty line
            parts = line.strip().split()
            if len(parts) >= 4:
                lang = parts[0]  # assumes language name has no spaces
                try:
                    f1 = float(parts[3])  # assumes f1-score is 4th column
                    f1_scores[lang] = f1
                except ValueError:
                    continue  # skip header or invalid rows

    return f1_scores

# Main Logic
hardest_languages = {}

for model_key, filename in models.items():
    file_path = os.path.join(results_folder, filename)
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        continue

    print(f"\nExtracting F1 scores from {filename}...")
    f1_scores = extract_f1_scores(file_path)

    if not f1_scores:
        print("No F1-scores found.")
        continue

    # Sort by F1 ascending and get bottom N
    sorted_langs = sorted(f1_scores.items(), key=lambda x: x[1])
    hardest_langs = [lang for lang, _ in sorted_langs[:num_hardest]]
    hardest_languages[model_key] = hardest_langs

    print(f"Hardest languages for {model_key.upper()}: {hardest_langs}")

# Save to file
with open("hardest_languages.json", "w", encoding="utf-8") as f:
    import json
    json.dump(hardest_languages, f, indent=4)

print("\nHardest languages saved to hardest_languages.json")
