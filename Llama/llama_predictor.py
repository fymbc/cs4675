import csv
import os
import time
from collections import defaultdict

import kagglehub
from sklearn.metrics import classification_report, confusion_matrix

from llama_cpp import Llama

# === CONFIGURATION ===
PHISHING_DATASET_ID = "ealvaradob/phishing-dataset"
PHISHING_FILE = "phishing_site_urls.csv"
LOCAL_CSV_PATH = "./data/phishing_site_urls.csv"

MODEL_PATH = "./models/llama-3-8b-instruct.Q5_K_M.gguf"  # Update with your actual model path
OUTPUT_DIR = "./results_llama"
MAX_TOKENS = 50
TEMPERATURE = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SETUP MODEL ===
print("[*] Loading LLaMA model...")
llama = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    temperature=TEMPERATURE,
    verbose=False,
)

# === PROMPT TEMPLATE ===
def make_prompt(url: str) -> str:
    return f"""[INST] Is the following URL likely to be a phishing website? Answer "1" for phishing, "0" for not phishing.

URL: {url}
Answer: [/INST]"""

# === LABEL MAPPING ===
# Ground truth: 0 = phishing, 1 = legitimate
# Model: 1 = phishing, 0 = not phishing
def map_llama_to_gt(label: str) -> int:
    label = label.strip()
    if label.startswith("1"):
        return 0  # 1 = phishing → Kaggle 0
    if label.startswith("0"):
        return 1  # 0 = not phishing → Kaggle 1
    return -1

# === DATA LOADING ===
def load_data():
    try:
        csv_path = kagglehub.model_download(PHISHING_DATASET_ID) + f"/{PHISHING_FILE}"
        print("[*] Loaded data from KaggleHub.")
    except:
        csv_path = LOCAL_CSV_PATH
        print("[!] KaggleHub failed, falling back to local file.")

    urls, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            urls.append(row[0])
            labels.append(int(row[1]))
    return urls, labels

# === PREDICTION LOOP ===
def predict():
    urls, ground_truths = load_data()
    predicted_labels = []
    groups = defaultdict(list)

    for i, (url, true_label) in enumerate(zip(urls, ground_truths)):
        prompt = make_prompt(url)
        result = llama(prompt, max_tokens=MAX_TOKENS)
        raw_output = result["choices"][0]["text"]
        pred_label = map_llama_to_gt(raw_output)

        predicted_labels.append(pred_label)

        key = (
            "correct" if pred_label == true_label else
            "false_positive" if pred_label == 0 else
            "false_negative"
        )
        groups[key].append((url, raw_output.strip()))

        print(f"[{i+1}/{len(urls)}] URL: {url} | Pred: {pred_label} | GT: {true_label}")

        # Optional sleep to simulate API delay or avoid overuse
        # time.sleep(0.25)

    return ground_truths, predicted_labels, groups

# === METRIC REPORTING ===
def save_metrics(y_true, y_pred, groups):
    print("\n=== METRICS ===")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))

    with open(f"{OUTPUT_DIR}/metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True", "Pred"])
        writer.writerows(zip(y_true, y_pred))

    for category, samples in groups.items():
        with open(f"{OUTPUT_DIR}/{category}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["URL", "Raw Output"])
            writer.writerows(samples)

if __name__ == "__main__":
    truths, preds, groupings = predict()
    save_metrics(truths, preds, groupings)
