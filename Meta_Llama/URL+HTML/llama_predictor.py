import math
import pandas as pd
import csv
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import requests
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration and Constants for Together AI LLaMA
# ------------------------------------------------------------
TOGETHER_API_KEY = ""
LLAMA_MODEL = "meta-llama/Llama-3-8b-chat-hf"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
MAX_TOKENS = 10

client = together.Llama(api_key=TOGETHER_API_KEY)
# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def build_single_prompt(item):
    return (
        f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )

def call_llama_api(prompt_content):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    try:
        response = requests.post(TOGETHER_URL, headers=headers, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("LLaMA API call failed:", e)
        return None

def group_results(samples, predictions):
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        try:
            api_int = int(api_pred)
        except:
            print(f"Skipping URL {sample['url']} due to invalid prediction '{api_pred}'")
            continue

        mapped = 0 if api_int == 1 else 1
        result = {
            "url": sample["url"],
            "label": sample["label"],
            "api_prediction": api_int,
            "mapped_prediction": mapped
        }

        if mapped == sample["label"]:
            grouped["correct"].append(result)
        elif mapped == 1 and sample["label"] == 0:
            grouped["false_positive"].append(result)
        elif mapped == 0 and sample["label"] == 1:
            grouped["false_negative"].append(result)
    return grouped

def write_grouped_results_to_csv(grouped_results):
    for group, items in grouped_results.items():
        fn = f"{group}_results.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "url", "label", "api_prediction", "mapped_prediction"])
            writer.writeheader()
            for it in items:
                writer.writerow({
                    "group": group,
                    "url": it["url"],
                    "label": it["label"],
                    "api_prediction": it["api_prediction"],
                    "mapped_prediction": it["mapped_prediction"]
                })
        print(f"Saved {group} → {fn}")

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    print("Loading dataset from KaggleHub…")
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "harisudhan411/phishing-and-legitimate-urls",
            "new_data_urls.csv"
        )
    except Exception as e:
        print("KaggleHub load failed:", e)
        df = pd.read_csv("new_data_urls.csv")

    print(f"Total rows: {len(df)}")
    if not {"url", "status"}.issubset(df.columns):
        raise ValueError("Dataset must have 'url' and 'status' columns")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    samples = [
        {"url": r["url"], "label": int(r["status"])}
        for _, r in df.iterrows()
        if isinstance(r["url"], str) and r["status"] in {0, 1}
    ][:1000]
    print(f"Prepared {len(samples)} samples.")

    raw_preds, true_labels = [], []
    for item in tqdm(samples, desc="Querying LLaMA"):
        prompt = build_single_prompt(item)
        resp = call_llama_api(prompt)
        if resp in {"0", "1"}:
            raw_preds.append(int(resp))
        else:
            raw_preds.append(-1)
        true_labels.append(item["label"])

    valid = [i for i, p in enumerate(raw_preds) if p != -1]
    y_true = [true_labels[i] for i in valid]
    y_api = [raw_preds[i] for i in valid]
    print(f"Got {len(valid)}/{len(samples)} valid predictions.")

    y_pred = [0 if p == 1 else 1 for p in y_api]

    acc = accuracy_score(y_true, y_pred)
    prec1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f11 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f10 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"Accuracy: {acc:.4f}")
    print(f"Legit→ Prec {prec1:.4f}, Rec {rec1:.4f}, F1 {f11:.4f}")
    print(f"Phish→ Prec {prec0:.4f}, Rec {rec0:.4f}, F1 {f10:.4f}")
    print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}")

    metrics = {
        "Accuracy": acc,
        "Precision (Legit=1)": prec1,
        "Recall (Legit=1)": rec1,
        "F1 (Legit=1)": f11,
        "Precision (Phish=0)": prec0,
        "Recall (Phish=0)": rec0,
        "F1 (Phish=0)": f10,
        "FPR": fpr,
        "FNR": fnr
    }
    pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_csv("metrics_results.csv", index=False)
    print("Metrics saved to metrics_results.csv")

    cm = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Phish (0)", "Pred Legit (1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Act Phish (0)", "Act Legit (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="red", fontsize=12)
    plt.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")
    plt.show()

    grouped = group_results([samples[i] for i in valid], [y_api[i] for i in range(len(y_api)) if i in valid])
    write_grouped_results_to_csv(grouped)

if __name__ == "__main__":
    main()
