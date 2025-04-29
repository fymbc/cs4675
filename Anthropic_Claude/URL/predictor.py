#Imports
import math
import pandas as pd
import csv
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import anthropic
import matplotlib.pyplot as plt

# Config
ANTHROPIC_API_KEY = ""  # (API key redacted for database)
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 10

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Build prompt
def build_single_prompt(item):
    return (
        f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )

# Call Anthropic API
def call_anthropic_api(prompt_content):
    try:
        response_message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt_content}]
        )
        content = response_message.content
        # get text
        if isinstance(content, list) and len(content) > 0:
            content = getattr(content[0], "text", str(content[0]))
        elif hasattr(content, "text"):
            content = content.text
        if not isinstance(content, str):
            content = str(content)
        return content.strip()
    except Exception as e:
        print("API call failed:", e)
        return None

# Group preds: correct, fp, fn
def group_results(samples, predictions):
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        if api_pred == 1:
            mapped_pred = 0
        elif api_pred == 0:
            mapped_pred = 1
        else:
            print(f"Skipping {sample['url']} due to invalid prediction {api_pred}")
            continue
        result = {
            "url": sample["url"],
            "label": sample["label"],
            "api_prediction": api_pred,
            "mapped_prediction": mapped_pred
        }
        if mapped_pred == sample["label"]:
            grouped["correct"].append(result)
        elif mapped_pred == 1 and sample["label"] == 0:
            grouped["false_positive"].append(result)
        elif mapped_pred == 0 and sample["label"] == 1:
            grouped["false_negative"].append(result)
    return grouped

# Group results --> CSV
def write_grouped_results_to_csv(grouped_results):
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["group", "url", "label", "api_prediction", "mapped_prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        print(f"Saved {group} results to {filename}")

def main():
    print("Loading dataset...")
    df = None
    file_path = "new_data_urls.csv"

    try:
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "harisudhan411/phishing-and-legitimate-urls", file_path)
    except Exception as e:
        print(f"Error loading KaggleHub dataset: {e}")
        try:
            df = pd.read_csv(file_path)
            print("Loaded locally.")
        except Exception as e_local:
            print(f"Local load failed: {e_local}")
            return

    if df is None or "url" not in df.columns or "status" not in df.columns:
        print("Dataset missing 'url' or 'status' columns.")
        return

    print("Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        if isinstance(row["url"], str) and not pd.isna(row["status"]):
            try:
                label = int(row["status"])
                if label in [0, 1]:
                    samples.append({"url": row["url"], "html": "", "label": label})
            except ValueError:
                continue

    sample_limit = 1000
    samples = samples[:sample_limit]
    print(f"Prepared {len(samples)} samples.")

    if not samples:
        print("No valid samples.")
        return

    print("Calling Anthropic API...")
    raw_api_predictions, ground_truth_labels = [], []

    for item in tqdm(samples, desc="Processing samples"):
        prompt_content = build_single_prompt(item)
        api_response_content = call_anthropic_api(prompt_content)
        prediction = -1
        if api_response_content == "1":
            prediction = 1
        elif api_response_content == "0":
            prediction = 0
        else:
            print(f"Unexpected response for {item['url']}: '{api_response_content}'")
        raw_api_predictions.append(prediction)
        ground_truth_labels.append(item["label"])

    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]
    if not valid_indices:
        print("No valid predictions.")
        return

    valid_samples = [samples[i] for i in valid_indices]
    y_true_mapped = [ground_truth_labels[i] for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]
    y_pred_mapped = [0 if pred == 1 else 1 for pred in y_pred_api]

    print(f"\nProcessed {len(samples)} samples, {len(valid_indices)} valid predictions.")

    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec_legit = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec_legit = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1_legit = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    prec_phish = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec_phish = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f1_phish = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)

    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        print("\nConfusion Matrix computed.")
    except Exception as e:
        print(f"Confusion matrix error: {e}")
        tn = fp = fn = tp = false_positive_rate = false_negative_rate = 0

    print("\n--- Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Legitimate=1): {prec_legit:.4f}")
    print(f"Recall (Legitimate=1): {rec_legit:.4f}")
    print(f"F1 Score (Legitimate=1): {f1_legit:.4f}")
    print(f"Precision (Phishing=0): {prec_phish:.4f}")
    print(f"Recall (Phishing=0): {rec_phish:.4f}")
    print(f"F1 Score (Phishing=0): {f1_phish:.4f}")
    print(f"FPR: {false_positive_rate:.4f}")
    print(f"FNR: {false_negative_rate:.4f}")

    metrics_df = pd.DataFrame([
        ["Accuracy", acc],
        ["Precision (Legitimate=1)", prec_legit],
        ["Recall (Legitimate=1)", rec_legit],
        ["F1 (Legitimate=1)", f1_legit],
        ["Precision (Phishing=0)", prec_phish],
        ["Recall (Phishing=0)", rec_phish],
        ["F1 (Phishing=0)", f1_phish],
        ["False Positive Rate (FPR)", false_positive_rate],
        ["False Negative Rate (FNR)", false_negative_rate],
    ], columns=["Metric", "Value"])
    metrics_df.to_csv("metrics_results.csv", index=False)
    print("Saved metrics to 'metrics_results.csv'.")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow([[tn, fp], [fn, tp]], cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nPhishing", "Predicted\nLegitimate"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual\nPhishing", "Actual\nLegitimate"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str([[tn, fp], [fn, tp]][i][j]), ha='center', va='center', color="red", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to 'confusion_matrix.png'.")
    plt.show()

    # Saving results for later use
    grouped_results = group_results(valid_samples, y_pred_api)
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()
