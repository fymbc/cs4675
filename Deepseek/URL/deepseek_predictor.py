import math
import requests
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time

# Config
DEEPSEEK_API_KEY = "" # Put in API Key (redacted)
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Build prompt for DeepSeek API
def build_single_prompt(item):
    prompt_content = (
        f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )
    return prompt_content

# Call DeepSeek API
def call_deepseek_api(prompt_content):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": prompt_content}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }

    try:
        response = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content'].strip()
            else:
                print("Unexpected response structure:", response_data)
                return None
        else:
            print("Unexpected response structure (no choices):", response_data)
            return None

    except requests.exceptions.Timeout:
        print("API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if response is not None and response.status_code == 429:
            print("Rate limit hit.")
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"Failed to parse API response: {e}")
        if 'response' in locals() and response is not None:
            print(f"Raw response text: {response.text}")
        return None

# Group samples into 3 groups: correct, f positive, f negative
def group_results(samples, predictions):
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        if api_pred == 1:
            mapped_pred = 0
        elif api_pred == 0:
            mapped_pred = 1
        else:
            print(f"Skipping grouping for URL {sample['url']} due to invalid prediction {api_pred}")
            continue

        result = {
            "url": sample["url"],
            "label": sample["label"],
            "api_prediction": api_pred,
            "mapped_prediction": mapped_pred
        }

        if mapped_pred == sample["label"]:
            grouped["correct"].append(result)
        else:
            if mapped_pred == 1 and sample["label"] == 0:
                grouped["false_positive"].append(result)
            elif mapped_pred == 0 and sample["label"] == 1:
                grouped["false_negative"].append(result)
            else:
                print(f"Unexpected grouping condition: Label={sample['label']}, MappedPred={mapped_pred}")

    return grouped

# Save grouped results to CSV
def write_grouped_results_to_csv(grouped_results):
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["group", "url", "label", "api_prediction", "mapped_prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                row = {
                    "group": group,
                    "url": item["url"],
                    "label": item["label"],
                    "api_prediction": item["api_prediction"],
                    "mapped_prediction": item["mapped_prediction"]
                }
                writer.writerow(row)
        print(f"Group '{group}' results saved to '{filename}'.")

# Main runner for other functions
def main():
    print("Getting dataset from KaggleHub.")
    df = None
    file_path = "new_data_urls.csv"

    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "harisudhan411/phishing-and-legitimate-urls",
            file_path,
        )
        print("Successfully loaded dataset.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nAttempting to load '{file_path}' from local...")
        try:
            df = pd.read_csv(file_path)
            print("Successfully loaded from local directory.")
        except Exception as e_local:
            print(f"Error loading local file '{file_path}': {e_local}")
            return

    if df is None:
        print("DataFrame could not be loaded. Exiting.")
        return

    print("Total rows in dataset:", len(df))

    if "url" not in df.columns or "status" not in df.columns:
        print("Expected columns 'url' and 'status' not found.")
        return

    print("Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        if row["url"] and isinstance(row["url"], str) and not pd.isna(row["status"]):
            try:
                label = int(row["status"])
                if label in [0, 1]:
                    sample = {
                        "url": row["url"],
                        "html": "",
                        "label": label
                    }
                    samples.append(sample)
            except ValueError:
                pass

    sample_limit = 1000
    samples = samples[:sample_limit]
    print(f"Prepared {len(samples)} samples.")

    if not samples:
        print("No valid samples found.")
        return

    raw_api_predictions = []
    ground_truth_labels = []

    print("Calling DeepSeek API...")
    for item in tqdm(samples, desc="Processing samples"):
        prompt_content = build_single_prompt(item)
        api_response_content = call_deepseek_api(prompt_content)

        prediction = -1
        if api_response_content is not None:
            if api_response_content == "1":
                prediction = 1
            elif api_response_content == "0":
                prediction = 0
            else:
                print(f"Unexpected response content for URL '{item['url']}': '{api_response_content}'")

        raw_api_predictions.append(prediction)
        ground_truth_labels.append(item["label"])

    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]

    if len(valid_indices) == 0:
        print("No valid predictions.")
        return

    valid_samples = [samples[i] for i in valid_indices]
    y_true_mapped = [ground_truth_labels[i] for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]
    y_pred_mapped = [0 if pred == 1 else 1 for pred in y_pred_api]

    print(f"\nProcessed {len(samples)} samples, {len(valid_indices)} valid predictions.")

    print("\n--- Evaluation Metrics (Positive Label = 1: Legitimate) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec_legit = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec_legit = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1_legit = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)

    print("\n--- Evaluation Metrics (Positive Label = 0: Phishing) ---")
    prec_phish = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec_phish = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f1_phish = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)

    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        print("\nConfusion Matrix")
        print("                     Predicted Phishing (0)  Predicted Legit (1)")
        print(f"Actual Phishing (0)      {tn:<20}  {fp:<20}")
        print(f"Actual Legit (1)         {fn:<20}  {tp:<20}")

    except ValueError as e:
        print(f"Confusion matrix error: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0
        false_positive_rate = 0
        false_negative_rate = 0

    print("\n--- Overall Metrics ---")
    print(f"Accuracy:                     {acc:.4f}")
    print(f"Precision (Legitimate=1):     {prec_legit:.4f}")
    print(f"Recall (Legitimate=1):        {rec_legit:.4f}")
    print(f"F1 Score (Legitimate=1):      {f1_legit:.4f}")
    print(f"Precision (Phishing=0):       {prec_phish:.4f}")
    print(f"Recall (Phishing=0):          {rec_phish:.4f}")
    print(f"F1 Score (Phishing=0):        {f1_phish:.4f}")
    print(f"False Positive Rate:          {false_positive_rate:.4f}")
    print(f"False Negative Rate:          {false_negative_rate:.4f}")

    results_filename = "results.csv"
    print(f"\nSaving results to {results_filename}...")

    results_data = [
        {"Metric": "Sample Limit Attempted", "Value": sample_limit},
        {"Metric": "Valid Predictions", "Value": len(valid_samples)},
        {"Metric": "Accuracy", "Value": f"{acc:.4f}"},
        {"Metric": "Precision (Legitimate=1)", "Value": f"{prec_legit:.4f}"},
        {"Metric": "Recall (Legitimate=1)", "Value": f"{rec_legit:.4f}"},
        {"Metric": "F1 Score (Legitimate=1)", "Value": f"{f1_legit:.4f}"},
        {"Metric": "Precision (Phishing=0)", "Value": f"{prec_phish:.4f}"},
        {"Metric": "Recall (Phishing=0)", "Value": f"{rec_phish:.4f}"},
        {"Metric": "F1 Score (Phishing=0)", "Value": f"{f1_phish:.4f}"},
        {"Metric": "True Negatives", "Value": tn},
        {"Metric": "False Positives", "Value": fp},
        {"Metric": "False Negatives", "Value": fn},
        {"Metric": "True Positives", "Value": tp},
        {"Metric": "False Positive Rate", "Value": f"{false_positive_rate:.4f}"},
        {"Metric": "False Negative Rate", "Value": f"{false_negative_rate:.4f}"},
    ]

    try:
        with open(results_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Metric", "Value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
        print("Results successfully saved.")

    except IOError as e:
        print(f"Error writing results: {e}")

    grouped_results = group_results(valid_samples, y_pred_api)
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()
