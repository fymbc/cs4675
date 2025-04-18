import google.generativeai as genai
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time
import zipfile
import os

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
GEMINI_API_KEY = ""
GENAI_MODEL = "gemini-1.5-pro-latest"
MAX_OUTPUT_TOKENS = 10

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GENAI_MODEL)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def call_gemini_api(prompt_content, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt_content)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini API request failed on attempt {attempt + 1}: {e}")
            time.sleep(5)
    print("Max retries exceeded for Gemini API.")
    return None

def group_results(samples, predictions):
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        mapped_pred = api_pred
        gt = int(sample["label"])
        result = {
            "url": sample["url"],
            "label": gt,
            "api_prediction": api_pred,
            "mapped_prediction": mapped_pred
        }
        if mapped_pred == gt:
            grouped["correct"].append(result)
        elif mapped_pred == 1 and gt == 0:
            grouped["false_positive"].append(result)
        elif mapped_pred == 0 and gt == 1:
            grouped["false_negative"].append(result)
    return grouped

def write_grouped_results_to_csv(grouped_results):
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["group", "url", "label", "api_prediction", "mapped_prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        print(f"Group '{group}' results saved to '{filename}'.")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("Unzipping and loading prompts from output_100.zip...")

    try:
        with zipfile.ZipFile("output_100.zip", 'r') as zip_ref:
            zip_ref.extractall("./unzipped")
        filepath = os.path.join("./unzipped", "output_100.txt")
        with open(filepath, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("File 'output_100.zip' or 'output_100.txt' not found.")
        return

    prompts = []
    ground_truth_labels = []
    for line in lines:
        parts = line.strip().rsplit(",", 1)
        if len(parts) == 2:
            prompt = parts[0].strip()
            ground_truth = parts[1].strip()
            if ground_truth in ["0", "1"]:
                prompts.append(prompt)
                ground_truth_labels.append(ground_truth)
            else:
                print(f"Skipping invalid ground truth for prompt: {prompt}")
        else:
            print(f"Skipping malformed line: {line.strip()}")

    print(f"Loaded {len(prompts)} valid prompts.")

    raw_api_predictions = []
    for prompt in tqdm(prompts, desc="Querying Gemini"):
        formatted_prompt = (
            f"Given the following input, determine if it is a phishing website or not:\n"
            f"{prompt}\n"
            "Respond ONLY with 1 (phishing) or 0 (not phishing)."
        )
        api_response_content = call_gemini_api(formatted_prompt)
        prediction = -1
        if api_response_content is not None:
            if api_response_content.strip() == "1":
                prediction = 1
            elif api_response_content.strip() == "0":
                prediction = 0
            else:
                print(f"Unexpected Gemini response: '{api_response_content}'")
        raw_api_predictions.append(prediction)

    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]
    if not valid_indices:
        print("No valid predictions from Gemini.")
        return

    valid_samples = [{"url": prompts[i], "label": int(ground_truth_labels[i])} for i in valid_indices]
    y_pred_mapped = [raw_api_predictions[i] for i in valid_indices]
    y_true_mapped = [int(ground_truth_labels[i]) for i in valid_indices]

    print("\n--- Evaluation Metrics (Phishing = 1) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n--- Evaluation Metrics (Not Phishing = 0) ---")
    prec0 = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f10 = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    print(f"Precision: {prec0:.4f}")
    print(f"Recall: {rec0:.4f}")
    print(f"F1 Score: {f10:.4f}")

    try:
        tn, fp, fn, tp = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1]).ravel()
        print("\nConfusion Matrix (Actual rows, Predicted columns)")
        print("                     Pred Not Phish (0)  Pred Phish (1)")
        print(f"Actual Not Phish (0): {tn:<20}  {fp:<20}")
        print(f"Actual Phishing (1):  {fn:<20}  {tp:<20}")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")

    grouped = group_results(valid_samples, y_pred_mapped)
    write_grouped_results_to_csv(grouped)

if __name__ == "__main__":
    main()
