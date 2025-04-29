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
import random

# ------------------------------------------------------------
# Configuration and Constants for Together AI LLaMA
# ------------------------------------------------------------
TOGETHER_API_KEY = ""
LLAMA_MODEL = "meta-llama/Llama-3-8b-chat-hf"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
MAX_TOKENS = 10

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def build_single_prompt(item):
    """
    Build the content for the user message for a single URL.
    """
    prompt_content = (
        f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
        "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
    )
    return prompt_content

def call_llama_api(prompt_content):
    """
    Call the Llama API with the given prompt content.

    Returns the API's response content text.
    """
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    # Structure payload according to Llama API documentation
    payload = {
        "model": LLAMA_MODEL,
        "input": prompt_content,
        "max_tokens": 10,  # Keep low to encourage only "0" or "1"
        "temperature": 0.1  # Low temperature for more deterministic output
    }

    try:
        response = requests.post(LLAMA_API_ENDPOINT, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        # Parse the JSON response and extract the message content
        response_data = response.json()

        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'text' in choice:
                content = choice['text']
                return content.strip()
            else:
                print("Unexpected response structure (missing 'text'):", response_data)
                return None
        else:
            print("Unexpected response structure (missing 'choices'):", response_data)
            return None

    except requests.exceptions.Timeout:
        print("API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if response and response.status_code == 429:
            print("Rate limit hit. Consider adding a delay.")
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"Failed to parse API response: {e}")
        if 'response' in locals() and response is not None:
            print(f"Raw response text: {response.text}")
        return None

def group_results(samples, predictions):
    """
    Group results into correct, false positive, and false negative.
    """
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        # Map API prediction (0=Not Phishing, 1=Phishing) to Kaggle Label (1=Not Phishing, 0=Phishing)
        mapped_pred = 1 if api_pred == 0 else 0  # API (0=Not Phishing, 1=Phishing) -> Kaggle (1=Not Phishing, 0=Phishing)

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

def write_grouped_results_to_csv(grouped_results):
    """
    Write each group into separate CSV files: one for correct predictions,
    one for false positives, and one for false negatives.
    """
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

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
def main():
    # 1. Load the dataset (adjust this part according to how you're loading your dataset)
    df = pd.read_csv("new_data_urls.csv")  # Modify if using KaggleHub or another source

    # 2. Prepare samples.
    samples = []
    for _, row in df.iterrows():
        if row["url"] and isinstance(row["url"], str) and not pd.isna(row["status"]):
            try:
                label = int(row["status"])
                if label in [0, 1]:  # Ensure label is strictly 0 or 1
                    sample = {
                        "url": row["url"],
                        "label": label  # 0 indicates phishing, 1 indicates legitimate.
                    }
                    samples.append(sample)
            except ValueError:
                pass

    # Limit to the first N samples for testing
    sample_limit = 1000
    samples = samples[:sample_limit]

    # 3. Process each sample with a Llama API call.
    raw_api_predictions = []
    ground_truth_labels = []

    for item in tqdm(samples, desc="Processing samples"):
        prompt_content = build_single_prompt(item)
        api_response_content = call_llama_api(prompt_content)

        prediction = -1  # Default to -1 for errors or unexpected responses
        if api_response_content is not None:
            if api_response_content == "1":
                prediction = 1  # API says Phishing
            elif api_response_content == "0":
                prediction = 0  # API says Not Phishing
            else:
                print(f"Unexpected response content for URL '{item['url']}': '{api_response_content}'")

        raw_api_predictions.append(prediction)
        ground_truth_labels.append(item["label"])

    # 4. Filter out samples where prediction failed (prediction == -1).
    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]

    if len(valid_indices) == 0:
        print("No valid predictions received from the API!")
        return

    valid_samples = [samples[i] for i in valid_indices]
    y_true_mapped = [ground_truth_labels[i] for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]

    # Map API predictions to match Kaggle label semantics
    y_pred_mapped = [0 if pred == 1 else 1 for pred in y_pred_api]  # 1->0, 0->1

    # 5. Compute Evaluation Metrics.
    print("\n--- Evaluation Metrics ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec_legit = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec_legit = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1_legit = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)

    prec_phish = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec_phish = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f1_phish = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Legit): {prec_legit:.4f}")
    print(f"Recall (Legit): {rec_legit:.4f}")
    print(f"F1 (Legit): {f1_legit:.4f}")
    print(f"Precision (Phishing): {prec_phish:.4f}")
    print(f"Recall (Phishing): {rec_phish:.4f}")
    print(f"F1 (Phishing): {f1_phish:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Group results into CSV files
    grouped_results = group_results(valid_samples, y_pred_api)
    write_grouped_results_to_csv(grouped_results)

if __name__


