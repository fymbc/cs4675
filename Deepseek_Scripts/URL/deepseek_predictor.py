import math
import requests
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time # Import time for potential rate limiting/retries

# ------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------
DEEPSEEK_API_KEY = "sk-bf14f88d01d44b568bb8140b9687b6eb" # Replace with your actual key

# Corrected DeepSeek API Endpoint for chat completions
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions" # Corrected Endpoint

# Specify the model you want to use
DEEPSEEK_MODEL = "deepseek-chat" # Or "deepseek-coder" if more appropriate, check DeepSeek docs

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

def call_deepseek_api(prompt_content):
    """
    Call the DeepSeek Chat Completions API with the given prompt content.

    Uses the standard chat completions format.
    Returns the API's response content text.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    # Structure payload according to DeepSeek Chat Completions API documentation
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            # You could add a system message here if desired
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_content}
        ],
        "max_tokens": 10,  # Keep low to encourage only "0" or "1"
        "temperature": 0.1 # Low temperature for more deterministic output
        # Add other parameters like 'stream', 'frequency_penalty' etc. if needed
    }

    try:
        response = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers, timeout=60)
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the JSON response and extract the message content
        response_data = response.json()

        # Add checks for expected response structure
        if 'choices' in response_data and len(response_data['choices']) > 0:
            choice = response_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                 content = choice['message']['content']
                 return content.strip()
            else:
                 print("Unexpected response structure (missing message/content):", response_data)
                 return None # Indicate an issue with parsing
        else:
             print("Unexpected response structure (missing choices):", response_data)
             return None # Indicate an issue with parsing

    # Catch specific exceptions for better error handling
    except requests.exceptions.Timeout:
        print("API request timed out.")
        return None # Indicate timeout
    except requests.exceptions.RequestException as e:
        # This catches connection errors, HTTP errors (already raised by raise_for_status), etc.
        print(f"API request failed: {e}")
        # If it's a 429 (Too Many Requests), you might want to wait and retry
        if response is not None and response.status_code == 429:
             print("Rate limit hit. Consider adding a delay.")
             # Example: time.sleep(5) # Wait 5 seconds
        return None # Indicate failure
    except (KeyError, IndexError, TypeError) as e:
        # Catch errors during JSON parsing or accessing nested keys
        print(f"Failed to parse API response: {e}")
        if 'response' in locals() and response is not None:
            print(f"Raw response text: {response.text}")
        return None # Indicate parsing failure


def group_results(samples, predictions):
    """
    Group results into correct, false positive, and false negative.
    Each result is a dict with the "url", "label" (ground truth),
    and "prediction" (from DeepSeek API).

    Note: Kaggle dataset: status=0 indicates phishing, status=1 indicates legitimate.
    The prompt asks for: 1 (PHISHING) OR 0 (NOT PHISHING).
    Therefore, we need to map the API output (0/1) to the Kaggle labels (1/0).
    API Prediction: 1 (Phishing)  -> Matches Kaggle Label: 0 (Phishing)
    API Prediction: 0 (Not Phishing) -> Matches Kaggle Label: 1 (Legitimate)
    """
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        # Map API prediction (0=Not Phishing, 1=Phishing) to Kaggle Label (1=Not Phishing, 0=Phishing)
        # If API predicts 1 (Phishing), mapped_pred is 0
        # If API predicts 0 (Not Phishing), mapped_pred is 1
        if api_pred == 1:
            mapped_pred = 0 # API said Phishing (1), which matches Kaggle Label 0
        elif api_pred == 0:
            mapped_pred = 1 # API said Not Phishing (0), which matches Kaggle Label 1
        else:
             # This case handles errors (-1) or unexpected values
             # We might want to exclude these from grouping or put in a separate 'error' group
             print(f"Skipping grouping for URL {sample['url']} due to invalid prediction {api_pred}")
             continue # Skip this sample for grouping

        result = {
            "url": sample["url"],
            "label": sample["label"],      # Ground truth (0=Phishing, 1=Legitimate)
            "api_prediction": api_pred,    # Raw API output (0=Not Phishing, 1=Phishing)
            "mapped_prediction": mapped_pred # API output mapped to Kaggle label meaning
        }

        # Compare the mapped prediction to the ground truth label
        if mapped_pred == sample["label"]:
            grouped["correct"].append(result)
        else:
            # False Positive: Predicted Legitimate (mapped_pred=1), but was Phishing (label=0)
            # This happens when API predicts 0 (Not Phishing)
            if mapped_pred == 1 and sample["label"] == 0:
                grouped["false_positive"].append(result)
            # False Negative: Predicted Phishing (mapped_pred=0), but was Legitimate (label=1)
            # This happens when API predicts 1 (Phishing)
            elif mapped_pred == 0 and sample["label"] == 1:
                grouped["false_negative"].append(result)
            # Add an else here just in case, though it shouldn't be logically reachable
            else:
                print(f"Unexpected grouping condition: Label={sample['label']}, MappedPred={mapped_pred}")

    return grouped

def write_grouped_results_to_csv(grouped_results):
    """
    Write each group into separate CSV files: one for correct predictions,
    one for false positives, and one for false negatives.
    Each CSV file will have columns: group, url, label, api_prediction, mapped_prediction.
    """
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Added api_prediction and mapped_prediction for clarity
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
    # 1. Load the dataset from KaggleHub using the correct method.
    print("Loading dataset from KaggleHub...")
    df = None # Initialize df to None
    file_path = "new_data_urls.csv" # The specific file within the dataset

    try:
        # --- Use the original, correct method for loading datasets ---
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS, # Specify you want a Pandas DataFrame
            "harisudhan411/phishing-and-legitimate-urls", # The dataset handle (owner/dataset-slug)
            file_path, # The path to the specific file within the dataset
        )
        print("Successfully loaded dataset using kagglehub.load_dataset.")
        # --- End original method ---

    except Exception as e:
        print(f"Error loading dataset using kagglehub.load_dataset: {e}")
        print("Ensure:")
        print("  1. Kaggle API credentials (kaggle.json) are correctly configured.")
        print("  2. The dataset handle 'harisudhan411/phishing-and-legitimate-urls' is correct.")
        print(f"  3. The file path '{file_path}' exists within that dataset on Kaggle.")
        print("  4. You have the necessary libraries installed/updated (`pip install --upgrade kagglehub pandas`).")

        # --- Fallback: Try loading directly if file is already local ---
        print(f"\nAttempting to load '{file_path}' from the current directory as a fallback...")
        try:
             df = pd.read_csv(file_path)
             print("Successfully loaded from local directory.")
        except FileNotFoundError:
             print(f"Fallback failed: '{file_path}' not found locally.")
             return # Exit if dataset cannot be loaded
        except Exception as e_local:
             print(f"Error loading local file '{file_path}': {e_local}")
             return # Exit if dataset cannot be loaded
        # --- End Fallback ---

    # Exit if DataFrame could not be loaded by any method
    if df is None:
        print("DataFrame could not be loaded. Exiting.")
        return

    print("Total rows in dataset:", len(df))

    # Check columns - adjust if needed based on the actual CSV header
    if "url" not in df.columns or "status" not in df.columns:
         print("Error: Expected columns 'url' and 'status' not found in the CSV.")
         print("Available columns:", df.columns)
         return

    
    # --- ADDED SHUFFLING ---
    print("Shuffling the DataFrame rows...")
    # frac=1 means sample 100% of the rows (i.e., shuffle all)
    # reset_index(drop=True) resets the index to 0, 1, 2,... after shuffling
    # random_state makes the shuffle reproducible; remove it for different randomness each run
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("DataFrame successfully shuffled.")
    # --- END SHUFFLING ---

    # 2. Prepare samples.
    samples = []
    for _, row in df.iterrows():
        # Add extra checks for valid data
        if row["url"] and isinstance(row["url"], str) and not pd.isna(row["status"]):
            try:
                label = int(row["status"])
                if label in [0, 1]: # Ensure label is strictly 0 or 1
                     sample = {
                         "url": row["url"],
                         "html": "",  # No HTML available in this dataset.
                         "label": label  # 0 indicates phishing, 1 indicates legitimate.
                     }
                     samples.append(sample)
                # else: skip rows with invalid status silently
            except ValueError:
                 # skip rows with non-integer status silently
                 pass
        # else: skip rows with missing URL or status silently

    # Limit to the first N samples for testing
    sample_limit = 1000 # Adjust as needed
    samples = samples[:sample_limit]
    print(f"Prepared {len(samples)} valid samples for evaluation (limited to first {sample_limit}).")

    if not samples:
        print("No valid samples found to process.")
        return

    # 3. Process each sample with a DeepSeek API call.
    raw_api_predictions = [] # Store the direct 0/1 output from API
    ground_truth_labels = [] # Store the corresponding 0/1 Kaggle labels

    print("Calling DeepSeek API for each sample...")
    for item in tqdm(samples, desc="Processing samples"):
        prompt_content = build_single_prompt(item)
        api_response_content = call_deepseek_api(prompt_content) # Returns content or None on error

        prediction = -1 # Default to -1 for errors or unexpected responses
        if api_response_content is not None:
             # Expect the API to return a single digit string ("0" or "1").
             if api_response_content == "1":
                 prediction = 1 # API says Phishing
             elif api_response_content == "0":
                 prediction = 0 # API says Not Phishing
             else:
                 print(f"Unexpected response content for URL '{item['url']}': '{api_response_content}'")
                 # Keep prediction as -1
        # else: Error already printed in call_deepseek_api

        raw_api_predictions.append(prediction)
        ground_truth_labels.append(item["label"]) # Store the Kaggle label (0=Phishing, 1=Legit)

        # Optional: Add a small delay to avoid hitting rate limits aggressively
        # time.sleep(0.1) # Sleep for 100ms between calls

    # 4. Filter out samples where prediction failed (prediction == -1).
    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]

    if len(valid_indices) == 0:
        print("No valid predictions received from the API!")
        return

    valid_samples = [samples[i] for i in valid_indices]
    # Ground truth labels for the valid predictions (0=Phishing, 1=Legit)
    y_true_mapped = [ground_truth_labels[i] for i in valid_indices]
    # Raw API predictions for the valid predictions (0=Not Phishing, 1=Phishing)
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]

    # Map API predictions to match Kaggle label semantics for metric calculation
    # API: 1=Phishing (maps to Kaggle=0), 0=Not Phishing (maps to Kaggle=1)
    # Mapped: 0=Phishing, 1=Not Phishing (Legitimate)
    y_pred_mapped = [0 if pred == 1 else 1 for pred in y_pred_api] # 1->0, 0->1

    print(f"\nProcessed {len(samples)} samples, received {len(valid_indices)} valid predictions.")

    # 5. Compute Evaluation Metrics.
    # Calculate metrics based on mapped predictions (y_pred_mapped) vs ground truth (y_true_mapped).
    # Labels used for metrics: 0=Phishing, 1=Legitimate

    # Calculate metrics where the positive label is 1 (Legitimate)
    print("\n--- Evaluation Metrics (Positive Label = 1: Legitimate) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec_legit = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec_legit = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1_legit = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)

    # Calculate metrics where the positive label is 0 (Phishing)
    print("\n--- Evaluation Metrics (Positive Label = 0: Phishing) ---")
    prec_phish = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec_phish = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f1_phish = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)

    # Confusion Matrix: labels=[0, 1] means rows/cols are Phishing, Legitimate
    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        # TN: Actual=0, Pred=0 | FP: Actual=0, Pred=1
        # FN: Actual=1, Pred=0 | TP: Actual=1, Pred=1
        tn, fp, fn, tp = cm.ravel()
        print("\nConfusion Matrix (Rows: Actual, Cols: Predicted)")
        print("                     Predicted Phishing (0)  Predicted Legit (1)")
        print(f"Actual Phishing (0)      {tn:<20}  {fp:<20}")
        print(f"Actual Legit (1)         {fn:<20}  {tp:<20}")

        # False Positive Rate (FPR): FP / (FP + TN) - Proportion of actual negatives (Phishing) incorrectly identified as positive (Legit).
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        # False Negative Rate (FNR): FN / (FN + TP) - Proportion of actual positives (Legit) incorrectly identified as negative (Phishing).
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    except ValueError as e:
        print(f"\nCould not compute confusion matrix. Maybe only one class present in results? Error: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0
        false_positive_rate = 0
        false_negative_rate = 0


    print("\n--- Overall & Specific Metrics ---")
    print(f"Accuracy:                     {acc:.4f}")
    print(f"Precision (Legitimate=1):     {prec_legit:.4f}")
    print(f"Recall (Legitimate=1):        {rec_legit:.4f}")
    print(f"F1 Score (Legitimate=1):      {f1_legit:.4f}")
    print(f"Precision (Phishing=0):       {prec_phish:.4f}")
    print(f"Recall (Phishing=0):          {rec_phish:.4f}")
    print(f"F1 Score (Phishing=0):        {f1_phish:.4f}")
    print(f"False Positive Rate (FPR):    {false_positive_rate:.4f} (Rate of actual Phishing classified as Legit)")
    print(f"False Negative Rate (FNR):    {false_negative_rate:.4f} (Rate of actual Legit classified as Phishing)")

    results_filename = "results.csv"
    print(f"\nSaving aggregate results to {results_filename}...")

    # Prepare data for CSV using the calculated metric variables
    results_data = [
        {"Metric": "Sample Limit Attempted", "Value": sample_limit},
        {"Metric": "Valid Predictions Obtained", "Value": len(valid_samples)},
        {"Metric": "Accuracy", "Value": f"{acc:.4f}"},
        {"Metric": "Precision (Legitimate=1)", "Value": f"{prec_legit:.4f}"},
        {"Metric": "Recall (Legitimate=1)", "Value": f"{rec_legit:.4f}"},
        {"Metric": "F1 Score (Legitimate=1)", "Value": f"{f1_legit:.4f}"},
        {"Metric": "Precision (Phishing=0)", "Value": f"{prec_phish:.4f}"},
        {"Metric": "Recall (Phishing=0)", "Value": f"{rec_phish:.4f}"},
        {"Metric": "F1 Score (Phishing=0)", "Value": f"{f1_phish:.4f}"},
        {"Metric": "True Negatives (TN - Actual Phish, Pred Phish)", "Value": tn},
        {"Metric": "False Positives (FP - Actual Phish, Pred Legit)", "Value": fp},
        {"Metric": "False Negatives (FN - Actual Legit, Pred Phish)", "Value": fn},
        {"Metric": "True Positives (TP - Actual Legit, Pred Legit)", "Value": tp},
        {"Metric": "False Positive Rate (FPR)", "Value": f"{false_positive_rate:.4f}"},
        {"Metric": "False Negative Rate (FNR)", "Value": f"{false_negative_rate:.4f}"},
    ]

    try:
        with open(results_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Metric", "Value"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results_data) # Write all rows from the list of dicts
        print(f"Aggregate results successfully saved to {results_filename}.")

    except IOError as e:
        print(f"Error writing aggregate results to {results_filename}: {e}")

    # 6. Group results into correct, false positive, and false negative.
    # Pass the valid original samples and the raw API predictions (y_pred_api) to group_results
    # The group_results function handles the mapping internally for comparison.
    grouped_results = group_results(valid_samples, y_pred_api)

    # 7. Write grouped results into separate CSV files.
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()