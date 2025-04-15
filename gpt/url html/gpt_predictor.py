import openai
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time

# ------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------
# IMPORTANT: Do not hardcode API keys in production code.
# Use environment variables or a secure key store instead.
OPENAI_API_KEY = ""
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Specify the model you want to use
OPENAI_MODEL = "gpt-4o-mini"  # Change this to another model if desired

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def call_openai_api(prompt_content):
    """
    Call the OpenAI Chat Completions API using the updated interface.
    This function sends the provided prompt (which is already formatted) to the API.
    Returns the response text.
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a phishing detection assistant. Your sole output should be the digit 0 or 1."
                },
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=10,      # Limit the response to a small number of tokens
            temperature=0.1     # Low temperature for more deterministic output
        )
        
        # Check if the response has 'choices' and that it's not empty.
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content.strip()
            return content
        else:
            print("Unexpected response structure from OpenAI:", response)
            return None

    except Exception as e:
        print(f"OpenAI API request failed: {e}")
        return None

def group_results(samples, predictions):
    """
    Group results into correct predictions, false positives, and false negatives.
    This function compares the API prediction with the ground truth (both as numbers, with:
      - 1 = phishing  
      - 0 = not phishing/legitimate)
    """
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred in zip(samples, predictions):
        # In this new file, no mapping inversion is needed because the ground truth
        # already uses the same convention as the API.
        mapped_pred = api_pred  # Use the prediction directly.
        gt = int(sample["label"])
        result = {
            "url": sample["url"],
            "label": gt,
            "api_prediction": api_pred,
            "mapped_prediction": mapped_pred
        }
        if mapped_pred == gt:
            grouped["correct"].append(result)
        else:
            # A false positive: predicted phishing (1) but ground truth is 0.
            if mapped_pred == 1 and gt == 0:
                grouped["false_positive"].append(result)
            # A false negative: predicted not phishing (0) but ground truth is 1.
            elif mapped_pred == 0 and gt == 1:
                grouped["false_negative"].append(result)
    return grouped

def write_grouped_results_to_csv(grouped_results):
    """
    Write each group into separate CSV files.
    """
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
# Main Script
# ------------------------------------------------------------

def main():
    # 1. Load the dataset from the output_100.txt file.
    # Each line should have the format: <prompt>,<ground_truth>
    print("Loading prompts from output_100.txt...")
    
    try:
        with open("output_100.txt", "r") as file:
            lines = file.readlines()
        prompts = []
        ground_truth_labels = []
        for line in lines:
            # Split from the last comma to avoid splitting the prompt itself
            parts = line.strip().rsplit(",", 1)
            if len(parts) == 2:
                prompt = parts[0].strip()  # Everything before the last comma is the prompt
                ground_truth = parts[1].strip()  # Everything after the last comma is the ground truth
                
                # Check if the ground truth is valid (0 or 1)
                if ground_truth in ["0", "1"]:
                    prompts.append(prompt)
                    ground_truth_labels.append(ground_truth)
                else:
                    print(f"Skipping invalid ground truth for prompt: {prompt}")
            else:
                # If the line does not contain exactly 2 parts (prompt, ground_truth), skip it
                print(f"Skipping malformed line: {line.strip()}")
                
        print(f"Successfully loaded {len(prompts)} prompts from the file.")
    except FileNotFoundError:
        print("File 'output_100.txt' not found.")
        return

    if not prompts:
        print("No valid prompts found in the file.")
        return

    # Initialize list to store API predictions.
    raw_api_predictions = []
    
    # 2. Process each sample with the OpenAI API.
    print("Calling OpenAI API for each prompt...")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        api_response_content = call_openai_api(prompt)
        prediction = -1  # Default for a failed API call.
        if api_response_content is not None:
            if api_response_content == "1":
                prediction = 1
            elif api_response_content == "0":
                prediction = 0
            else:
                print(f"Unexpected response content for prompt '{prompt}': '{api_response_content}'")
        raw_api_predictions.append(prediction)
        # Optional delay (to handle rate limits)
        time.sleep(1)

    # 3. Filter out samples with invalid predictions (where prediction == -1).
    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]
    
    if len(valid_indices) == 0:
        print("No valid predictions received from the API!")
        return

    # Retain only the valid samples.
    valid_samples = [{"url": prompts[i], "label": ground_truth_labels[i]} for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]
    y_true_mapped = [int(ground_truth_labels[i]) for i in valid_indices]  # Ground truth from file.
    y_pred_mapped = [int(prediction) for prediction in y_pred_api]

    # Optional: Check that lengths match.
    if len(y_true_mapped) != len(y_pred_mapped):
        print(f"Length mismatch: y_true_mapped has {len(y_true_mapped)} samples, but y_pred_mapped has {len(y_pred_mapped)} samples.")
        return

    # 4. Compute evaluation metrics.
    print("\n--- Evaluation Metrics (Positive Label = 1: Phishing) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n--- Evaluation Metrics (Positive Label = 0: Not Phishing/Legitimate) ---")
    prec0 = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f10 = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    print(f"Precision: {prec0:.4f}")
    print(f"Recall: {rec0:.4f}")
    print(f"F1 Score: {f10:.4f}")

    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print("\nConfusion Matrix (Rows: Actual, Columns: Predicted)")
        print("                     Predicted Not Phishing (0)  Predicted Phishing (1)")
        print(f"Actual Not Phishing (0): {tn:<20}  {fp:<20}")
        print(f"Actual Phishing (1):     {fn:<20}  {tp:<20}")
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
        print(f"False Negative Rate (FNR): {false_negative_rate:.4f}")
    except ValueError as e:
        print(f"\nCould not compute confusion matrix. Error: {e}")
        tn = fp = fn = tp = 0

    # 5. Group the results.
    grouped_results = group_results(valid_samples, y_pred_api)

    # 6. Write grouped results to CSV files.
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()
