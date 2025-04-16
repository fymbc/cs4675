import anthropic
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time
import zipfile
import os

# ------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------
ANTHROPIC_API_KEY = ""
MAX_TOKENS = 10
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ------------------------------------------------------------
# Token Bucket Variables
# ------------------------------------------------------------
# We allow 2 queries per minute, assuming each query can consume up to ~20k tokens
TOKEN_BUCKET_CAPACITY = 2         # Max queries "burst" in one go
TOKENS_PER_MINUTE = 2            # Refill 2 tokens per 60 seconds
REFILL_RATE = TOKENS_PER_MINUTE / 60.0  # tokens per second

current_tokens = TOKEN_BUCKET_CAPACITY
last_refill_time = time.time()

def refill_bucket():
    """
    Refill the bucket based on how much time has passed since last refill.
    """
    global current_tokens, last_refill_time
    now = time.time()
    elapsed = now - last_refill_time
    # Determine how many tokens to add based on elapsed time and refill rate
    tokens_to_add = elapsed * REFILL_RATE
    if tokens_to_add > 0:
        # Do not exceed the bucket capacity
        current_tokens = min(TOKEN_BUCKET_CAPACITY, current_tokens + tokens_to_add)
        last_refill_time = now

def acquire_token():
    """
    Acquire a single token (i.e., permission to make 1 API query) from the bucket.
    If no tokens are available, wait until at least one becomes available.
    """
    global current_tokens
    while True:
        refill_bucket()
        if current_tokens >= 1:
            current_tokens -= 1
            return
        # Sleep briefly, then check again
        time.sleep(0.1)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def call_anthropic_api(prompt_content, max_retries=5):
    """
    Calls the Anthropic API with a refined strategy:
      - Acquire a token before each call to stay under ~40k tokens/min limit.
      - Retry up to 'max_retries' times on RateLimitError or other errors.
    """
    for attempt in range(max_retries):
        try:
            # Acquire a "query" token to avoid hitting 40k tokens/min
            acquire_token()

            response_message = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "user", "content": prompt_content}
                ]
            )
            content = response_message.content
            # Normalize the response content
            if isinstance(content, list):
                if len(content) > 0:
                    first_item = content[0]
                    if hasattr(first_item, "text"):
                        content = first_item.text
                    else:
                        content = str(first_item)
                else:
                    content = ""
            elif hasattr(content, "text"):
                content = content.text

            if not isinstance(content, str):
                content = str(content)
            return content.strip()

        except anthropic.RateLimitError as e:
            # If we somehow still hit a rate limit, wait a bit, then retry
            print(f"Rate limit error encountered (attempt {attempt+1}/{max_retries}). Waiting 10s then retrying...")
            time.sleep(10)
        except Exception as e:
            print(f"Anthropic API request failed: {e}")
            return None

    print("Exceeded maximum retry attempts due to rate limiting or other errors.")
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
        else:
            if mapped_pred == 1 and gt == 0:
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
# Main Script
# ------------------------------------------------------------

def main():
    print("Unzipping and loading prompts from output_100.zip...")

    try:
        with zipfile.ZipFile("output_100.zip", 'r') as zip_ref:
            zip_ref.extractall("./unzipped")
        print("Unzipping completed.")
        filepath = os.path.join("./unzipped", "output_100.txt")
        with open(filepath, "r") as file:
            lines = file.readlines()
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
        print(f"Successfully loaded {len(prompts)} prompts from the file.")
    except FileNotFoundError:
        print("File 'output_100.zip' or 'output_100.txt' not found.")
        return

    if not prompts:
        print("No valid prompts found in the file.")
        return

    raw_api_predictions = []
    print("Calling Anthropic API for each prompt...")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        formatted_prompt = (
            f"Given the following input, determine if it is a phishing website or not:\n"
            f"{prompt}\n"
            "Respond ONLY with 1 (phishing) or 0 (not phishing)."
        )
        api_response_content = call_anthropic_api(formatted_prompt)
        prediction = -1
        if api_response_content is not None:
            if api_response_content == "1":
                prediction = 1
            elif api_response_content == "0":
                prediction = 0
            else:
                print(f"Unexpected response content for prompt '{prompt}': '{api_response_content}'")
        raw_api_predictions.append(prediction)
        # Because we have a token bucket controlling rate, we do NOT necessarily need an extra sleep here.

    # Filter out invalid predictions
    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]
    if len(valid_indices) == 0:
        print("No valid predictions received from the API!")
        return

    valid_samples = [{"url": prompts[i], "label": int(ground_truth_labels[i])} for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]
    y_true_mapped = [int(ground_truth_labels[i]) for i in valid_indices]
    y_pred_mapped = [int(prediction) for prediction in y_pred_api]

    print("\n--- Evaluation Metrics (Positive Label = 1: Phishing) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n--- Evaluation Metrics (Positive Label = 0: Not Phishing) ---")
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

    grouped_results = group_results(valid_samples, y_pred_api)
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()
