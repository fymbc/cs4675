#imports
import anthropic
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time
import zipfile
import os

# Config
ANTHROPIC_API_KEY = ""
MAX_TOKENS = 10
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Token bucket rate limiters
TOKEN_BUCKET_CAPACITY = 2
TOKENS_PER_MINUTE = 2
REFILL_RATE = TOKENS_PER_MINUTE / 60.0
current_tokens = TOKEN_BUCKET_CAPACITY
last_refill_time = time.time()

# Refill token bucket
def refill_bucket():
    global current_tokens, last_refill_time
    now = time.time()
    elapsed = now - last_refill_time
    tokens_to_add = elapsed * REFILL_RATE
    if tokens_to_add > 0:
        current_tokens = min(TOKEN_BUCKET_CAPACITY, current_tokens + tokens_to_add)
        last_refill_time = now

# Get a token
def acquire_token():
    global current_tokens
    while True:
        refill_bucket()
        if current_tokens >= 1:
            current_tokens -= 1
            return
        time.sleep(0.1)

# Call Anthropic API with retries
def call_anthropic_api(prompt_content, max_retries=5):
    for attempt in range(max_retries):
        try:
            acquire_token()
            response_message = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_content}]
            )
            content = response_message.content
            if isinstance(content, list) and content:
                content = getattr(content[0], "text", str(content[0]))
            elif hasattr(content, "text"):
                content = content.text
            if not isinstance(content, str):
                content = str(content)
            return content.strip()
        except anthropic.RateLimitError:
            print(f"Rate limit error (attempt {attempt+1}/{max_retries}). Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"API request failed: {e}")
            return None
    print("Exceeded maximum retries.")
    return None

# Group prediction results
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

# Save grouped results to CSV
def write_grouped_results_to_csv(grouped_results):
    for group, items in grouped_results.items():
        filename = f"{group}_results.csv"
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["group", "url", "label", "api_prediction", "mapped_prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        print(f"Saved {group} results to '{filename}'.")

def main():
    print("Unzipping prompts...")
    try:
        with zipfile.ZipFile("output_100.zip", 'r') as zip_ref:
            zip_ref.extractall("./unzipped")
        filepath = os.path.join("./unzipped", "output_100.txt")
        with open(filepath, "r") as file:
            lines = file.readlines()
        prompts, ground_truth_labels = [], []
        for line in lines:
            parts = line.strip().rsplit(",", 1)
            if len(parts) == 2:
                prompt, label = parts
                if label in ["0", "1"]:
                    prompts.append(prompt.strip())
                    ground_truth_labels.append(label.strip())
                else:
                    print(f"Skipping invalid label: {prompt}")
            else:
                print(f"Skipping malformed line: {line.strip()}")
        print(f"Loaded {len(prompts)} prompts.")
    except FileNotFoundError:
        print("output_100.zip or output_100.txt not found.")
        return

    if not prompts:
        print("No valid prompts loaded.")
        return

    raw_api_predictions = []
    print("Calling API...")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        formatted_prompt = (
            f"Given the following input, determine if it is a phishing website or not:\n"
            f"{prompt}\n"
            "Respond ONLY with 1 (phishing) or 0 (not phishing)."
        )
        api_response_content = call_anthropic_api(formatted_prompt)
        prediction = -1
        if api_response_content == "1":
            prediction = 1
        elif api_response_content == "0":
            prediction = 0
        else:
            print(f"Unexpected response for prompt '{prompt}': {api_response_content}")
        raw_api_predictions.append(prediction)

    valid_indices = [i for i, pred in enumerate(raw_api_predictions) if pred != -1]
    if not valid_indices:
        print("No valid predictions.")
        return

    valid_samples = [{"url": prompts[i], "label": int(ground_truth_labels[i])} for i in valid_indices]
    y_pred_api = [raw_api_predictions[i] for i in valid_indices]
    y_true_mapped = [int(ground_truth_labels[i]) for i in valid_indices]
    y_pred_mapped = [int(pred) for pred in y_pred_api]

    print("\n--- Metrics (Positive = Phishing) ---")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n--- Metrics (Positive = Not Phishing) ---")
    prec0 = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f10 = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    print(f"Precision: {prec0:.4f}")
    print(f"Recall: {rec0:.4f}")
    print(f"F1 Score: {f10:.4f}")

    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print("\nConfusion Matrix")
        print(f"Actual 0 → Pred 0: {tn}, Pred 1: {fp}")
        print(f"Actual 1 → Pred 0: {fn}, Pred 1: {tp}")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
    except Exception as e:
        print(f"Confusion matrix error: {e}")

    grouped_results = group_results(valid_samples, y_pred_api)
    write_grouped_results_to_csv(grouped_results)

if __name__ == "__main__":
    main()