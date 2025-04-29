#Imports
import requests # Changed from anthropic
import json     # json data
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import time
import zipfile
import os

# Put your api key here (redacted for codebase)
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions" # Endpoint for properly
DEEPSEEK_MODEL = "deepseek-chat" # we used deepseek chat for the predictions
MAX_TOKENS = 10 # Keep max tokens low since only needs to output 1 or 0

TOKEN_BUCKET_CAPACITY = 2 
TOKENS_PER_MINUTE = 2   
REFILL_RATE = TOKENS_PER_MINUTE / 60.0  # requests per sec.

current_tokens = TOKEN_BUCKET_CAPACITY
last_refill_time = time.time()

def refill_bucket():
    global current_tokens, last_refill_time
    now = time.time()
    elapsed = now - last_refill_time
    tokens_to_add = elapsed * REFILL_RATE
    if tokens_to_add > 0:
        # can't go over bucket amount max
        current_tokens = min(TOKEN_BUCKET_CAPACITY, current_tokens + tokens_to_add)
        last_refill_time = now

def acquire_token():
    global current_tokens
    while True:
        refill_bucket()
        if current_tokens >= 1:
            current_tokens -= 1
            return
        time.sleep(0.1)


# does the deepseek call with all the factors from above (rate limiting).
def call_deepseek_api(prompt_content, max_retries=5):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt_content}],
        "max_tokens": MAX_TOKENS
    }
    print(f"\nAttempting API call for prompt length: {len(prompt_content)}")


    for attempt in range(max_retries):
        print(f"--- Attempt {attempt + 1}/{max_retries} ---")
        try:
            # Get "query" token for request rate
            acquire_token()

            print(f"Sending payload to {DEEPSEEK_API_ENDPOINT}:")
            try:
                # if the len is too long then can deal with it here
                payload_to_print = payload.copy()
                if len(payload['messages'][0]['content']) > 300: # Print first/last chars
                     payload_to_print['messages'] = [{"role": "user", "content": payload['messages'][0]['content'][:150] + " " + payload['messages'][0]['content'][-150:]}]
                else:
                     payload_to_print['messages'] = [{"role": "user", "content": payload['messages'][0]['content']}]

                print(json.dumps(payload_to_print, indent=2))
            except Exception as print_e:
                print(f"(Could not print detailed payload: {print_e})")


            response = requests.post(
                DEEPSEEK_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60
            )

            print(f"Response Status Code: {response.status_code}")

            response.raise_for_status()

            response_data = response.json()


            # Get back what Deepseek said for each prompt, and make sure its structured right.
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message")
                if message and message.get("content"):
                    content = message["content"]
                    print(f"API call successful. Received content: '{content[:50]}'") 
                    return content.strip()
                else:
                    print(f"Unexpected response structure: 'message' or 'content' missing. Full Response: {response_data}")
                    return None
            else:
                print(f"Unexpected response structure: 'choices' missing or empty. Full Response: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API request failed (attempt {attempt+1}/{max_retries}). Error: {e}")
            if e.response is not None:
                print(f"Response Body: {e.response.text}") # for 404 errors

            status_code = e.response.status_code if e.response is not None else "N/A"
            if status_code == 429: #for rate limit
                print("Rate limti so Waiting 10 seconds then retrying.")
                time.sleep(10)
            elif attempt < max_retries - 1: # retry for other errors
                 print("Waiting 10 seconds before retrying.")
                 time.sleep(10)

        except Exception as e: # other possible errors (json)
            print(f"An unexpected error occurred during DeepSeek API call or processing (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Wait another 10 seconds then retrying.")
                time.sleep(10)

    print(f"Exceeded maximum retry attempts ({max_retries}) for this prompt.")
    return None

# groups the results from deepseek into correct, false positives, false negatives based on the truth val in the dataset.
def group_results(samples, predictions):
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred_str in zip(samples, predictions):
        
        try:
            mapped_pred = int(api_pred_str) if api_pred_str in ['0', '1'] else -1 # Map only '0' or '1'
        except (ValueError, TypeError):
             mapped_pred = -1 #invalid response

        # Skip if the pred not how it supposed to be
        if mapped_pred == -1:
            print(f"Skipping result grouping for URL {sample.get('url', 'N/A')} due to invalid API prediction: '{api_pred_str}'")
            continue

        gt = int(sample["label"])
        result = {
            "url": sample["url"],
            "label": gt,
            "api_prediction": api_pred_str, # raw string from API
            "mapped_prediction": mapped_pred # mapped int
        }

        if mapped_pred == gt:
            grouped["correct"].append(result)
        # this only checks the valid ones, FP/FN
        elif mapped_pred == 1 and gt == 0:
                grouped["false_positive"].append(result)
        elif mapped_pred == 0 and gt == 1:
                grouped["false_negative"].append(result)
    return grouped


def write_grouped_results_to_csv(grouped_results):
    """Writes the grouped results into separate CSV files."""
   
    for group, items in grouped_results.items():
        
        if not items:
            print(f"Skipping CSV for group '{group}' as it is empty.")
            continue

        filename = f"{group}_results.csv"
        
        fieldnames = ["url", "label", "api_prediction", "mapped_prediction"]
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for item in items:

                row_data = {field: item.get(field, 'N/A') for field in fieldnames}
                writer.writerow(row_data)
        print(f"Group '{group}' results saved to '{filename}'.")


def main():
    print("Loading in output_100.zip")
    prompts_data = []

    try:
        zip_path = "output_100.zip"
        extract_path = "./unzipped"
        txt_filename = "output_100.txt"
        filepath = os.path.join(extract_path, txt_filename)

        if not os.path.exists(extract_path):
            os.makedirs(extract_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Unzipped '{zip_path}' to '{extract_path}'.")

        if not os.path.exists(filepath):
             print(f"Error: Extracted file '{txt_filename}' not found in '{extract_path}'.")
             return

        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        processed_count = 0
        skipped_count = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                skipped_count += 1
                continue

            parts = line.rsplit(",", 1)
            if len(parts) == 2:
                url_html_content = parts[0].strip()
                ground_truth_str = parts[1].strip()
                if ground_truth_str in ["0", "1"]:
                    try:
                        ground_truth_int = int(ground_truth_str)
                        
                        prompts_data.append({"url_html": url_html_content, "label": ground_truth_int})
                        processed_count += 1
                    except ValueError:
                         print(f"Skipping line {i+1} due to non-integer ground truth: '{ground_truth_str}'")
                         skipped_count += 1
                else:
                    print(f"Skipping line {i+1} due to invalid ground truth value: '{ground_truth_str}' (Expected '0' or '1')")
                    skipped_count += 1
            else:
                print(f"Skipping malformed line {i+1}: Could not split into exactly two parts by the last comma. Line content: '{line[:100]}'")
                skipped_count += 1

        print(f"Successfully loaded {processed_count} prompts.")
        if skipped_count > 0:
             print(f"Skipped {skipped_count} lines due to formatting or invalid labels.")

    except FileNotFoundError:
        print(f"Error: File '{zip_path}' not found in the current directory.")
        return
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zip file or is corrupted.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return


    if not prompts_data:
        print("No valid prompts. Can't do anything")
        return

    raw_api_predictions_str = []
    print("\nCalling DeepSeek now for each propmpt.")

    # Define character limit for the URL+HTML so that there is less chance of errors, rl problems, etc.
    MAX_URL_HTML_CHARS = 150000
    KEEP_CHARS = MAX_URL_HTML_CHARS // 2

    for i, item in enumerate(tqdm(prompts_data, desc="Processing prompts")):
        url_html = item["url_html"] # Get the full URL+HTML content
        original_length = len(url_html)
        truncated = False

        if original_length > MAX_URL_HTML_CHARS:
            print(f"\n[INFO] Prompt {i+1} content length ({original_length}) exceeds limit ({MAX_URL_HTML_CHARS}). Truncating.")
            start_chunk = url_html[:KEEP_CHARS]
            end_chunk = url_html[-KEEP_CHARS:]
            # Add a clear marker that content was cut
            url_html = f"{start_chunk}\n\n... [CONTENT TRUNCATED DUE TO CONTEXT LENGTH LIMIT]\n\n{end_chunk}"
            truncated = True
            print(f"[INFO] Prompt {i+1} new truncated length: {len(url_html)}")

        # prompt for deepseek, takes into consideration if the url+html was truncated or not.
        formatted_prompt = (
            f"Given the following input (URL and potentially partial HTML content), "
            # Add note about potential truncation if it happened
            f"{'[Note: HTML content may be truncated due to length limits] ' if truncated else ''}"
            f"determine if it represents a phishing website or not:\n\n"
            f"--- Input Start ---\n"
            f"{url_html}\n" # Use the potentially truncated version
            f"--- Input End ---\n\n"
            "Is this phishing? Respond ONLY with the integer 1 (phishing) or 0 (not phishing)."
            " Do not provide any explanation or other text."
        )

        api_response_content = call_deepseek_api(formatted_prompt)

        # Store the raw response strin
        raw_api_predictions_str.append(api_response_content)

    print("Processing API Responses and Evaluating")

    valid_indices = []
    y_pred_mapped = [] # 0 or 1
    y_pred_api_raw = [] #'0' or '1'
    y_true_mapped = [] # int value, groud truth.

    for i, raw_pred_str in enumerate(raw_api_predictions_str):
        
        gt_label = prompts_data[i]["label"]

        # case where the raw str is either 0 or 1 as should be.
        if raw_pred_str is not None and raw_pred_str in ['0', '1']:
             valid_indices.append(i)
             y_pred_mapped.append(int(raw_pred_str))
             y_pred_api_raw.append(raw_pred_str)
             y_true_mapped.append(gt_label)
        else:
            url_info = prompts_data[i]['url_html'].split('\n', 1)[0]
            print(f"[WARN] Invalid or missing API response for prompt {i+1} (URL starting with: {url_info[:100]}). Raw response: '{raw_pred_str}'. Excluding from metrics.")

    if len(valid_indices) == 0:
        print("No valid predictions.")
        valid_samples_for_grouping = [] # Eempty list for grouping
        grouped = group_results(valid_samples_for_grouping, y_pred_api_raw)
        write_grouped_results_to_csv(grouped)
        return

    print(f"\nEvaluated {len(valid_indices)} prompts with valid API responses ('0' or '1').")

    # data prep
    valid_samples_for_grouping = []
    for i in valid_indices:
         url_identifier = prompts_data[i]["url_html"].split('\n', 1)[0]
         valid_samples_for_grouping.append({
             "url": url_identifier, # explanation: just using the url as identification
             "label": prompts_data[i]["label"] # The ground truth label
         })

    # Metrics.
    print("Evaluation Metrics (label = 1: Phishing)")
    acc = accuracy_score(y_true_mapped, y_pred_mapped)
    prec = precision_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    rec = recall_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    f1 = f1_score(y_true_mapped, y_pred_mapped, pos_label=1, zero_division=0)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Phishing): {prec:.4f}")
    print(f"Recall (Phishing): {rec:.4f}")
    print(f"F1 Score (Phishing): {f1:.4f}")

    print("\n--- Evaluation Metrics (Positive Label = 0: Not Phishing) ---")
    prec0 = precision_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    f10 = f1_score(y_true_mapped, y_pred_mapped, pos_label=0, zero_division=0)
    print(f"Precision (Not Phishing): {prec0:.4f}")
    print(f"Recall (Not Phishing): {rec0:.4f}")
    print(f"F1 Score (Not Phishing): {f10:.4f}")

    # Confusion Matrix.
    try:
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print("\nConfusion Matrix (Rows: Actual, Columns: Predicted)")
        print("                     Predicted Not Phishing (0)  Predicted Phishing (1)")
        print(f"Actual Not Phishing (0): {tn:<20}  {fp:<20}")
        print(f"Actual Phishing (1):     {fn:<20}  {tp:<20}")

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0 
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0 

        print(f"\nTrue Positive Rate (TPR/Recall-Phishing): {true_positive_rate:.4f}")
        print(f"False Positive Rate (FPR):                {false_positive_rate:.4f}")
        print(f"True Negative Rate (TNR/Specificity):     {true_negative_rate:.4f}")
        print(f"False Negative Rate (FNR):                {false_negative_rate:.4f}")

    except ValueError as e:
        print(f"\nCould not compute confusion matrix or related rates. Error: {e}")
        print(f"Unique True Labels: {sorted(list(set(y_true_mapped)))}")
        print(f"Unique Predicted Labels: {sorted(list(set(y_pred_mapped)))}")


    grouped = group_results(valid_samples_for_grouping, y_pred_api_raw)
    write_grouped_results_to_csv(grouped)

    print("\nScript finished.")

# checkpoint needed.
if __name__ == "__main__":
    main()