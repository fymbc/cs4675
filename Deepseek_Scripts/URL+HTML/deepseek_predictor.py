import requests # Changed from anthropic
import json     # Added for handling JSON data
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
# DeepSeek Credentials and Configuration
DEEPSEEK_API_KEY = "sk-bf14f88d01d44b568bb8140b9687b6eb" # Your actual key
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions" # DeepSeek Chat Completions Endpoint
DEEPSEEK_MODEL = "deepseek-chat" # Specify the model

# Shared Configuration
MAX_TOKENS = 10 # Max tokens for the response (0 or 1)

# Removed Anthropic client initialization:
# client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ------------------------------------------------------------
# Token Bucket Variables (Rate Limiting - unchanged logic)
# ------------------------------------------------------------
# We allow 2 queries per minute
TOKEN_BUCKET_CAPACITY = 2         # Max queries "burst" in one go
TOKENS_PER_MINUTE = 2            # Refill 2 requests per 60 seconds
REFILL_RATE = TOKENS_PER_MINUTE / 60.0  # requests per second

current_tokens = TOKEN_BUCKET_CAPACITY
last_refill_time = time.time()

def refill_bucket():
    """
    Refill the bucket based on how much time has passed since last refill.
    """
    global current_tokens, last_refill_time
    now = time.time()
    elapsed = now - last_refill_time
    # Determine how many tokens (request allowances) to add
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

import requests
import json
import time

# Assume DEEPSEEK_API_KEY, DEEPSEEK_API_ENDPOINT, DEEPSEEK_MODEL, MAX_TOKENS
# and acquire_token() are defined elsewhere as in your previous script.

# --- Replace the existing call_deepseek_api function with this one ---

def call_deepseek_api(prompt_content, max_retries=5):
    """
    Calls the DeepSeek API with rate limiting, retries, simplified payload,
    and enhanced error logging.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    # Simplified payload - removed temperature and stream
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt_content}],
        "max_tokens": MAX_TOKENS
        # "temperature": 0.1, # Removed for troubleshooting
        # "stream": False # Removed for troubleshooting
    }

    # DEBUG: Print the prompt hash or length to identify which one fails easily
    # import hashlib
    # prompt_hash = hashlib.md5(prompt_content.encode()).hexdigest()[:8]
    # print(f"\nAttempting API call for prompt hash: {prompt_hash}")
    # Or just print length:
    print(f"\nAttempting API call for prompt length: {len(prompt_content)}")


    for attempt in range(max_retries):
        print(f"--- Attempt {attempt + 1}/{max_retries} ---")
        try:
            # Acquire a "query" token to control request rate
            acquire_token()

            # DEBUG: Print payload just before sending (be mindful of large prompts in logs)
            print(f"Sending payload to {DEEPSEEK_API_ENDPOINT}:")
            try:
                # Truncate message content for printing if too long
                payload_to_print = payload.copy()
                if len(payload['messages'][0]['content']) > 300: # Print first/last chars
                     payload_to_print['messages'] = [{"role": "user", "content": payload['messages'][0]['content'][:150] + "..." + payload['messages'][0]['content'][-150:]}]
                else:
                     payload_to_print['messages'] = [{"role": "user", "content": payload['messages'][0]['content']}]

                print(json.dumps(payload_to_print, indent=2))
            except Exception as print_e:
                print(f"(Could not print detailed payload: {print_e})")


            response = requests.post(
                DEEPSEEK_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60 # Add a timeout (in seconds)
            )

            # DEBUG: Print status code always
            print(f"Response Status Code: {response.status_code}")

            # Check for HTTP errors (4xx or 5xx)
            response.raise_for_status() # This will raise an exception for 4xx/5xx

            response_data = response.json()
            # DEBUG: Print successful response data
            # print("Successful Response Data:")
            # print(json.dumps(response_data, indent=2))


            # Extract content - Adjust based on actual DeepSeek API response structure
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message")
                if message and message.get("content"):
                    content = message["content"]
                    print(f"API call successful. Received content: '{content[:50]}...'") # Log success
                    return content.strip()
                else:
                    print(f"Unexpected response structure: 'message' or 'content' missing. Full Response: {response_data}")
                    return None
            else:
                print(f"Unexpected response structure: 'choices' missing or empty. Full Response: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API request failed (attempt {attempt+1}/{max_retries}). Error: {e}")
            # DEBUG: Print response body text if available, this often has the specific error message
            if e.response is not None:
                print(f"Response Body: {e.response.text}") # <<< THIS IS VERY IMPORTANT FOR 400 ERRORS

            # Retry logic (same as before)
            status_code = e.response.status_code if e.response is not None else "N/A"
            if status_code == 429: # Specific handling for rate limit
                print("Rate limit error (429) encountered. Waiting 10s then retrying...")
                time.sleep(10)
            elif attempt < max_retries - 1: # General retry for other request errors
                 print("Waiting 10s before retrying...")
                 time.sleep(10)
            # Else (last attempt), let the loop end

        except Exception as e: # Catch other potential errors like JSON parsing
            print(f"An unexpected error occurred during DeepSeek API call or processing (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("Waiting 10s before retrying...")
                time.sleep(10)

    print(f"Exceeded maximum retry attempts ({max_retries}) for this prompt.")
    return None

# --- Make sure the rest of your script (main, etc.) calls this modified function ---

def group_results(samples, predictions):
    """Groups results into correct, false positive, and false negative."""
    # --- This function remains unchanged ---
    grouped = {"correct": [], "false_positive": [], "false_negative": []}
    for sample, api_pred_str in zip(samples, predictions):
        # Handle potential non-integer API responses gracefully
        try:
            mapped_pred = int(api_pred_str) if api_pred_str in ['0', '1'] else -1 # Map only '0' or '1'
        except (ValueError, TypeError):
             mapped_pred = -1 # Treat unexpected responses as invalid

        # Skip processing if the prediction wasn't valid (0 or 1)
        if mapped_pred == -1:
            print(f"Skipping result grouping for URL {sample.get('url', 'N/A')} due to invalid API prediction: '{api_pred_str}'")
            continue

        gt = int(sample["label"]) # Ground truth should already be int
        result = {
            "url": sample["url"],
            "label": gt,
            "api_prediction": api_pred_str, # Store the raw string from API
            "mapped_prediction": mapped_pred # Store the mapped int (0 or 1)
        }

        if mapped_pred == gt:
            grouped["correct"].append(result)
        # Only consider valid mapped predictions for FP/FN
        elif mapped_pred == 1 and gt == 0:
                grouped["false_positive"].append(result)
        elif mapped_pred == 0 and gt == 1:
                grouped["false_negative"].append(result)
    return grouped


def write_grouped_results_to_csv(grouped_results):
    """Writes the grouped results into separate CSV files."""
    # --- This function remains unchanged ---
    for group, items in grouped_results.items():
        # Only create CSV if there are items in the group
        if not items:
            print(f"Skipping CSV for group '{group}' as it is empty.")
            continue

        filename = f"{group}_results.csv"
        # Ensure all potential keys exist in fieldnames
        fieldnames = ["url", "label", "api_prediction", "mapped_prediction"]
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # Ignore extra keys if any
            writer.writeheader()
            for item in items:
                 # Ensure item has all keys expected by fieldnames, provide default if missing
                row_data = {field: item.get(field, 'N/A') for field in fieldnames}
                writer.writerow(row_data)
        print(f"Group '{group}' results saved to '{filename}'.")

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------

def main():
    print("Unzipping and loading prompts from output_100.zip...")
    prompts_data = [] # Store dictionaries {url_html: ..., label: ...}

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
             # Attempt to clean up extraction directory (optional, be careful)
             # try:
             #     if os.path.exists(extract_path):
             #         import shutil
             #         shutil.rmtree(extract_path)
             #         print(f"Cleaned up directory '{extract_path}'.")
             # except OSError as e:
             #     print(f"Error cleaning up directory '{extract_path}': {e}")
             return

        with open(filepath, "r", encoding="utf-8") as file: # Added encoding
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
                        # Store as dict including the original content
                        prompts_data.append({"url_html": url_html_content, "label": ground_truth_int})
                        processed_count += 1
                    except ValueError:
                         print(f"Skipping line {i+1} due to non-integer ground truth: '{ground_truth_str}'")
                         skipped_count += 1
                else:
                    print(f"Skipping line {i+1} due to invalid ground truth value: '{ground_truth_str}' (Expected '0' or '1')")
                    skipped_count += 1
            else:
                print(f"Skipping malformed line {i+1}: Could not split into exactly two parts by the last comma. Line content: '{line[:100]}...'")
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
    # Optional: Clean up extracted files (consider if needed)
    # finally:
    #      try:
    #          if os.path.exists(extract_path):
    #              import shutil
    #              shutil.rmtree(extract_path)
    #              print(f"Cleaned up extracted directory '{extract_path}'.")
    #      except Exception as e:
    #          print(f"Could not clean up directory '{extract_path}': {e}")


    if not prompts_data:
        print("No valid prompts loaded. Exiting.")
        return

    # --- API Call Section ---
    raw_api_predictions_str = [] # Store raw string responses from API ('0', '1', or None/Error)
    print("\nCalling DeepSeek API for each prompt...")

    # Define character limit for the URL+HTML part to avoid context length errors
    # DeepSeek context limit is ~65k tokens. Assuming ~3-4 chars/token, aim lower.
    MAX_URL_HTML_CHARS = 150000  # Max characters for the url_html input part
    KEEP_CHARS = MAX_URL_HTML_CHARS // 2 # Keep half from start, half from end if truncating

    # Loop through the loaded data using enumerate to get index 'i'
    for i, item in enumerate(tqdm(prompts_data, desc="Processing prompts")):
        url_html = item["url_html"] # Get the full URL+HTML content
        original_length = len(url_html)
        truncated = False

        # --- Truncation Logic ---
        if original_length > MAX_URL_HTML_CHARS:
            print(f"\n[INFO] Prompt {i+1} content length ({original_length}) exceeds limit ({MAX_URL_HTML_CHARS}). Truncating.")
            start_chunk = url_html[:KEEP_CHARS]
            end_chunk = url_html[-KEEP_CHARS:]
            # Add a clear marker that content was cut
            url_html = f"{start_chunk}\n\n... [CONTENT TRUNCATED DUE TO CONTEXT LENGTH LIMIT] ...\n\n{end_chunk}"
            truncated = True
            print(f"[INFO] Prompt {i+1} new truncated length: {len(url_html)}")
        # --- End Truncation Logic ---

        # Create the formatted prompt using the potentially truncated url_html
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

        # Call the API function (which handles rate limiting, retries, API call itself)
        api_response_content = call_deepseek_api(formatted_prompt)

        # Store the raw response string (or None if error during API call)
        raw_api_predictions_str.append(api_response_content)
        # Rate limiting is handled within call_deepseek_api via acquire_token()

    # --- Post-processing and Evaluation Section ---
    print("\n--- Processing API Responses and Evaluating ---")

    valid_indices = []
    y_pred_mapped = [] # Store integer predictions (0 or 1) for valid responses
    y_pred_api_raw = [] # Store the raw API string ('0' or '1') for valid responses
    y_true_mapped = [] # Store integer ground truth for prompts with valid responses

    for i, raw_pred_str in enumerate(raw_api_predictions_str):
        # Get the corresponding ground truth label from the original loaded data
        gt_label = prompts_data[i]["label"] # Should be int (0 or 1)

        if raw_pred_str is not None and raw_pred_str in ['0', '1']:
             valid_indices.append(i) # Store index of the valid prompt
             y_pred_mapped.append(int(raw_pred_str)) # Store the prediction as integer
             y_pred_api_raw.append(raw_pred_str) # Store the valid raw string ('0' or '1')
             y_true_mapped.append(gt_label) # Store the corresponding ground truth
        else:
            # Log invalid/missing responses - extract URL part for context
            url_info = prompts_data[i]['url_html'].split('\n', 1)[0] # Get first line (likely URL)
            print(f"[WARN] Invalid or missing API response for prompt {i+1} (URL starting with: {url_info[:100]}...). Raw response: '{raw_pred_str}'. Excluding from metrics.")

    if len(valid_indices) == 0:
        print("\nNo valid predictions ('0' or '1') received from the API! Cannot calculate metrics.")
        # Still attempt to group results (will likely be empty), need valid_samples structure
        valid_samples_for_grouping = [] # Create empty list for grouping
        grouped = group_results(valid_samples_for_grouping, y_pred_api_raw) # Call with empty data
        write_grouped_results_to_csv(grouped)
        return

    print(f"\nEvaluated {len(valid_indices)} prompts with valid API responses ('0' or '1').")

    # Prepare data for grouping function - extract details only for valid responses
    valid_samples_for_grouping = []
    for i in valid_indices:
         # Extract URL part (first line) as identifier
         url_identifier = prompts_data[i]["url_html"].split('\n', 1)[0]
         valid_samples_for_grouping.append({
             "url": url_identifier, # Use URL part as identifier
             "label": prompts_data[i]["label"] # The ground truth label
         })

    # --- Calculate Metrics ---
    print("\n--- Evaluation Metrics (Positive Label = 1: Phishing) ---")
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

    # --- Confusion Matrix ---
    try:
        # Ensure labels=[0, 1] to handle cases where only one class might be present/predicted
        cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() # This works directly if cm is 2x2

        print("\nConfusion Matrix (Rows: Actual, Columns: Predicted)")
        print("                     Predicted Not Phishing (0)  Predicted Phishing (1)")
        print(f"Actual Not Phishing (0): {tn:<20}  {fp:<20}")
        print(f"Actual Phishing (1):     {fn:<20}  {tp:<20}")

        # Calculate Rates only if denominators are non-zero
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall(1)
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0 # Specificity / Recall(0)

        print(f"\nTrue Positive Rate (TPR/Recall-Phishing): {true_positive_rate:.4f}")
        print(f"False Positive Rate (FPR):                {false_positive_rate:.4f}")
        print(f"True Negative Rate (TNR/Specificity):     {true_negative_rate:.4f}")
        print(f"False Negative Rate (FNR):                {false_negative_rate:.4f}")

    except ValueError as e:
        # This might happen if only one class is present in y_true_mapped or y_pred_mapped
        print(f"\nCould not compute confusion matrix or related rates. Error: {e}")
        print(f"Unique True Labels: {sorted(list(set(y_true_mapped)))}")
        print(f"Unique Predicted Labels: {sorted(list(set(y_pred_mapped)))}")


    # --- Group Results and Write CSV ---
    # Pass the samples (URL identifier + label) and the raw API prediction string ('0' or '1')
    # for the valid predictions only
    grouped = group_results(valid_samples_for_grouping, y_pred_api_raw)
    write_grouped_results_to_csv(grouped)

    print("\nScript finished.")


# Make sure the other functions (call_deepseek_api, group_results, write_grouped_results_to_csv,
# refill_bucket, acquire_token) and necessary imports/constants are defined outside main().

# Standard Python entry point check
if __name__ == "__main__":
    main()