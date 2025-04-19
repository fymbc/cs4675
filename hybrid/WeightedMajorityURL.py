import math
import pandas as pd
import csv
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
from kagglehub import KaggleDatasetAdapter
import json
import argparse
from datasets import load_dataset
import zipfile  # Added
import os       # Added


# --- LLM API Client Imports (fill in as needed) ---
import anthropic
import openai
import google.generativeai as genai
import requests
# llamastuff

# --- API Keys and Model Settings (fill in your own keys) ---
ANTHROPIC_API_KEY = ""
DEEPSEEK_API_KEY = ""
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
#llamastuff

# --- Model Names/Endpoints (anything you need to add for model access basically) ---
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

GEMINI_MODEL = "gemini-1.5-pro-latest"
genai.configure(api_key=GEMINI_API_KEY)
gemini_client_model = genai.GenerativeModel(GEMINI_MODEL) # Renamed variable to avoid conflict

OPENAI_MODEL = "gpt-4o"

#llamastuff

# --- URL+HTML Data Settings (Added) ---
URL_HTML_ZIP_PATH = "output_100.zip" # Path to your zip file
URL_HTML_TXT_FILENAME = "output_100.txt" # Name of the text file inside the zip
URL_HTML_EXTRACT_PATH = "./unzipped_html_data" # Temporary extraction folder

# --- Truncation Settings for URL+HTML (Added) ---
# Set a sensible limit based on typical LLM context windows (e.g., ~150k chars might be ~30-40k tokens)
# Adjust based on model limits and testing
MAX_URL_HTML_CHARS = 30000
KEEP_CHARS = MAX_URL_HTML_CHARS // 2 # Keep half from start, half from end if truncating


#weights config + decison threshold
MODEL_WEIGHTS = {
    'anthropic': 0.2451,
    'deepseek': 0.2486,
    'gemini': 0.2526,
    'openai': 0.2536
}
DECISION_THRESHOLD = 0.0


# --- Prompt Template (Modified) ---
def build_single_prompt(item, task):
    """
    Build the prompt for phishing (URL), truthfulness, or URL+HTML phishing.

    Args:
        item (dict):
            - for task="phish": {'url': str, 'label': int}
            - for task="truth": {'claim': str, 'true_label': int}
            - for task="url_html": {'url_html': str, 'label': int}
        task (str): "phish", "truth", or "url_html"

    Returns:
        str: a single‐string prompt for the LLM.
    """
    if task == "phish":
        return (
            f"Given this URL: {item['url']}, determine if it is a phishing website or not. "
            "ONLY OUTPUT 1 (PHISHING) OR 0 (NOT PHISHING). DO NOT WRITE ANYTHING ELSE."
        )
    elif task == "truth":
        return (
            f"Given the following statement:\n\n\"{item['claim']}\"\n\n"
            "Determine if it is TRUE or FALSE. "
            "ONLY OUTPUT 1 (TRUE) OR 0 (FALSE). DO NOT WRITE ANYTHING ELSE."
        )
    elif task == "url_html":
        url_html_content = item['url_html']
        original_length = len(url_html_content)
        truncated = False

        # --- Truncation Logic (Integrated from reference) ---
        if original_length > MAX_URL_HTML_CHARS:
            print(f"\n[INFO] Prompt content length ({original_length}) exceeds limit ({MAX_URL_HTML_CHARS}). Truncating.")
            start_chunk = url_html_content[:KEEP_CHARS]
            end_chunk = url_html_content[-KEEP_CHARS:]
            # Add a clear marker that content was cut
            url_html_content = f"{start_chunk}\n\n... [CONTENT TRUNCATED DUE TO CONTEXT LENGTH LIMIT] ...\n\n{end_chunk}"
            truncated = True
            # print(f"[INFO] New truncated length: {len(url_html_content)}") # Optional: uncomment for debug
        # --- End Truncation Logic ---

        return (
            f"Given the following input (URL and potentially partial HTML content), "
            # Add note about potential truncation if it happened
            f"{'[Note: HTML content may be truncated due to length limits] ' if truncated else ''}"
            f"determine if it represents a phishing website or not:\n\n"
            f"--- Input Start ---\n"
            f"{url_html_content}\n" # Use the potentially truncated version
            f"--- Input End ---\n\n"
            "Is this phishing? Respond ONLY with the integer 1 (phishing) or 0 (not phishing)."
            " Do not provide any explanation or other text."
        )
    else:
        raise ValueError(f"Unexpected task: {task!r}. Must be 'phish', 'truth', or 'url_html'.")

# --- Model Call Functions (Unchanged, but ensure they handle potentially long url_html prompts if truncation fails) ---
def call_anthropic(prompt):
    """Call Anthropic API with error handling and response normalization"""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=10, # Keep max_tokens low as we only expect 0/1
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract content from different response formats
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].text if hasattr(content[0], "text") else str(content[0])
        elif hasattr(content, "text"):
            content = content.text

        stripped_content = str(content).strip()
        return stripped_content if stripped_content in ("0", "1") else "-1" # Normalize here
    except Exception as e:
        print(f"Anthropic API Error: {e}")
        return "-1"  # Flag for error state

def map_anthropic_prediction(pred):
    """Map Anthropic's prediction to -1 (not phishing/false) or 1 (phishing/true)"""
    # Pred is already normalized to '0', '1', or '-1' by call_anthropic
    try:
        # Check if pred is '-1' first (error case)
        if pred == "-1":
            return 0 # Map errors/invalid to 0 for weighted sum exclusion
        # Otherwise, map '1' to 1 and '0' to -1
        return 1 if int(pred) == 1 else -1
    except ValueError:
        # This shouldn't happen if normalization in call_anthropic works
        print(f"Unexpected value in map_anthropic_prediction: {pred}")
        return 0

def call_deepseek(prompt):
    """Call DeepSeek API with comprehensive error handling and response normalization"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10, # Keep max_tokens low
        "temperature": 0.1 # Low temp for deterministic output
    }

    try:
        # Consider adding rate limiting like in the reference script if needed
        # acquire_token() # Would need the token bucket functions added
        response = requests.post(DEEPSEEK_API_ENDPOINT, json=payload, headers=headers, timeout=60)
        response.raise_for_status() # Check for HTTP errors

        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            print("Empty choices in DeepSeek response")
            return "-1"

        choice = response_data['choices'][0]
        if 'message' not in choice or 'content' not in choice['message']:
            print("Malformed DeepSeek response structure")
            return "-1"

        content = choice['message']['content'].strip()
        return content if content in ("0", "1") else "-1" # Normalize output

    except requests.exceptions.Timeout:
        print("DeepSeek API timeout")
        return "-1"
    except requests.exceptions.HTTPError as e:
        # Basic rate limit check
        if e.response is not None and e.response.status_code == 429:
             print("DeepSeek rate limit hit (429) - consider adding backoff/rate limiting logic")
        else:
            print(f"DeepSeek HTTP error: {e}")
            # Print response body if available for debugging
            if e.response is not None:
                 print(f"Response Body: {e.response.text}")
        return "-1"
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"DeepSeek response parsing error: {e}")
        return "-1"
    except Exception as e:
        print(f"An unexpected error occurred during DeepSeek call: {e}")
        return "-1"


def map_deepseek_prediction(pred):
    """Map DeepSeek's prediction to -1/1"""
    # Similar logic to anthropic mapping
    try:
        if pred == "-1":
            return 0
        return 1 if int(pred) == 1 else -1
    except ValueError:
        print(f"Unexpected value in map_deepseek_prediction: {pred}")
        return 0

def call_gemini(prompt, max_output_tokens=10):
    """Call Gemini API with output token limit and error handling"""
    try:
        # Use the configured model client
        response = gemini_client_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_output_tokens}
        )
        # Check response parts more carefully
        if response and response.parts:
             # Find the first part with text
             text_part = next((part.text for part in response.parts if hasattr(part, 'text')), None)
             if text_part:
                 stripped = text_part.strip()
                 # print(f"Gemini raw response: '{stripped}'") # Debugging
                 return stripped if stripped in ("0", "1") else "-1" # Normalize
        # Fallback/Error cases
        # print(f"Gemini response missing text part or empty: {response}") # Debugging
        return "-1"
    # Catch specific Gemini exceptions if known, otherwise general Exception
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Attempt to get more details if available in the exception object
        # error_details = getattr(e, 'message', str(e))
        # print(f"Gemini Error Details: {error_details}")
        return "-1"


def map_gemini_prediction(pred):
    """Map Gemini's prediction to -1/1"""
    # Similar logic to anthropic mapping
    try:
        if pred == "-1":
            return 0
        return 1 if int(pred) == 1 else -1
    except ValueError:
        print(f"Unexpected value in map_gemini_prediction: {pred}")
        return 0

def call_openai(prompt):
    """Call OpenAI API using provided prompt"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                # Keep system prompt simple if main instructions are in user prompt
                {"role": "system", "content": "Output only 0 or 1."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10, # Keep low
            temperature=0.1 # Keep low
        )
        content = response.choices[0].message.content.strip()
        return content if content in ("0", "1") else "-1" # Normalize
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "-1"

def map_openai_prediction(pred):
    """Map OpenAI's prediction to -1/1"""
    # Similar logic to anthropic mapping
    try:
        if pred == "-1":
            return 0
        return 1 if int(pred) == 1 else -1
    except ValueError:
        print(f"Unexpected value in map_openai_prediction: {pred}")
        return 0


# --- Load Dataset Functions (Added url_html loader) ---
def load_url_dataset(limit):
    """
    Load and shuffle the Kaggle URL dataset, returning up to `limit` samples.
    Each sample is {'url': str, 'label': int}.
    """
    # --- This function remains unchanged ---
    try:
        # Attempt to load using kagglehub first
        print("Attempting to load URL dataset via Kaggle Hub...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "harisudhan411/phishing-and-legitimate-urls",
            "new_data_urls.csv"
        )
        print("Loaded URL dataset via Kaggle Hub.")
    except Exception as e_kg:
        print(f"Kaggle Hub load failed ({e_kg}). Trying local 'new_data_urls.csv'...")
        try:
            df = pd.read_csv("new_data_urls.csv")
            print("Loaded URL dataset from local CSV.")
        except FileNotFoundError:
             print("Error: Neither Kaggle Hub nor local 'new_data_urls.csv' found for URL dataset.")
             return [] # Return empty list if data source fails
        except Exception as e_csv:
             print(f"Error reading local CSV 'new_data_urls.csv': {e_csv}")
             return []

    if "url" not in df.columns or "status" not in df.columns:
        print("Error: URL Dataset must have 'url' and 'status' columns")
        return []

    # Map 'status' (which might be string '0'/'1') to integer, handle errors
    try:
        df['status'] = df['status'].astype(int)
    except ValueError:
        print("Error: Could not convert 'status' column to integer in URL dataset.")
        # Optionally filter out rows where conversion fails
        # df = df[pd.to_numeric(df['status'], errors='coerce').notna()]
        # df['status'] = df['status'].astype(int)
        return []


    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    samples = [
        {"url": row["url"], "label": int(row["status"])}
        for _, row in df.iterrows()
        # Add stricter checks for valid data before adding
        if isinstance(row["url"], str) and row["url"].strip() != "" and row["status"] in (0, 1)
    ]
    print(f"Prepared {len(samples)} valid URL samples.")
    return samples[:limit]


def load_fever_dataset(limit):
    """
    Load and preprocess the FEVER 'labelled_dev' split for truthfulness evaluation.
    Returns: List[Dict]: Each dict has keys 'claim' and 'true_label' (1 for SUPPORTS/True, 0 for REFUTES/False).
    """
    # --- This function remains largely unchanged ---
    try:
        print("Loading FEVER dataset via Hugging Face datasets...")
        ds = load_dataset(
            "fever",
            "v1.0",
            split="labelled_dev",
            trust_remote_code=True # Keep trust_remote_code if required by dataset script
        )
        df = pd.DataFrame(ds)
        print("FEVER dataset loaded.")
    except Exception as e:
        print(f"Error loading FEVER dataset: {e}")
        return [] # Return empty on error

    # Filter and map labels
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])]
    if df.empty:
        print("No 'SUPPORTS' or 'REFUTES' labels found in FEVER dev split.")
        return []

    df["true_label"] = df["label"].map({"SUPPORTS": 1, "REFUTES": 0})

    # Shuffle and select columns
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    truth_samples = df[["claim", "true_label"]].to_dict(orient="records")

    # Filter out any potential rows with missing claims
    truth_samples = [s for s in truth_samples if isinstance(s.get("claim"), str) and s["claim"].strip()]

    print(f"Prepared {len(truth_samples)} valid FEVER samples.")
    return truth_samples[:limit]

# --- Added URL+HTML Dataset Loader ---
def load_url_html_dataset(limit, zip_path=URL_HTML_ZIP_PATH, txt_filename=URL_HTML_TXT_FILENAME, extract_path=URL_HTML_EXTRACT_PATH):
    """
    Loads URL+HTML data from a specified zip file containing a text file.
    Expected text file format per line: <url_and_html_content>,<label (0 or 1)>
    Returns: List[Dict] like {'url_html': str, 'label': int}
    """
    print(f"Attempting to load URL+HTML data from '{zip_path}'...")
    samples = []

    try:
        # Ensure extraction path exists
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
            print(f"Created extraction directory: '{extract_path}'")

        # Extract the specific file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check if file exists in zip before extraction
            if txt_filename not in zip_ref.namelist():
                 print(f"Error: File '{txt_filename}' not found inside '{zip_path}'.")
                 # Clean up extraction directory if empty/created
                 try:
                     if not os.listdir(extract_path): # Check if empty
                         os.rmdir(extract_path)
                 except OSError: pass # Ignore errors during cleanup
                 return []

            zip_ref.extract(txt_filename, extract_path)
            print(f"Extracted '{txt_filename}' to '{extract_path}'.")

        filepath = os.path.join(extract_path, txt_filename)

        # Read the extracted file
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        processed_count = 0
        skipped_count = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                skipped_count += 1
                continue

            # Use rsplit(',', 1) to split only on the last comma
            parts = line.rsplit(",", 1)
            if len(parts) == 2:
                url_html_content = parts[0].strip()
                ground_truth_str = parts[1].strip()

                if ground_truth_str in ["0", "1"] and url_html_content: # Check content not empty
                    try:
                        ground_truth_int = int(ground_truth_str)
                        samples.append({"url_html": url_html_content, "label": ground_truth_int})
                        processed_count += 1
                    except ValueError: # Should not happen with 'in ["0", "1"]' check, but safety first
                         print(f"Skipping line {i+1} due to unexpected non-integer ground truth: '{ground_truth_str}'")
                         skipped_count += 1
                else:
                    # More specific skip reasons
                    if not url_html_content:
                         print(f"Skipping line {i+1} due to empty URL/HTML content.")
                    else:
                         print(f"Skipping line {i+1} due to invalid ground truth value: '{ground_truth_str}' (Expected '0' or '1')")
                    skipped_count += 1
            else:
                print(f"Skipping malformed line {i+1}: Could not split into content and label using the last comma. Line start: '{line[:100]}...'")
                skipped_count += 1

        print(f"Successfully loaded {processed_count} URL+HTML samples.")
        if skipped_count > 0:
             print(f"Skipped {skipped_count} lines from '{txt_filename}' due to formatting or invalid labels.")

    except FileNotFoundError:
        print(f"Error: Zip file '{zip_path}' not found.")
        return []
    except zipfile.BadZipFile:
        print(f"Error: '{zip_path}' is not a valid zip file or is corrupted.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during URL+HTML data loading: {e}")
        return []
    # Optional: Clean up extracted file/directory
    # finally:
    #      try:
    #          filepath = os.path.join(extract_path, txt_filename)
    #          if os.path.exists(filepath):
    #              os.remove(filepath)
    #          if os.path.exists(extract_path) and not os.listdir(extract_path): # Remove dir if empty
    #              os.rmdir(extract_path)
    #          # print(f"Cleaned up extracted file/directory '{extract_path}'.") # Optional msg
    #      except Exception as e_clean:
    #          print(f"Warning: Could not clean up extracted file/directory '{extract_path}': {e_clean}")

    # Shuffle the loaded samples (optional, but good practice)
    import random
    random.seed(42)
    random.shuffle(samples)

    return samples[:limit]


# --- Weighted Decision (Unchanged, but relies on map_* functions returning -1/1 or 0) ---
def weighted_decision(predictions):
    """
    Calculate weighted sum of predictions.
    Args:
        predictions: list of tuples (model_name, mapped_prediction [-1, 1, or 0 for invalid])
    Returns:
        1 (phishing/true) if sum >= threshold, 0 (not phishing/false) otherwise.
        Returns -1 if no valid predictions were made.
    """
    total = 0.0
    valid_models_count = 0 # Count models that gave a valid (-1 or 1) prediction

    for model_name, pred_vote in predictions:
        # Check if the prediction is valid (-1 or 1)
        if pred_vote in (-1, 1):
            # Use the weight for this model
            weight = MODEL_WEIGHTS.get(model_name, 0) # Default to 0 weight if model unknown
            if weight > 0:
                total += weight * pred_vote
                valid_models_count += 1
        # else: pred_vote is 0 (invalid/error), so we just ignore it

    # If no models provided a valid prediction, return an error indicator
    if valid_models_count == 0:
        # print("Warning: No valid predictions from any model for this item.") # Optional debug msg
        return -1  # Indicate no valid decision could be made

    # Make the final decision based on the threshold
    # Note: If total is exactly 0 (e.g., one model says -1, another says 1, with equal weights),
    # this threshold determines the outcome. DECISION_THRESHOLD=0.0 means 0 maps to 0 (non-phishing/false).
    return 1 if total >= DECISION_THRESHOLD else 0


# --- Main Script (Modified) ---
def main():
    # --- Parse command‑line arguments ---
    parser = argparse.ArgumentParser(description="Run LLM ensemble for phishing or truthfulness.")
    parser.add_argument(
        "--task",
        choices=["phish", "truth", "url_html", "all"], # Added url_html and all
        default="all", # Changed default to run all tasks
        help="Which evaluation(s) to run: 'phish' (URL only), 'truth' (FEVER claims), 'url_html' (URL+HTML phishing), or 'all'."
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=100, # Reduced default for quicker testing
        help="Max samples per task type."
    )
    args = parser.parse_args()

    # --- Prepare task runners based on selection ---
    tasks_to_run = []
    if args.task == "phish" or args.task == "all":
        tasks_to_run.append("phish")
    if args.task == "truth" or args.task == "all":
        tasks_to_run.append("truth")
    if args.task == "url_html" or args.task == "all":
        tasks_to_run.append("url_html")

    # --- Load data for selected tasks ---
    loaded_data = {}
    if "phish" in tasks_to_run:
        print(f"\nLoading phishing URLs (limit: {args.sample_limit})...")
        loaded_data["phish"] = load_url_dataset(args.sample_limit)
    if "truth" in tasks_to_run:
        print(f"\nLoading FEVER truth statements (limit: {args.sample_limit})...")
        loaded_data["truth"] = load_fever_dataset(args.sample_limit)
    if "url_html" in tasks_to_run:
        print(f"\nLoading URL+HTML phishing data (limit: {args.sample_limit})...")
        loaded_data["url_html"] = load_url_html_dataset(args.sample_limit) # Using new loader

    # --- Define models to use ---
    # Could make this configurable via args too
    models_to_call = ["anthropic", "deepseek", "gemini", "openai"]


    # --- Execute each selected task ---
    for task in tasks_to_run:
        samples = loaded_data.get(task)
        if not samples:
            print(f"\n--- Skipping task '{task}': No data loaded or available. ---")
            continue # Skip to next task if data loading failed

        print(f"\n=== Running task '{task}' on {len(samples)} items using models: {models_to_call} ===")
        results = []
        start_time = time.time()

        for item_index, item in enumerate(tqdm(samples, desc=f"Ensembling for '{task}'")):
            # 1) Build prompt for the current task
            #    Handles truncation internally for 'url_html'
            prompt = build_single_prompt(item, task)

            # 2) Call each specified model and get raw strings ('0', '1', or '-1' for errors)
            raw_preds = {}
            for model_name in models_to_call:
                 call_function = globals().get(f"call_{model_name}")
                 if call_function:
                     raw_preds[model_name] = call_function(prompt)
                 else:
                     print(f"Warning: Call function for model '{model_name}' not found.")
                     raw_preds[model_name] = "-1" # Treat as error if function missing

            # 3) Map raw outputs to +1 / -1 votes (or 0 for errors/invalid)
            mapped_votes = {}
            for model_name in models_to_call:
                 map_function = globals().get(f"map_{model_name}_prediction")
                 raw_pred = raw_preds.get(model_name, "-1") # Default to error if prediction missing
                 if map_function:
                     mapped_votes[model_name] = map_function(raw_pred)
                 else:
                     print(f"Warning: Mapping function for model '{model_name}' not found.")
                     mapped_votes[model_name] = 0 # Treat as invalid if function missing

            # 4) Compute weighted ensemble decision (0, 1, or -1 for no valid votes)
            final_vote = weighted_decision(list(mapped_votes.items()))

            # 5) Collect results in a row
            #    Use appropriate key for input data ('url', 'claim', or 'url_html')
            input_data_key = "url" if task == "phish" else "claim" if task == "truth" else "url_html"
            #    Use appropriate key for ground truth ('label' or 'true_label')
            ground_truth_key = "true_label" if task == "truth" else "label"

            row = {
                # Include original input identifier and ground truth
                input_data_key: item[input_data_key],
                ground_truth_key: item[ground_truth_key],
                # Add raw predictions from each model
                **{f"{m}_raw": raw_preds.get(m, "N/A") for m in models_to_call},
                # Add mapped votes from each model
                **{f"{m}_vote": mapped_votes.get(m, "N/A") for m in models_to_call},
                # Add the final ensemble decision
                f"{task}_ensemble": final_vote
            }
            results.append(row)

        # --- End of loop for items ---
        end_time = time.time()
        print(f"Task '{task}' completed in {end_time - start_time:.2f} seconds.")


        # --- Save per-item results to CSV ---
        results_df = pd.DataFrame(results)
        results_csv = f"{task}_ensemble_results_{len(samples)}items.csv"
        try:
            results_df.to_csv(results_csv, index=False, quoting=csv.QUOTE_ALL) # Ensure proper quoting for complex fields like url_html
            print(f"Saved detailed results for task '{task}' to {results_csv}")
        except Exception as e_csv_save:
            print(f"Error saving results CSV for task '{task}': {e_csv_save}")


        # --- Compute & save overall metrics ---
        print(f"\nCalculating metrics for task '{task}'...")
        ground_truth_key = "true_label" if task == "truth" else "label"
        ensemble_key = f"{task}_ensemble"

        # Filter results where the ensemble made a valid prediction (0 or 1)
        valid_results = [r for r in results if r[ensemble_key] in (0, 1)]

        if not valid_results:
             print(f"No valid ensemble predictions (0 or 1) for task '{task}'. Cannot calculate metrics.")
             num_invalid = len([r for r in results if r[ensemble_key] == -1])
             if num_invalid > 0:
                 print(f"({num_invalid} items had no valid consensus from models)")
             continue # Skip metrics calculation for this task

        y_true = [r[ground_truth_key] for r in valid_results]
        y_pred = [r[ensemble_key] for r in valid_results]

        # Check if we have both classes present in y_true for robust metrics
        # (otherwise precision/recall for the missing class is ill-defined)
        labels_present = set(y_true)
        print(f"Labels present in ground truth for metric calculation: {labels_present}")
        print(f"Predictions made by ensemble: {set(y_pred)}")


        # Calculate standard metrics
        acc    = accuracy_score(y_true, y_pred)
        # Use labels=[0, 1] to handle cases where only one class might be predicted
        prec1  = precision_score(y_true, y_pred, pos_label=1, zero_division=0, labels=[0, 1])
        rec1   = recall_score(y_true, y_pred, pos_label=1, zero_division=0, labels=[0, 1])
        f11    = f1_score(y_true, y_pred, pos_label=1, zero_division=0, labels=[0, 1])
        prec0  = precision_score(y_true, y_pred, pos_label=0, zero_division=0, labels=[0, 1])
        rec0   = recall_score(y_true, y_pred, pos_label=0, zero_division=0, labels=[0, 1])
        f10    = f1_score(y_true, y_pred, pos_label=0, zero_division=0, labels=[0, 1])

        # Calculate confusion matrix, ensure labels are specified
        try:
            # Get counts for TN, FP, FN, TP
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            # Handle case where cm might not be 2x2 if only one class predicted/true
            if cm.shape == (2, 2):
                 tn, fp, fn, tp = cm.ravel()
            elif len(labels_present) == 1 or len(set(y_pred)) == 1:
                 print("Warning: Only one class present in ground truth or predictions. Confusion matrix might be incomplete.")
                 # Attempt to extract based on which class is present (e.g., if only class 1 is true/pred)
                 if 1 in labels_present or 1 in set(y_pred): # Check if class 1 exists
                     tp = cm[0,0] if 1 in labels_present and 1 in set(y_pred) else 0 # TP needs both true=1 and pred=1
                     fn = cm[0,0] if 1 in labels_present and 0 in set(y_pred) else 0 # FN needs true=1 and pred=0 - requires both classes in prediction space
                     fp = cm[0,0] if 0 in labels_present and 1 in set(y_pred) else 0 # FP needs true=0 and pred=1 - requires both classes in truth space
                     tn = cm[0,0] if 0 in labels_present and 0 in set(y_pred) else 0 # TN needs true=0 and pred=0
                     # This logic is simplified and might need adjustment based on specific cases
                 else: # Only class 0 present
                     tn = cm[0,0]
                     tp, fn, fp = 0, 0, 0
            else: # Should not happen if labels=[0,1] used, but as fallback
                 print("Error: Unexpected confusion matrix shape.")
                 tn, fp, fn, tp = 0, 0, 0, 0
        except Exception as e_cm:
            print(f"Could not compute confusion matrix components: {e_cm}")
            tn, fp, fn, tp = 'N/A', 'N/A', 'N/A', 'N/A'


        metrics = {
            "Task": task,
            "Num_Valid_Samples": len(valid_results),
            "Num_Total_Samples": len(samples),
            "Accuracy": f"{acc:.4f}",
            "Prec(1)": f"{prec1:.4f}", "Rec(1)": f"{rec1:.4f}", "F1(1)": f"{f11:.4f}",
            "Prec(0)": f"{prec0:.4f}", "Rec(0)": f"{rec0:.4f}", "F1(0)": f"{f10:.4f}",
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        }
        # Print metrics to console
        print("\n--- Metrics ---")
        for key, value in metrics.items():
             print(f"{key}: {value}")
        print("---------------")

        # Save metrics to CSV
        # Create a list of dictionaries for easier DataFrame creation
        metrics_list = [{"Metric": key, "Value": value} for key, value in metrics.items()]
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv = f"{task}_ensemble_metrics_{len(samples)}items.csv"
        try:
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"Saved metrics for task '{task}' to {metrics_csv}")
        except Exception as e_metrics_save:
            print(f"Error saving metrics CSV for task '{task}': {e_metrics_save}")

    print("\n=== All selected tasks finished ===")


if __name__ == "__main__":
    # Ensure API keys are set (you might want to load from env variables)
    if not all([ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY]):
         print("Warning: One or more API keys are not set in the script.")
         # Optionally exit or prompt user
         # exit("Please set API keys before running.")

    main()
