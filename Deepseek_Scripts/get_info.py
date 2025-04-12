import math
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Use the Kaggle dataset filename as published (do not use a full local path)
file_path = "new_data_urls.csv"

# Load the dataset from Kaggle using KaggleHub.
# Ensure that your Kaggle API credentials are set up properly.
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "harisudhan411/phishing-and-legitimate-urls",
    file_path,
)

# Display total number of URLs loaded
print("Total URLs in dataset:", len(df))


def estimate_tokens(text):
    """
    Estimate the number of tokens in the input text.
    According to the Deepseek docs: approximately 1 English character ≈ 0.3 token.
    We use math.ceil to round up, ensuring that even very short strings count as at least one token.
    """
    return max(math.ceil(len(text) * 0.3), 1)


# Calculate tokens for each URL in the dataset.
df['tokens'] = df['url'].apply(estimate_tokens)

# Deepseek pricing assumption:
# For deepseek-chat with standard pricing (cache hit), the cost is $0.07 per 1,000,000 tokens.
cost_per_token = 0.135 / 1_000_000

# Compute the cost per URL based on its token count.
df['cost_usd'] = df['tokens'] * cost_per_token

# Calculate totals.
total_tokens = df['tokens'].sum()
total_cost = df['cost_usd'].sum()

# Print the aggregated results.
print("\nTotal tokens required to process all URLs:", total_tokens)
print("Total estimated cost to process all URLs with Deepseek (cache hit pricing): ${:.8f}".format(total_cost))
