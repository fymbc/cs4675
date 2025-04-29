import math
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "new_data_urls.csv"


df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "harisudhan411/phishing-and-legitimate-urls",
    file_path,
)

# Total number of URLs loaded
print("Total URLs in dataset:", len(df))


def estimate_tokens(text):
    """
    Estimate the number of tokens in the input text.
    According to the Deepseek docs: approximately 1 English character ≈ 0.3 token.
    We use math.ceil to round up, ensuring that even very short strings count as at least one token.
    """
    return max(math.ceil(len(text) * 0.3), 1)


# Calc tokens for each URL.
df['tokens'] = df['url'].apply(estimate_tokens)

# Deepseek pricing
cost_per_token = 0.135 / 1_000_000

df['cost_usd'] = df['tokens'] * cost_per_token

# Calc totals.
total_tokens = df['tokens'].sum()
total_cost = df['cost_usd'].sum()

# Print results.
print("\nTotal tokens required to process all URLs:", total_tokens)
print("Total estimated cost to process all URLs with Deepseek (cache hit pricing): ${:.8f}".format(total_cost))
