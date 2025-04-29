# Real-Time Phishing Detection Using Large Language Models with Extended Truthfulness Evaluation

This project implements a **hybrid ensemble** of multiple large language models (LLMs) to perform:

- **Truthfulness Detection** (fact-checking claims)
- **Phishing Detection** via:
  - URL-only analysis
  - URL + HTML page content analysis

A **Google Chrome Extension** frontend connects to a **FastAPI backend** that calls and aggregates the results of 5 different LLMs to improve detection reliability.

### 1. Truthfulness Detection API

- **Endpoint:** `/analyze`
- **Input:** Text claim
- **Output:** Classify claim as `"TRUE"` or `"FALSE"`
- **Models Used:** Anthropic, DeepSeek, Gemini, GPT-4o

### 2. URL-Only Phishing Detection API

- **Endpoint:** `/analyze-url`
- **Input:** Single URL
- **Output:** Classify as `"PHISHING"` or `"LEGITIMATE"`
- **Models Used:** Anthropic, DeepSeek, Gemini, GPT-4o

### 3. URL + HTML Phishing Detection API

- **Endpoint:** `/analyze-url-html`
- **Input:** URL + partial HTML snippet
- **Output:** Classify as `"PHISHING"` or `"LEGITIMATE"`
- **Models Used:** Anthropic, DeepSeek, Gemini, GPT-4o


