# Real-Time Phishing Detection Using Large Language Models with Extended Truthfulness Evaluation

This project implements a **hybrid ensemble** of multiple large language models (LLMs) to perform:

- **Truthfulness Detection** (fact-checking claims)
- **Phishing Detection** via:
  - URL-only analysis
  - URL + HTML page content analysis

## Project Deliverables

An evaluation on the effectiveness of 5 LLMs and a weighted hybrid ensemble of the models for phishing and truthfulness detection. An additional real-time phishing and truthfulness detection **Google Chrome Extension** frontend which connects to a **FastAPI backend** that calls and aggregates the results of 5 different LLMs into a weighted ensemble model. 

### Backend APIs 

### Truthfulness Detection API

- **Endpoint:** `/analyze`
- **Input:** Text truthfulness claim
- **Output:** Classify claim as `"TRUE"` or `"FALSE"`

### URL-Only Phishing Detection API

- **Endpoint:** `/analyze-url`
- **Input:** Single URL
- **Output:** Classify as `"PHISHING"` or `"LEGITIMATE"`

### URL + HTML Phishing Detection API

- **Endpoint:** `/analyze-url-html`
- **Input:** URL + partial HTML snippet
- **Output:** Classify as `"PHISHING"` or `"LEGITIMATE"`

### 5 Large Language Models Evaluated
| **Anthropic Claude-3.5 Sonnet** |
| **DeepSeek Chat** |
| **Google Gemini 1.5 Pro** |
| **OpenAI GPT-4o-mini** |
| **Meta Llama-3 8B Chat** |
