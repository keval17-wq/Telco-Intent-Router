# Telco Intent Router (Mini Training Project)

A compact, portfolio-ready project for an **AI Prompt Engineer (Telecom & Web)** role.

## What this shows

- You can **curate data**, **train** a classifier, and **evaluate** it.
- You understand how to **route customer messages** (text/voice transcripts) to the right workflow.
- This baseline can sit **in front of LLM prompts** to pick the right system prompt/tool (e.g., Twilio IVR, Dialogflow agent, or a Jambonz app).

## Quickstart

```bash
pip install -r requirements.txt
python infer.py "I want to cancel my plan"
```

Expected output:
```
Predicted intent: cancellation
Top-3:
  - cancellation: 0.91
  - billing_issue: 0.05
  - new_connection: 0.02
```

## Train / Evaluate (already trained in repo)

If you want to retrain:
```python
# See train_eval.ipynb or replicate steps from create_dataset_and_train.py
```

## Files
- `dataset.csv` — synthetic telco utterances with 7 intents
- `model.pkl` — trained TF‑IDF + Logistic Regression pipeline
- `infer.py` — quick CLI inference
- `confusion_matrix.png` — visual evaluation
- `requirements.txt` — versions used
- `README.md` — this file

## How this plugs into **Prompt Engineering**
- **Router → Prompt**: Use the predicted intent to select a **prompt template** (e.g., Billing SOP vs. Outage troubleshooting).
- **Voice**: Feed **ASR transcripts** (e.g., Twilio/Whisper) into this router, then choose a function call or knowledge base article.
- **Eval Harness**: Add a small YAML with canonical answers per intent and verify LLM output for **tone/accuracy**.
- **Privacy**: Redact PII (account numbers, addresses) before logging.

## Next steps (resume bullets)
- Add a **few-shot prompt library** per intent and an **evaluation script** measuring exactness, helpfulness, and tone on a seed set.
- Integrate with **Twilio** (webhook) or **Jambonz** and push transcripts through this router.
- Replace synthetic data with anonymised real transcripts; fine‑tune a small LLM for response style.
- Implement **guardrails**: profanity filter, policy do/don'ts, escalation rules (e.g., outage→status page).

## Results (this run)
- Accuracy: 0.976
- Macro F1: 0.976

See `confusion_matrix.png` for class-wise performance.
