#!/usr/bin/env python3
# Simple CLI inference for the Telco Intent Router
# Usage: python infer.py "my internet is down"

import sys
import joblib

clf = joblib.load("model.pkl")

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "my internet is down"
pred = clf.predict([text])[0]
proba = clf.predict_proba([text])[0]
labels = clf.classes_
topk = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)[:3]

print("Text:", text)
print("Predicted intent:", pred)
print("Top-3:")
for label, p in topk:
    print(f"  - {label}: {p:.3f}")
