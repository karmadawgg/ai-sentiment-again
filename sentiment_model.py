"""
sentiment_model.py
Wraps a transformer sentiment-analysis model behind a small, stable interface
so the rest of the app (API, batch jobs, notebooks) never touches HuggingFace
internals directly.

Default model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Already fine-tuned for 3-class sentiment (negative / neutral / positive),
  which matches the Positive/Negative/Neutral requirement out of the box —
  no extra fine-tuning needed to get started.
- Swap MODEL_NAME for a domain fine-tuned BERT checkpoint later without
  changing any calling code.
"""
from __future__ import annotations  # lets "str | None" work on Python 3.9+

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# The Cardiff model's label order — kept explicit rather than trusting
# config.id2label blindly, since label order bugs are a classic silent
# failure mode in sentiment pipelines.
LABELS = ["negative", "neutral", "positive"]


@dataclass
class SentimentResult:
    label: str
    confidence: float
    scores: dict  # {"negative": .., "neutral": .., "positive": ..}


class SentimentClassifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> SentimentResult:
        return self.predict_batch([text])[0]

    @torch.inference_mode()
    def predict_batch(self, texts: Iterable[str], batch_size: int = 32) -> list[SentimentResult]:
        texts = list(texts)
        results: list[SentimentResult] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()

            for p in probs:
                scores = dict(zip(LABELS, p))
                label = max(scores, key=scores.get)
                results.append(SentimentResult(label=label, confidence=scores[label], scores=scores))

        return results


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    """Process-wide singleton so the model loads once, not per-request."""
    return SentimentClassifier()


if __name__ == "__main__":
    clf = get_classifier()
    samples = [
        "The delivery was three days late and support never replied.",
        "Works fine, nothing special.",
        "Absolutely love the new update, so much faster now!",
    ]
    for s, r in zip(samples, clf.predict_batch(samples)):
        print(f"{r.label:8s} ({r.confidence:.2f})  {s}")
