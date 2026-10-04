"""
seed_data.py
Populates the local database with realistic sample feedback (so /trends and
/issues return something on a fresh install) and writes dashboard_sample.json
— a static snapshot the standalone dashboard.html preview reads, since a
browser-only demo can't call your local API directly.

Run:
    python seed_data.py
"""

import json
import random
import datetime as dt

from database import init_db, SessionLocal, Feedback
from preprocessing import clean_text, extract_keywords
from sentiment_model import get_classifier
from collections import Counter

SAMPLE_FEEDBACK = [
    ("The app crashed twice during checkout, lost my cart both times.", "app_review"),
    ("Support resolved my billing issue in under ten minutes, great work.", "support_ticket"),
    ("Shipping took 9 days when it promised 3. Pretty frustrating.", "survey"),
    ("It's fine. Does what it says, nothing more.", "app_review"),
    ("Love the new dashboard redesign, so much easier to navigate!", "app_review"),
    ("Customer service was rude and unhelpful when I called.", "support_ticket"),
    ("Product quality is good but the price went up again.", "survey"),
    ("Setup was confusing and the docs are outdated.", "support_ticket"),
    ("Best purchase I've made this year, exceeded expectations.", "app_review"),
    ("The refund process took way too long, still waiting.", "support_ticket"),
    ("Average experience overall, met basic expectations.", "survey"),
    ("Delivery driver was great but packaging was damaged.", "survey"),
    ("Constant login errors this week, very annoying.", "app_review"),
    ("Great value for the price, would recommend to friends.", "app_review"),
    ("Nothing has changed since the last update.", "survey"),
]


def main():
    init_db()
    clf = get_classifier()
    db = SessionLocal()

    now = dt.datetime.utcnow()
    for i in range(120):  # replay the sample pool across the last 30 days
        text, source = random.choice(SAMPLE_FEEDBACK)
        cleaned = clean_text(text)
        result = clf.predict(cleaned)
        created = now - dt.timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))

        db.add(Feedback(
            raw_text=text,
            cleaned_text=cleaned,
            source=source,
            sentiment=result.label,
            confidence=result.confidence,
            score_negative=result.scores["negative"],
            score_neutral=result.scores["neutral"],
            score_positive=result.scores["positive"],
            created_at=created,
        ))

    db.commit()

    # ---- export a static snapshot for the standalone dashboard preview ----
    rows = db.query(Feedback).all()
    by_day = {}
    for r in rows:
        day = r.created_at.date().isoformat()
        by_day.setdefault(day, {"positive": 0, "neutral": 0, "negative": 0})
        by_day[day][r.sentiment] += 1

    negatives = [r.cleaned_text for r in rows if r.sentiment == "negative"]
    kw_counter = Counter()
    for t in negatives:
        kw_counter.update(set(extract_keywords(t)))

    snapshot = {
        "generated_at": now.isoformat(),
        "summary": {
            "total": len(rows),
            "positive": sum(1 for r in rows if r.sentiment == "positive"),
            "neutral": sum(1 for r in rows if r.sentiment == "neutral"),
            "negative": sum(1 for r in rows if r.sentiment == "negative"),
        },
        "daily": [{"date": d, **c} for d, c in sorted(by_day.items())],
        "issues": [{"keyword": k, "mentions": c} for k, c in kw_counter.most_common(10)],
        "recent": [
            {"text": r.raw_text, "sentiment": r.sentiment, "confidence": round(r.confidence, 2)}
            for r in sorted(rows, key=lambda r: r.created_at, reverse=True)[:8]
        ],
    }

    with open("dashboard_sample.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Seeded {len(rows)} feedback rows and wrote dashboard_sample.json")


if __name__ == "__main__":
    main()
