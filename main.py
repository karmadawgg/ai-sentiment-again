"""
main.py
FastAPI service tying preprocessing -> model -> database together, plus the
aggregation endpoints the dashboard reads from.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /feedback           classify + store one piece of feedback (real-time inference)
    POST /feedback/batch     classify + store many at once (e.g. CSV import)
    GET  /feedback           list/filter stored feedback
    GET  /trends/daily       {date, positive, neutral, negative} series for charts
    GET  /trends/summary     overall counts + average confidence
    GET  /issues             top keywords driving negative feedback
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from preprocessing import clean_text, extract_keywords
from sentiment_model import get_classifier
from database import init_db, get_db, Feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_classifier()  # warm the model once at boot, not on first request
    yield


app = FastAPI(title="Customer Feedback Sentiment API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your dashboard's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    """Health check — hosting platforms (Render/Railway) ping this to confirm the service is up."""
    return {"status": "ok", "service": "customer-feedback-sentiment-api"}


# ---------- schemas ----------

class FeedbackIn(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = "unknown"


class FeedbackBatchIn(BaseModel):
    items: list[FeedbackIn]


class FeedbackOut(BaseModel):
    id: int
    raw_text: str
    sentiment: str
    confidence: float
    source: str
    created_at: dt.datetime

    class Config:
        from_attributes = True


# ---------- core inference + storage ----------

def _classify_and_store(db: Session, text: str, source: str) -> Feedback:
    cleaned = clean_text(text)
    result = get_classifier().predict(cleaned)

    row = Feedback(
        raw_text=text,
        cleaned_text=cleaned,
        source=source,
        sentiment=result.label,
        confidence=result.confidence,
        score_negative=result.scores["negative"],
        score_neutral=result.scores["neutral"],
        score_positive=result.scores["positive"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@app.post("/feedback", response_model=FeedbackOut)
def submit_feedback(item: FeedbackIn, db: Session = Depends(get_db)):
    return _classify_and_store(db, item.text, item.source)


@app.post("/feedback/batch", response_model=list[FeedbackOut])
def submit_feedback_batch(payload: FeedbackBatchIn, db: Session = Depends(get_db)):
    return [_classify_and_store(db, i.text, i.source) for i in payload.items]


@app.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    sentiment: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(Feedback)
    if sentiment:
        q = q.filter(Feedback.sentiment == sentiment)
    if source:
        q = q.filter(Feedback.source == source)
    return q.order_by(Feedback.created_at.desc()).limit(limit).all()


# ---------- dashboard aggregation ----------

@app.get("/trends/daily")
def trends_daily(days: int = 30, db: Session = Depends(get_db)):
    since = dt.datetime.utcnow() - dt.timedelta(days=days)
    rows = db.query(Feedback).filter(Feedback.created_at >= since).all()

    by_day = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for r in rows:
        day = r.created_at.date().isoformat()
        by_day[day][r.sentiment] += 1

    return [{"date": day, **counts} for day, counts in sorted(by_day.items())]


@app.get("/trends/summary")
def trends_summary(db: Session = Depends(get_db)):
    total = db.query(func.count(Feedback.id)).scalar() or 0
    if total == 0:
        return {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "avg_confidence": 0}

    counts = dict(
        db.query(Feedback.sentiment, func.count(Feedback.id)).group_by(Feedback.sentiment).all()
    )
    avg_conf = db.query(func.avg(Feedback.confidence)).scalar() or 0

    return {
        "total": total,
        "positive": counts.get("positive", 0),
        "neutral": counts.get("neutral", 0),
        "negative": counts.get("negative", 0),
        "avg_confidence": round(avg_conf, 3),
    }


@app.get("/issues")
def top_issues(limit: int = 10, db: Session = Depends(get_db)):
    """Naive but effective: most frequent keywords across negative feedback,
    as a stand-in for topic modeling (swap in BERTopic/LDA later)."""
    negatives = db.query(Feedback.cleaned_text).filter(Feedback.sentiment == "negative").all()

    counter = Counter()
    for (text,) in negatives:
        counter.update(set(extract_keywords(text)))

    return [{"keyword": k, "mentions": c} for k, c in counter.most_common(limit)]
