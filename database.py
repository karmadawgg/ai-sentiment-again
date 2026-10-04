"""
database.py
Storage layer. Defaults to local SQLite so the project runs with zero setup;
point DATABASE_URL at Postgres/MySQL for production without changing any
other file.
"""

import os
import datetime as dt

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./feedback.db")

# Render/Heroku-style hosts hand out "postgres://", but SQLAlchemy 2.x + psycopg2
# require the "postgresql://" scheme — rewrite it so DATABASE_URL just works either way.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    source = Column(String(64), default="unknown")       # e.g. "app_review", "support_ticket", "survey"
    sentiment = Column(String(16), nullable=False)        # positive / negative / neutral
    confidence = Column(Float, nullable=False)
    score_negative = Column(Float, nullable=False)
    score_neutral = Column(Float, nullable=False)
    score_positive = Column(Float, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
