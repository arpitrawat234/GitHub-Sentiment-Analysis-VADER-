"""
sentiment_engine.py
────────────────────
Cleans raw GitHub text and scores it with VADER.
"""

import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()

# ── Cleaning ──────────────────────────────────────────────────
_CODE_BLOCK  = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`]+`")
_URL         = re.compile(r"https?://\S+")
_MENTION     = re.compile(r"@\w+")
_ISSUE_REF   = re.compile(r"#\d+")
_HTML_TAG    = re.compile(r"<[^>]+>")
_WHITESPACE  = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove code blocks, URLs, mentions, HTML and extra whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _ISSUE_REF.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ── VADER Scoring ─────────────────────────────────────────────
def score_text(text: str) -> dict:
    """Return VADER scores for a single string."""
    cleaned = clean_text(text)
    if not cleaned:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0, "label": "neutral"}
    scores  = _sia.polarity_scores(cleaned)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "pos":      scores["pos"],
        "neg":      scores["neg"],
        "neu":      scores["neu"],
        "compound": compound,
        "label":    label,
    }


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned_text and VADER sentiment columns to a DataFrame.
    Input df must have a 'text' column.
    """
    if df.empty:
        return df

    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)

    scores = df["cleaned_text"].apply(score_text).apply(pd.Series)
    df = pd.concat([df, scores], axis=1)

    # Drop rows where cleaned text is empty
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    return df


# ── Summary helpers ───────────────────────────────────────────
def summary_stats(df: pd.DataFrame) -> dict:
    """Return high-level sentiment statistics."""
    total = len(df)
    if total == 0:
        return {}
    counts   = df["label"].value_counts().to_dict()
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral  = counts.get("neutral",  0)
    avg_compound = df["compound"].mean()
    return {
        "total":        total,
        "positive":     positive,
        "negative":     negative,
        "neutral":      neutral,
        "pct_positive": round(positive / total * 100, 1),
        "pct_negative": round(negative / total * 100, 1),
        "pct_neutral":  round(neutral  / total * 100, 1),
        "avg_compound": round(avg_compound, 4),
        "overall_mood": (
            "Positive 😊" if avg_compound >= 0.05
            else "Negative 😟" if avg_compound <= -0.05
            else "Neutral 😐"
        ),
    }


def top_items(df: pd.DataFrame, label: str, n: int = 5) -> pd.DataFrame:
    """Return top-n most extreme items for a given sentiment label."""
    sub = df[df["label"] == label].copy()
    if label == "positive":
        return sub.nlargest(n, "compound")[["type", "author", "cleaned_text", "compound", "date"]]
    elif label == "negative":
        return sub.nsmallest(n, "compound")[["type", "author", "cleaned_text", "compound", "date"]]
    else:
        return sub.head(n)[["type", "author", "cleaned_text", "compound", "date"]]
