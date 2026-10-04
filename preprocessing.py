"""
preprocessing.py
Cleans raw customer feedback text before it is sent to the sentiment model.

Design notes:
- Kept dependency-light (pure regex) so it runs anywhere without a spaCy/NLTK
  download step. Swap in spaCy/NLTK lemmatization later if you need it —
  the `clean_text` function is the single seam to extend.
- We deliberately do NOT strip punctuation that carries sentiment signal
  (e.g. "!", "?", "...") before tokenizing for a lexicon model, but since
  we're feeding a transformer (which has its own tokenizer/subword model),
  we keep the text close to natural language and only strip noise.
"""

import re
import html
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
MULTI_SPACE_RE = re.compile(r"\s+")
MULTI_PUNCT_RE = re.compile(r"([!?.]){2,}")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def clean_text(raw: str, *, keep_emoji: bool = False) -> str:
    """Normalize a single piece of feedback text.

    Steps: unescape HTML entities -> unicode normalize -> strip URLs/mentions
    -> unwrap hashtags -> collapse repeated punctuation/whitespace.
    """
    if not raw:
        return ""

    text = html.unescape(raw)
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)  # "#slowdelivery" -> "slowdelivery"

    if not keep_emoji:
        text = EMOJI_RE.sub(" ", text)

    text = MULTI_PUNCT_RE.sub(r"\1", text)  # "!!!" -> "!"
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def batch_clean(texts: list[str], **kwargs) -> list[str]:
    return [clean_text(t, **kwargs) for t in texts]


# Very small stoplist for the keyword/issue-extraction feature in main.py.
# Deliberately short — over-stripping hurts the "what are people complaining
# about" signal more than it helps.
STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "and", "or", "but", "to",
    "of", "in", "on", "for", "with", "it", "this", "that", "i", "we", "you",
    "my", "our", "your", "at", "be", "as", "so", "very", "just", "not",
    "have", "has", "had", "did", "do", "does", "can", "could", "would",
}


def extract_keywords(text: str, min_len: int = 4) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]
