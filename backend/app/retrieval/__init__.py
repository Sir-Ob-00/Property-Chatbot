import json
import os
import re
from decimal import Decimal
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_LISTINGS = Path(__file__).with_name("sample_listings.json")
PROPERTY_TYPES = {
    "house": r"houses?|homes?",
    "land": r"land|lands|plots?",
    "apartment": r"apartments?|flats?",
}
NUMBER = r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|m\b|k\b)?"


def load_listings():
    path = Path(os.getenv("LISTINGS_FILE", str(DEFAULT_LISTINGS)))
    return json.loads(path.read_text(encoding="utf-8"))


def filters_from(message, listings):
    text = message.casefold()
    locations = sorted({item["location"] for item in listings}, key=lambda name: (-len(name), name))
    filters = {}
    for location in locations:
        if re.search(r"(?<!\w)" + re.escape(location.casefold()) + r"(?!\w)", text):
            filters["location"] = location
            break
    for property_type, pattern in PROPERTY_TYPES.items():
        if re.search(r"\b(?:" + pattern + r")\b", text):
            filters["property_type"] = property_type
            break
    bedroom = re.search(r"\b(\d+)\s*[- ]?\s*(?:bedrooms?|beds?)\b", text)
    if bedroom:
        filters["bedrooms"] = int(bedroom[1])
    budget = re.search(
        r"\b(under|below|less than|up to|at most|max(?:imum)?|budget(?:\s+is)?|no more than)"
        r"[:\s]+(?:gh\s*[₵¢s]?\s*|₵\s*)?" + NUMBER,
        text,
    )
    if budget:
        multiplier = {"k": 1000, "m": 1000000, "million": 1000000}.get(budget[3], 1)
        filters["max_price"] = float(Decimal(budget[2].replace(",", "")) * multiplier)
        filters["exclusive_max"] = budget[1] in {"under", "below", "less than"}
    return filters


def retrieve(message, listings, filters, limit=3):
    candidates = []
    for item in listings:
        if item["status"] != "active":
            continue
        if any(item[field] != filters[field] for field in ("location", "property_type", "bedrooms") if field in filters):
            continue
        if "max_price" in filters:
            price = item["price_ghs"]
            if price > filters["max_price"] or (filters["exclusive_max"] and price == filters["max_price"]):
                continue
        candidates.append(item)
    if not candidates:
        return []
    documents = [f'{item["title"]} {item["description"]}' for item in candidates]
    vectors = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([message, *documents])
    scores = (vectors[1:] @ vectors[0].T).toarray().ravel()
    ranked = sorted(zip(scores, candidates), key=lambda pair: (-pair[0], pair[1]["price_ghs"], pair[1]["id"]))
    return [item for _, item in ranked[:limit]]
