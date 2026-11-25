import json
import os
import re
import logging
import math
from urllib.request import Request, urlopen
from urllib.parse import urlparse
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


class CatalogueUnavailable(Exception):
    pass


def demo_mode():
    return not os.getenv("LISTINGS_URL", "").strip()


def validate_listings(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty listing array")
    ids = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid listing record")
        listing_id = item.get("id")
        if not isinstance(listing_id, str) or not listing_id.strip():
            raise ValueError("Missing listing field: id")
        if listing_id in ids or not re.fullmatch(r"[A-Za-z0-9_-]+", listing_id):
            raise ValueError("Invalid or duplicate listing ID")
        ids.add(listing_id)

        for field in ("title", "location", "description"):
            value = item.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError("Invalid listing field: " + field)
        property_type = item.get("property_type")
        if property_type is not None and (
            not isinstance(property_type, str) or property_type not in PROPERTY_TYPES
        ):
            raise ValueError("Invalid property type")
        status = item.get("status")
        if status is not None and (
            not isinstance(status, str)
            or status not in {"active", "unavailable", "sold", "reserved", "inactive", "unknown"}
        ):
            raise ValueError("Invalid property type or status")
        price = item.get("price_ghs")
        if price is not None and (
            type(price) not in (int, float)
            or price < 0
            or (isinstance(price, float) and not math.isfinite(price))
        ):
            raise ValueError("Invalid price")
        bedrooms = item.get("bedrooms")
        if bedrooms is not None and (type(bedrooms) is not int or bedrooms < 0):
            raise ValueError("Invalid bedrooms")
        features = item.get("features")
        if features is not None and (
            not isinstance(features, list)
            or any(feature is not None and not isinstance(feature, str) for feature in features)
        ):
            raise ValueError("Invalid features")
        url_value = item.get("url")
        if url_value is not None:
            if not isinstance(url_value, str):
                raise ValueError("Invalid listing URL")
            url = urlparse(url_value)
            if url.scheme != "https" or not url.netloc:
                raise ValueError("Listing URLs must use HTTPS")

        item.setdefault("price_ghs", None)
        item.setdefault("bedrooms", None)
    return records


def load_listings():
    try:
        url = os.getenv("LISTINGS_URL", "").strip()
        if not url:
            return validate_listings(json.loads(DEFAULT_LISTINGS.read_text(encoding="utf-8")))
        if urlparse(url).scheme != "https":
            raise ValueError("Configure an HTTPS LISTINGS_URL")
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "PrinceB-PropertyAssistant/1.0"})
        with urlopen(request, timeout=8) as response:
            if getattr(response, "status", 200) != 200:
                raise ValueError("Catalogue returned an unsuccessful status")
            if urlparse(response.url).scheme != "https":
                raise ValueError("Catalogue redirected away from HTTPS")
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Catalogue too large")
        return validate_listings(json.loads(raw.decode("utf-8-sig")))
    except Exception as error:
        logging.getLogger(__name__).warning("Catalogue load failed: %s", type(error).__name__)
        raise CatalogueUnavailable("The property catalogue is temporarily unavailable. Please try again shortly or contact our team.") from error


def filters_from(message, listings):
    text = message.casefold()
    locations = sorted(
        {item["location"] for item in listings if isinstance(item.get("location"), str)},
        key=lambda name: (-len(name), name),
    )
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
        if item.get("status") != "active":
            continue
        if any(item.get(field) != filters[field] for field in ("location", "property_type", "bedrooms") if field in filters):
            continue
        if "max_price" in filters:
            price = item.get("price_ghs")
            if price is None or price > filters["max_price"] or (filters["exclusive_max"] and price == filters["max_price"]):
                continue
        candidates.append(item)
    if not candidates:
        return []
    documents = [
        " ".join(
            value for value in (
                item.get("title") or "",
                item.get("description") or "",
                " ".join(feature for feature in (item.get("features") or []) if isinstance(feature, str)),
            )
            if value
        )
        for item in candidates
    ]
    vectors = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([message, *documents])
    scores = (vectors[1:] @ vectors[0].T).toarray().ravel()
    ranked = sorted(
        zip(scores, candidates),
        key=lambda pair: (
            -pair[0],
            pair[1].get("price_ghs") if pair[1].get("price_ghs") is not None else float("inf"),
            pair[1]["id"],
        ),
    )
    return [item for _, item in ranked[:limit]]
