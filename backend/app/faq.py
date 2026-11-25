import ipaddress
import json
import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)
DEFAULT_FAQS_PATH = Path(__file__).resolve().parent / "retrieval" / "faqs.json"
FAQ_TIMEOUT_SECONDS = 8
MAX_FAQ_BYTES = 1_000_000
FAQ_UNAVAILABLE_MESSAGE = "FAQ information is temporarily unavailable."
STOP_WORDS = {
    "a", "an", "are", "as", "at", "be", "can", "do", "does", "for",
    "i", "in", "is", "it", "me", "of", "on", "or", "the", "this", "to",
    "we", "when", "which", "who", "why", "you", "your",
}
SYNONYMS = {
    "location": {"address", "based", "located", "location", "office", "situated"},
    "contact": {"call", "connect", "contact", "reach", "speak", "talk", "telephone", "touch"},
    "service": {"assist", "assistance", "help", "offer", "provide", "service", "services"},
    "buy": {"buy", "buying", "purchase", "purchasing"},
    "sell": {"sell", "selling", "sale"},
    "inspect": {"inspection", "inspect", "see", "tour", "view", "viewing", "visit"},
    "price": {"cost", "price", "pricing"},
    "document": {"document", "documentation", "documents", "paperwork", "title"},
    "verify": {"check", "confirm", "genuine", "verify"},
    "photo": {"image", "images", "photo", "photograph", "photographs", "picture", "pictures"},
    "valuation": {"appraisal", "value", "valuation", "worth"},
    "payment": {"pay", "payment", "payments"},
    "property": {"home", "homes", "house", "houses", "land", "plot", "plots", "property", "properties"},
}
SYNONYM_LOOKUP = {word: root for root, words in SYNONYMS.items() for word in words}


class FAQUnavailable(Exception):
    pass


def _is_allowed_faq_url(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
    except ValueError:
        return False
    if not hostname or parsed.username or parsed.password:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    hostname = hostname.casefold().rstrip(".")
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_faqs(records):
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty FAQ array")
    ids = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid FAQ record")
        for field in ("id", "category", "question", "answer"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Missing FAQ field: " + field)
        if item["id"] in ids:
            raise ValueError("Duplicate FAQ ID")
        ids.add(item["id"])
        if not isinstance(item.get("featured"), bool):
            raise ValueError("Invalid FAQ featured flag")
    return records


def _parse_faqs(raw, source, default_missing_featured=False):
    try:
        records = json.loads(raw.decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        logger.warning("FAQ load failed: malformed JSON in %s", source)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error
    if default_missing_featured and isinstance(records, list):
        for item in records:
            if isinstance(item, dict):
                item.setdefault("featured", False)
    try:
        return validate_faqs(records)
    except (TypeError, ValueError) as error:
        logger.warning("FAQ load failed: unexpected response format in %s", source)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error


def _load_local_faqs():
    try:
        raw = DEFAULT_FAQS_PATH.read_bytes()
    except FileNotFoundError as error:
        logger.warning("FAQ load failed: local FAQ file is missing: %s", DEFAULT_FAQS_PATH)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error
    except OSError as error:
        logger.warning("FAQ load failed: could not read local FAQ file (%s)", type(error).__name__)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error
    return _parse_faqs(raw, "local FAQ file", default_missing_featured=True)


def load_faqs():
    url = os.getenv("FAQS_URL", "").strip()
    if not url:
        return _load_local_faqs()

    try:
        if not _is_allowed_faq_url(url):
            logger.warning("FAQ load failed: invalid FAQS_URL configuration")
            raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE)
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "PrinceB-PropertyAssistant/1.0"})
        with urlopen(request, timeout=FAQ_TIMEOUT_SECONDS) as response:
            if getattr(response, "status", 200) != 200:
                logger.warning("FAQ load failed: HTTP error status=%s", response.status)
                raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE)
            response_url = getattr(response, "url", url)
            if not _is_allowed_faq_url(response_url):
                logger.warning("FAQ load failed: unexpected API response URL")
                raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE)
            raw = response.read(MAX_FAQ_BYTES + 1)
    except HTTPError as error:
        logger.warning("FAQ load failed: HTTP error status=%s", error.code)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error
    except FAQUnavailable:
        raise
    except (TimeoutError, URLError, OSError) as error:
        logger.warning("FAQ load failed: HTTP request error (%s)", type(error).__name__)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error
    except Exception as error:
        logger.exception("FAQ load failed unexpectedly (%s)", type(error).__name__)
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE) from error

    if len(raw) > MAX_FAQ_BYTES:
        logger.warning("FAQ load failed: unexpected response format (response too large)")
        raise FAQUnavailable(FAQ_UNAVAILABLE_MESSAGE)
    return _parse_faqs(raw, "FAQ API response")


def _tokens(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = re.findall(r"[a-z0-9]+", normalized)
    return {
        SYNONYM_LOOKUP.get(word) or word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def _similarity(message, record):
    query_tokens = _tokens(message)
    question_tokens = _tokens(record["question"])
    if not query_tokens or not question_tokens:
        return 0.0
    overlap = len(query_tokens & question_tokens) / ((len(query_tokens) * len(question_tokens)) ** 0.5)
    normalized_query = " ".join(sorted(query_tokens))
    normalized_question = " ".join(sorted(question_tokens))
    character_similarity = SequenceMatcher(None, normalized_query, normalized_question).ratio()
    return 0.75 * overlap + 0.25 * character_similarity


def _normalize_exact_question(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return normalized[:-1].rstrip() if normalized.endswith("?") else normalized


def find_exact_faq(message, records):
    normalized_message = _normalize_exact_question(message)
    return next(
        (record for record in records if _normalize_exact_question(record["question"]) == normalized_message),
        None,
    )


def match_faq(message, records):
    normalized = " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", message).casefold()))
    matches = []
    for record in records:
        question = " ".join(re.findall(
            r"[a-z0-9]+",
            unicodedata.normalize("NFKC", record["question"]).casefold(),
        ))
        score = 1.0 if normalized == question else _similarity(message, record)
        matches.append((score, record))
    matches.sort(key=lambda result: result[0], reverse=True)
    best_score, best_record = matches[0]
    runner_up = matches[1][0] if len(matches) > 1 else 0.0
    return best_record, best_score, best_score - runner_up
