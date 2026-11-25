import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generation import generate
from .faq import FAQUnavailable, find_exact_faq, load_faqs, match_faq
from .intent import MODEL_PATH, classify
from .moderation import moderate
from .retrieval import CatalogueUnavailable, demo_mode, filters_from, load_listings, retrieve

app = FastAPI(title="Property Sales Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "https://jaydeladem.github.io").split(",") if origin.strip()],
    allow_methods=["POST", "GET"], allow_headers=["Content-Type"],
)



class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1200)
    previous_query: str | None = Field(default=None, max_length=1200)
    previous_listing_ids: list[str] = Field(default_factory=list, max_length=3)


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
    confidence: float | None = None
    moderation: str
    source: Literal["template", "gemini"] = "template"
    listings: list[dict] = Field(default_factory=list)
    demo: bool = False
    search_query: str | None = None


STATIC = {
    "greeting": "Hello! What type of property are you looking for, and where?",
    "faq": "Once you find a listing you like, you can ask the sales team about the next steps.",
    "book_viewing": "Please request a viewing through the website’s contact page and include the listing ID.",
    "contact_agent": "Use the website’s contact page to reach the sales team.",
    "out_of_scope": "I can help with properties for sale, listing details and viewing enquiries.",
}
FRUSTRATION_REPLY = "I’m sorry this has been frustrating. What went wrong? I’ll do my best to help."
ORDINALS = {"first": 0, "second": 1, "third": 2}
FAQ_MATCH_THRESHOLD = 0.34
QUESTION_START = re.compile(r"^(?:who|what|when|where|why|how|can|could|do|does|is|are|should|will)\b", re.IGNORECASE)
PROPERTY_TERMS = re.compile(
    r"\b(?:properties|property|houses?|homes?|apartments?|flats?|land|plots?|listings?|bedrooms?|beds?|price)\b",
    re.IGNORECASE,
)
IDENTITY_QUESTION = re.compile(
    r"\b(?:who are you|what are you|what(?:'s| is) your name|your name|"
    r"are you (?:(?:a|an) )?(?:bot|chatbot|robot|ai|assistant)|are you human)\b",
    re.IGNORECASE,
)
LISTING_ID = re.compile(r"(?<![\w-])(?:house|apartment|land)-\d+(?![\w-])", re.IGNORECASE)
PROPERTY_SEARCH_ACTION = re.compile(
    r"\b(?:show|search|browse|list|find me|look(?:ing)? for|available|for sale|options|"
    r"do you have|any)\b",
    re.IGNORECASE,
)


@app.get("/health")
def health():
    return {"status": "ok", "classifier_artifact_present": MODEL_PATH.is_file()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(422, "Enter a message.")
    moderation = moderate(message)
    demo = demo_mode()
    if moderation == "threat":
        return ChatResponse(reply="I can’t assist with threatening messages.", moderation=moderation, demo=demo)

    prefix = "Please keep the conversation respectful. " if moderation == "directed_abuse" else ""
    if IDENTITY_QUESTION.search(message):
        return ChatResponse(
            reply=prefix + "I’m Prinbot, the Prince B Realty Consult property assistant. I can help with property searches and questions about our services.",
            intent="assistant_identity",
            moderation=moderation,
            demo=demo,
        )

    explicit_id_pattern = LISTING_ID.search(message)
    ordinal = re.search(r"\b(first|second|third)\s+(?:one|listing|property)\b", message, re.IGNORECASE)
    follow_up = bool(request.previous_query and re.match(r"^(?:and|what about|how about|only)\b", message, re.IGNORECASE))
    pronoun_follow_up = bool(
        request.previous_listing_ids
        and re.search(r"\b(?:this|that|it|one|listing|place)\b", message, re.IGNORECASE)
    )
    listing_context = bool(explicit_id_pattern or ordinal or follow_up or pronoun_follow_up)
    question_like = bool(QUESTION_START.search(message) or "?" in message)
    faqs = None
    faq_load_attempted = False
    if not listing_context and question_like:
        faq_load_attempted = True
        try:
            faqs = load_faqs()
        except FAQUnavailable:
            pass
        else:
            exact_faq = find_exact_faq(message, faqs)
            if exact_faq:
                return ChatResponse(
                    reply=prefix + exact_faq["answer"],
                    intent="faq",
                    confidence=1.0,
                    moderation=moderation,
                    demo=demo,
                    search_query=request.previous_query,
                )

    try:
        intent, confidence = classify(message)
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error

    search_query = request.previous_query

    def respond(reply, listings=None, source="template"):
        return ChatResponse(
            reply=prefix + reply, intent=intent, confidence=confidence, moderation=moderation,
            listings=listings or [], source=source, demo=demo,
            search_query=search_query,
        )

    try:
        listings = load_listings()
    except CatalogueUnavailable as error:
        listings = None
        catalogue_error = error
    else:
        catalogue_error = None

    explicit_id = next(
        (item["id"] for item in listings or [] if re.search(
            r"(?<![\w-])" + re.escape(item["id"]) + r"(?![\w-])", message, re.IGNORECASE
        )),
        None,
    )
    if explicit_id:
        listing_context = True

    filters = filters_from(message, listings) if listings else {}
    is_clear_search = bool(
        "bedrooms" in filters
        or "max_price" in filters
        or (
            "location" in filters
            and ("property_type" in filters or PROPERTY_SEARCH_ACTION.search(message))
        )
        or (
            "property_type" in filters
            and PROPERTY_SEARCH_ACTION.search(message)
        )
    )
    listing_context = listing_context or is_clear_search
    if is_clear_search and intent != "property_details":
        intent = "search_property"

    faq_request = not listing_context and (
        question_like
        or intent == "faq"
    )
    command_action = bool(
        re.match(r"^(?:book|schedule|arrange|reserve|call|connect|contact)\b", message, re.IGNORECASE)
        and not question_like
    )
    if faq_request and not command_action:
        if not faq_load_attempted:
            faq_load_attempted = True
            try:
                faqs = load_faqs()
            except FAQUnavailable:
                pass
        if faqs is None:
            if intent == "faq":
                intent = "faq"
                confidence = None
                return respond(
                    "FAQ information is temporarily unavailable right now. Please contact Prince B Realty Consult through the website for help."
                )
        else:
            faq, score, margin = match_faq(message, faqs)
            if score >= FAQ_MATCH_THRESHOLD and margin >= 0.025:
                intent = "faq"
                confidence = score
                return respond(faq["answer"])

    has_listing_reference = bool(explicit_id_pattern or ordinal or follow_up or pronoun_follow_up or explicit_id)
    has_clear_property_request = bool(
        is_clear_search
        or (
            PROPERTY_TERMS.search(message)
            and PROPERTY_SEARCH_ACTION.search(message)
        )
        or (
            intent == "property_details"
            and PROPERTY_TERMS.search(message)
        )
    )
    has_agent_request = intent in {"contact_agent", "book_viewing"} and confidence >= 0.26
    if moderation in {"directed_abuse", "mild_profanity"} and not (
        has_listing_reference or has_clear_property_request or has_agent_request
    ):
        intent = "out_of_scope"
        prefix = ""
        return respond(FRUSTRATION_REPLY)

    if listings is None:
        if intent in {"book_viewing", "contact_agent"} and confidence >= 0.26:
            return respond(STATIC[intent])
        raise HTTPException(status_code=503, detail=str(catalogue_error)) from catalogue_error

    if not (explicit_id or ordinal or follow_up) and intent in {"book_viewing", "contact_agent"} and confidence >= 0.26:
        return respond(STATIC[intent])
    if explicit_id or ordinal:
        intent, confidence = "property_details", None
    elif follow_up:
        intent, confidence = "search_property", None
    elif confidence < 0.26:
        intent = "out_of_scope"
        return respond("Are you looking for a property or asking about a specific listing?")
    if intent in STATIC:
        return respond(STATIC[intent])

    if intent == "property_details":
        listing_id = explicit_id
        if not listing_id and ordinal:
            position = ORDINALS[ordinal[1].lower()]
            if position < len(request.previous_listing_ids):
                listing_id = request.previous_listing_ids[position]
        if not listing_id:
            return respond("Which listing do you mean? Please enter its listing ID.")
        matches = [item for item in listings if item["id"] == listing_id and item.get("status") == "active"]
        if not matches:
            return respond("That listing is currently unavailable in the catalogue.")
    else:
        filters = filters_from(request.previous_query, listings) if follow_up else {}
        new_filters = filters_from(message, listings)
        if follow_up and not new_filters:
            return respond("Which location, budget or property type would you like to change?")
        if new_filters.get("property_type") == "land":
            filters.pop("bedrooms", None)
        filters.update(new_filters)
        # Encode the effective filters, avoiding accumulation of stale locations/budgets.
        search_query = " ".join(str(filters[key]) for key in ("location", "property_type") if key in filters)
        if "bedrooms" in filters:
            search_query += f" {filters['bedrooms']}-bedroom"
        if "max_price" in filters:
            search_query += f" {'under' if filters['exclusive_max'] else 'up to'} {filters['max_price']:.2f}"

        # Ask about locations outside the fixture instead of searching everywhere.
        if "location" not in new_filters and re.search(r"\b(?:in|around|near)\s+(?!a\b|an\b|the\b)[\w-]+", message, re.IGNORECASE):
            return respond("Which area should I search? Use the location name shown in the catalogue.")
        matches = retrieve(message, listings, filters)
        if not matches:
            return respond("I couldn’t find an active listing matching those filters. Which one can you change?")

    facts = []
    for item in matches:
        price_value = item.get("price_ghs")
        price = "Price unknown" if price_value is None else f"GHS {price_value:,.2f}"
        bedrooms = "Bedrooms unknown" if item.get("bedrooms") is None else f'{item["bedrooms"]} bedrooms'
        features_value = item.get("features")
        if isinstance(features_value, list):
            features = "; ".join(feature for feature in features_value if isinstance(feature, str)) or "Amenities unknown"
        else:
            features = features_value or "Amenities unknown"
        details = [
            f"{key.replace('_', ' ').capitalize()}: {value if value is not None else 'unknown'}"
            for key, value in item.items()
            if key not in {"id", "title", "location", "property_type", "bedrooms", "price_ghs", "status", "description", "features", "url"}
        ]
        url = item.get("url") or "Property link unknown"
        facts.append(
            f'{item["id"]} · {item.get("title") or "Title unknown"} · '
            f'{item.get("property_type") or "Property type unknown"} · '
            f'{item.get("location") or "Location unknown"} · {bedrooms} · {price} · '
            f'Status: {item.get("status") or "unknown"}. '
            f'Description: {item.get("description") or "unknown"}. Features: {features}. '
            f'{"; ".join(details)} {url}'
        )
        if item.get("property_type") == "land":
            facts.append("Land documentation is not verified by this catalogue. Contact Prince B Realty Consult to confirm documentation.")
    fallback = "\n".join(facts) + "\nDetails not stated in the catalogue are unknown; please confirm with our team."
    reply, source = generate(message, matches, fallback)
    if demo:
        reply = "Sample catalogue\n" + reply
    return respond(reply, matches, source)
