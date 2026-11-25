import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generation import generate
from .intent import MODEL_PATH, classify
from .moderation import moderate
from .retrieval import DEFAULT_LISTINGS, filters_from, load_listings, retrieve

app = FastAPI(title="Property Sales Chatbot")


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
    demo: bool = True


STATIC = {
    "greeting": "Hello! What type of property are you looking for, and where?",
    "faq": "Once you find a listing you like, you can ask the sales team about the next steps.",
    "book_viewing": "Please request a viewing through the website’s contact page and include the listing ID.",
    "contact_agent": "Use the website’s contact page to reach the sales team.",
    "out_of_scope": "I can help with properties for sale, listing details and viewing enquiries.",
}
ORDINALS = {"first": 0, "second": 1, "third": 2}


@app.get("/health")
def health():
    return {"status": "ok", "classifier_artifact_present": MODEL_PATH.is_file()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(422, "Enter a message.")
    moderation = moderate(message)
    demo = os.getenv("LISTINGS_FILE") in (None, str(DEFAULT_LISTINGS))
    if moderation == "threat":
        return ChatResponse(reply="I can’t assist with threatening messages.", moderation=moderation, demo=demo)

    prefix = "Please keep the conversation respectful. " if moderation == "directed_abuse" else ""
    try:
        intent, confidence = classify(message)
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error

    def respond(reply, listings=None, source="template"):
        return ChatResponse(
            reply=prefix + reply, intent=intent, confidence=confidence, moderation=moderation,
            listings=listings or [], source=source, demo=demo,
        )

    explicit_id = re.search(r"\bP\d+\b", message, re.IGNORECASE)
    ordinal = re.search(r"\b(first|second|third)\s+(?:one|listing|property)\b", message, re.IGNORECASE)
    follow_up = bool(request.previous_query and re.match(r"^(?:and|what about|how about|only)\b", message, re.IGNORECASE))
    if intent in {"book_viewing", "contact_agent"} and confidence >= 0.26:
        return respond(STATIC[intent])
    if explicit_id or ordinal:
        intent, confidence = "property_details", None
    elif follow_up:
        intent, confidence = "search_property", None
    elif confidence < 0.26:
        return respond("Are you looking for a property or asking about a specific listing?")
    if intent in STATIC:
        return respond(STATIC[intent])

    listings = load_listings()
    if intent == "property_details":
        listing_id = explicit_id[0].upper() if explicit_id else None
        if not listing_id and ordinal:
            position = ORDINALS[ordinal[1].lower()]
            if position < len(request.previous_listing_ids):
                listing_id = request.previous_listing_ids[position]
        if not listing_id:
            return respond("Which listing do you mean? Please enter its listing ID.")
        matches = [item for item in listings if item["id"] == listing_id and item["status"] == "active"]
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
        # Ask about locations outside the fixture instead of searching everywhere.
        if "location" not in new_filters and re.search(r"\b(?:in|around|near)\s+(?!a\b|an\b|the\b)[\w-]+", message, re.IGNORECASE):
            return respond("Which area should I search? Use the location name shown in the catalogue.")
        matches = retrieve(message, listings, filters)
        if not matches:
            return respond("I couldn’t find an active listing matching those filters. Which one can you change?")

    facts = "\n".join(
        f'{item["id"]} · {item["title"]} · GH₵{item["price_ghs"]:,.2f}. {item["description"]}'
        for item in matches
    )
    fallback = facts
    reply, source = generate(message, matches, fallback)
    if demo:
        reply = "Sample catalogue\n" + reply
    return respond(reply, matches, source)
