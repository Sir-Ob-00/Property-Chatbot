import asyncio
import copy
from io import BytesIO
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.pop("GEMINI_API_KEY", None)
from backend.app import main, retrieval
from backend.app.generation import generate
from fastapi import HTTPException


LIVE_URL = "https://jaydeladem.github.io/prince-b-realty-consult/listings.json"
PROPERTY_BASE = "https://jaydeladem.github.io/prince-b-realty-consult/property.html?id="
records = [
    {
        "id": "house-1",
        "title": "Three-bedroom house in East Legon",
        "location": "East Legon",
        "property_type": "house",
        "bedrooms": 3,
        "price_ghs": 850000,
        "status": "active",
        "description": "Family home with a dedicated study.",
        "features": ["Covered parking"],
        "url": PROPERTY_BASE + "house-1",
    },
    {
        "id": "apartment-1",
        "title": "Spintex apartment",
        "location": "Spintex",
        "property_type": "apartment",
        "bedrooms": 2,
        "price_ghs": 420000,
        "status": "active",
        "description": "Two-bedroom apartment.",
        "features": ["Secure parking"],
        "url": PROPERTY_BASE + "apartment-1",
    },
    {
        "id": "land-1",
        "title": "Oyibi residential land",
        "location": "Oyibi",
        "property_type": "land",
        "bedrooms": None,
        "bathrooms": None,
        "price_ghs": 180000,
        "status": "active",
        "description": "Residential plot measuring 50 × 70 ft.",
        "features": ["Documentation status unknown"],
        "documentation_verified": None,
        "url": PROPERTY_BASE + "land-1",
    },
]


class IntegrationTests(unittest.TestCase):
    def chat(self, message, **kwargs):
        with (
            patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}),
            patch.object(main, "classify", return_value=("search_property", 0.9)),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
        ):
            return main.chat(main.ChatRequest(message=message, **kwargs))

    def test_search_and_property_link(self):
        result = self.chat("Show me a 3-bedroom house in East Legon under GH₵900,000")
        self.assertFalse(result.demo)
        self.assertEqual(result.listings[0]["id"], "house-1")
        self.assertEqual(result.listings[0]["price_ghs"], 850000)
        self.assertEqual(result.listings[0]["url"], PROPERTY_BASE + "house-1")
        self.assertEqual(result.source, "template")
        self.assertFalse(self.chat("Show me houses in East Legon under 850000").listings)

    def test_direct_listing_questions_and_unknown_details(self):
        apartment = self.chat("Tell me about apartment-1")
        self.assertEqual(apartment.listings[0]["location"], "Spintex")
        self.assertEqual(apartment.listings[0]["price_ghs"], 420000)
        self.assertIn("Secure parking", apartment.reply)

        land = self.chat("Tell me about land-1")
        self.assertIn("50 × 70 ft", land.reply)
        self.assertIn("documentation is not verified", land.reply)
        self.assertIn("Prince B Realty Consult", land.reply)
        self.assertIn("Bathrooms: unknown", land.reply)
        self.assertIsNone(land.listings[0]["bathrooms"])

    def test_gemini_context_includes_listing_description_and_features(self):
        response = BytesIO(json.dumps({
            "candidates": [{"content": {"parts": [{"text": "Parking is listed."}]}}]
        }).encode())
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "test-model"}),
            patch("backend.app.generation.urlopen", return_value=response) as mocked_urlopen,
        ):
            reply, source = generate("Does house-1 have parking?", [records[0]], "fallback")

        payload = json.loads(mocked_urlopen.call_args.args[0].data)
        context = json.loads(payload["contents"][0]["parts"][0]["text"])
        self.assertEqual(source, "gemini")
        self.assertEqual(reply, "Parking is listed.")
        self.assertEqual(context["listings"][0]["description"], records[0]["description"])
        self.assertEqual(context["listings"][0]["features"], records[0]["features"])

    def test_second_listing_follow_up(self):
        result = self.chat("Does the second one have parking?", previous_listing_ids=["land-1", "apartment-1"])
        self.assertEqual(result.listings[0]["id"], "apartment-1")
        self.assertIn("Secure parking", result.reply)

    def test_live_catalogue_failure_is_http_503(self):
        with (
            patch.object(main, "classify", return_value=("search_property", 0.9)),
            patch.object(main, "load_listings", side_effect=retrieval.CatalogueUnavailable("Catalogue unavailable")),
            self.assertRaises(HTTPException) as raised,
        ):
            main.chat(main.ChatRequest(message="Show me houses"))
        self.assertEqual(raised.exception.status_code, 503)

    def test_live_fetch_and_validation(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.url = LIVE_URL
        response.status = 200
        response.read.return_value = json.dumps(records).encode()
        with patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}), patch.object(retrieval, "urlopen", return_value=response):
            self.assertEqual(retrieval.load_listings()[0]["id"], "house-1")

        with patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}), patch.object(retrieval, "urlopen", side_effect=TimeoutError):
            with self.assertRaises(retrieval.CatalogueUnavailable):
                retrieval.load_listings()

        malformed = copy.deepcopy(records)
        malformed[0]["price_ghs"] = "850000"
        with self.assertRaises(ValueError):
            retrieval.validate_listings(malformed)

    def test_local_sample_is_selected_when_url_is_unset(self):
        with patch.dict(os.environ, {"LISTINGS_URL": "", "DEMO_MODE": "false"}):
            self.assertTrue(retrieval.demo_mode())
            sample = retrieval.load_listings()
            self.assertTrue(sample)
            with patch.object(main, "classify", return_value=("search_property", 0.9)):
                result = main.chat(main.ChatRequest(message="Show me 3-bedroom houses in Tema under 900000"))
            self.assertTrue(result.demo)
            self.assertEqual(result.listings[0]["id"], "P101")

    def test_browser_cors_preflight(self):
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "OPTIONS",
            "scheme": "https",
            "path": "/chat",
            "raw_path": b"/chat",
            "query_string": b"",
            "headers": [
                (b"origin", b"https://jaydeladem.github.io"),
                (b"access-control-request-method", b"POST"),
                (b"access-control-request-headers", b"content-type"),
            ],
            "server": ("test", 443),
            "client": ("test", 12345),
        }
        with patch.dict(os.environ, {"CORS_ORIGINS": "https://jaydeladem.github.io"}):
            asyncio.run(main.app(scope, receive, send))
        headers = dict(sent[0]["headers"])
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(headers[b"access-control-allow-origin"], b"https://jaydeladem.github.io")
        self.assertIn(b"POST", headers[b"access-control-allow-methods"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
