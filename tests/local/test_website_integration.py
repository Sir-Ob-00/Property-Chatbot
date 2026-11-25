import asyncio
import copy
from io import BytesIO
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.pop("GEMINI_API_KEY", None)
from backend.app import main, retrieval
from backend.app import faq as faq_module
from backend.app.generation import generate
from fastapi import HTTPException
from backend.app.faq import FAQUnavailable


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
faq_records = [
    {
        "id": "faq-location",
        "category": "General Questions",
        "question": "Where is Prince B Realty Consult located?",
        "answer": "Our office is at Agbotui Flat, Ashaiman Roundabout, Greater Accra Region, Ghana.",
        "featured": True,
    },
    {
        "id": "faq-buying",
        "category": "Property Buying Questions",
        "question": "Can Prince B Realty Consult help me find a property to buy?",
        "answer": "Yes. Share your preferred location, budget and property requirements.",
        "featured": False,
    },
    {
        "id": "faq-inspection",
        "category": "Property Buying Questions",
        "question": "Can I inspect a property before buying it?",
        "answer": "Contact us to arrange an inspection, subject to availability and the owner's approval.",
        "featured": False,
    },
]


class IntegrationTests(unittest.TestCase):
    def post_chat(self, message, **kwargs):
        body = json.dumps({"message": message, **kwargs}, ensure_ascii=False).encode("utf-8")
        sent = []
        requests = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return requests.pop(0) if requests else {"type": "http.disconnect"}

        async def send(event):
            sent.append(event)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/chat",
            "raw_path": b"/chat",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "server": ("test", 8000),
            "client": ("test", 12345),
        }
        asyncio.run(main.app(scope, receive, send))
        status = next(event["status"] for event in sent if event["type"] == "http.response.start")
        response_body = b"".join(event.get("body", b"") for event in sent if event["type"] == "http.response.body")
        return status, json.loads(response_body)

    def chat(self, message, **kwargs):
        with (
            patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}),
            patch.object(main, "classify", return_value=("search_property", 0.9)),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
            patch.object(main, "load_faqs", return_value=copy.deepcopy(faq_records)),
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

    def test_exact_and_paraphrased_faqs_override_classifier(self):
        exact = self.chat("Where is Prince B Realty Consult located?")
        self.assertIn("Agbotui Flat", exact.reply)
        self.assertEqual(exact.intent, "faq")
        self.assertFalse(exact.listings)

        paraphrase = self.chat("Where are you based?")
        self.assertIn("Agbotui Flat", paraphrase.reply)
        self.assertEqual(paraphrase.intent, "faq")

    def test_exact_local_faqs_precede_classifier_intent(self):
        with patch.dict(os.environ, {"FAQS_URL": ""}):
            local_faqs = faq_module.load_faqs()
            for message, faq_id in (
                ("Can I make payments in instalments?", "faq-36"),
                ("What factors affect the value of a property?", "faq-23"),
                ("  WHAT   FACTORS affect the value of a property?  ", "faq-23"),
            ):
                approved_answer = next(item["answer"] for item in local_faqs if item["id"] == faq_id)
                with (
                    patch.object(main, "classify") as classifier,
                    patch.object(main, "load_listings") as listing_loader,
                ):
                    result = main.chat(main.ChatRequest(message=message))
                self.assertEqual(result.intent, "faq")
                self.assertEqual(result.reply, approved_answer)
                classifier.assert_not_called()
                listing_loader.assert_not_called()

    def test_listing_ids_and_ordinal_followups_bypass_exact_faq_lookup(self):
        with (
            patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}),
            patch.object(main, "classify", return_value=("property_details", 0.9)),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
            patch.object(main, "load_faqs") as faq_loader,
        ):
            by_id = main.chat(main.ChatRequest(message="Is house-1 still available?"))
            by_ordinal = main.chat(main.ChatRequest(
                message="Does the second one have parking?",
                previous_listing_ids=["land-1", "apartment-1"],
            ))
        self.assertEqual(by_id.listings[0]["id"], "house-1")
        self.assertEqual(by_ordinal.listings[0]["id"], "apartment-1")
        self.assertIn("Secure parking", by_ordinal.reply)
        faq_loader.assert_not_called()

    def test_faq_still_answers_when_property_classifier_would_search(self):
        with (
            patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}),
            patch.object(main, "classify", return_value=("search_property", 0.9)),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
            patch.object(main, "load_faqs", return_value=copy.deepcopy(faq_records)),
        ):
            result = main.chat(main.ChatRequest(message="Can I inspect a property before buying it?"))
        self.assertIn("subject to availability", result.reply)
        self.assertEqual(result.intent, "faq")

    def test_uncertain_faq_match_asks_for_clarification(self):
        with (
            patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}),
            patch.object(main, "classify", return_value=("out_of_scope", 0.8)),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
            patch.object(main, "load_faqs", return_value=copy.deepcopy(faq_records)),
        ):
            result = main.chat(main.ChatRequest(message="What is your contact?"))
        self.assertEqual(result.intent, "out_of_scope")
        self.assertIn("I can help with properties", result.reply)

    def test_faq_api_failure_is_helpful_and_does_not_block_property_search(self):
        with (
            patch.dict(os.environ, {"LISTINGS_URL": LIVE_URL}),
            patch.object(main, "classify", return_value=("faq", 0.9)),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
            patch.object(main, "load_faqs", side_effect=FAQUnavailable("Unavailable")),
        ):
            faq_result = main.chat(main.ChatRequest(message="Where are you based?"))
            property_result = main.chat(main.ChatRequest(
                message="Show me a 3-bedroom house in East Legon under 900000"
            ))
        self.assertIn("FAQ information is temporarily unavailable", faq_result.reply)
        self.assertEqual(property_result.listings[0]["id"], "house-1")

    def test_moderation_precedes_faq_fetch_and_identity_uses_prinbot(self):
        with patch.object(main, "load_faqs") as load_faqs:
            result = main.chat(main.ChatRequest(message="I will kill you. Where are you based?"))
        self.assertEqual(result.moderation, "threat")
        load_faqs.assert_not_called()

        with patch.object(main, "classify", return_value=("out_of_scope", 0.9)):
            identity = self.chat("Who are you?")
        self.assertIn("Prinbot", identity.reply)
        self.assertEqual(identity.intent, "assistant_identity")

    def test_chat_routing_regressions_through_post(self):
        with patch.dict(os.environ, {"FAQS_URL": "", "LISTINGS_URL": LIVE_URL}):
            local_faqs = faq_module.load_faqs()
            valuation_answer = next(item["answer"] for item in local_faqs if item["id"] == "faq-23")
            payment_answer = next(item["answer"] for item in local_faqs if item["id"] == "faq-36")
            with patch.object(main, "load_listings", return_value=copy.deepcopy(records)):
                cases = [
                    ("Hi", "greeting"),
                    ("Are you a bot?", "assistant_identity"),
                    ("What factors affect the value of a property?", "faq"),
                    ("Can I make payments in instalments?", "faq"),
                    ("Show me a 3-bedroom house in East Legon under GH₵900,000", "search_property"),
                    ("Is house-1 still available?", "property_details"),
                    ("Can you fix my laptop?", "out_of_scope"),
                ]
                responses = {}
                for message, _ in cases:
                    status, payload = self.post_chat(
                        message,
                        previous_listing_ids=["house-1", "apartment-1", "land-1"]
                        if message == "Does the second one have parking?"
                        else [],
                    )
                    self.assertEqual(status, 200, message)
                    responses[message] = payload

                status, followup = self.post_chat(
                    "Does the second one have parking?",
                    previous_listing_ids=["house-1", "apartment-1", "land-1"],
                )
                self.assertEqual(status, 200)

        for message, expected_intent in cases:
            self.assertEqual(responses[message]["intent"], expected_intent, message)
        self.assertIn("Prinbot", responses["Are you a bot?"]["reply"])
        self.assertEqual(responses["What factors affect the value of a property?"]["reply"], valuation_answer)
        self.assertEqual(responses["Can I make payments in instalments?"]["reply"], payment_answer)
        self.assertEqual(responses["Show me a 3-bedroom house in East Legon under GH₵900,000"]["listings"][0]["id"], "house-1")
        self.assertEqual(responses["Is house-1 still available?"]["listings"][0]["id"], "house-1")
        self.assertEqual(followup["intent"], "property_details")
        self.assertEqual(followup["listings"][0]["id"], "apartment-1")
        laptop = responses["Can you fix my laptop?"]
        self.assertNotEqual(laptop["intent"], "faq")
        self.assertNotIn(laptop["reply"], [item["answer"] for item in local_faqs])

    def test_frustrated_and_profane_messages_route_through_post(self):
        with patch.dict(os.environ, {"FAQS_URL": ""}):
            local_faqs = faq_module.load_faqs()
        valuation_answer = next(item["answer"] for item in local_faqs if item["id"] == "faq-23")
        messages = [
            "Shame, Are you out of your mind, bullshit?",
            "Hi",
            "bullshit, show me a 3-bedroom house in East Legon under GH₵900,000",
            "bullshit, I want to speak to someone",
            "bullshit, What factors affect the value of a property?",
            "shame",
        ]
        with (
            patch.dict(os.environ, {"FAQS_URL": "", "LISTINGS_URL": LIVE_URL}),
            patch.object(main, "load_listings", return_value=copy.deepcopy(records)),
        ):
            responses = {}
            for message in messages:
                status, payload = self.post_chat(message)
                self.assertEqual(status, 200, message)
                responses[message] = payload

        frustration = responses[messages[0]]
        self.assertEqual(frustration["intent"], "out_of_scope")
        self.assertEqual(frustration["moderation"], "directed_abuse")
        self.assertIn("What went wrong?", frustration["reply"])
        self.assertNotIn("respectful", frustration["reply"].lower())

        self.assertEqual(responses["Hi"]["intent"], "greeting")
        self.assertEqual(responses["Hi"]["moderation"], "clean")

        property_search = responses[messages[2]]
        self.assertEqual(property_search["intent"], "search_property")
        self.assertEqual(property_search["moderation"], "mild_profanity")
        self.assertEqual(property_search["listings"][0]["id"], "house-1")

        human_request = responses[messages[3]]
        self.assertEqual(human_request["intent"], "contact_agent")
        self.assertEqual(human_request["moderation"], "mild_profanity")
        self.assertIn("contact page", human_request["reply"])

        faq_request = responses[messages[4]]
        self.assertEqual(faq_request["intent"], "faq")
        self.assertEqual(faq_request["reply"], valuation_answer)

        shame_alone = responses["shame"]
        self.assertEqual(shame_alone["moderation"], "clean")
        self.assertNotIn("respectful", shame_alone["reply"].lower())

    def test_faq_api_loader_uses_full_url_and_validates_response(self):
        response = BytesIO(json.dumps(faq_records).encode())
        response.status = 200
        response.url = "https://cms.example.com/api/v1/faqs"
        with (
            patch.dict(os.environ, {"FAQS_URL": response.url}),
            patch.object(faq_module, "urlopen", return_value=response) as mocked_urlopen,
        ):
            loaded = faq_module.load_faqs()
        self.assertEqual(loaded, faq_records)
        self.assertEqual(mocked_urlopen.call_args.args[0].full_url, response.url)
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], faq_module.FAQ_TIMEOUT_SECONDS)

    def test_faq_api_loader_allows_mocked_loopback_http_only(self):
        response = BytesIO(json.dumps(faq_records).encode())
        response.status = 200
        response.url = "http://127.0.0.1:3000/api/v1/faqs"
        with (
            patch.dict(os.environ, {"FAQS_URL": response.url}),
            patch.object(faq_module, "urlopen", return_value=response) as mocked_urlopen,
        ):
            self.assertEqual(faq_module.load_faqs(), faq_records)
        self.assertEqual(mocked_urlopen.call_args.args[0].full_url, response.url)

        with (
            patch.dict(os.environ, {"FAQS_URL": "http://example.com/api/v1/faqs"}),
            patch.object(faq_module, "urlopen") as mocked_urlopen,
            self.assertRaises(FAQUnavailable),
        ):
            faq_module.load_faqs()
        mocked_urlopen.assert_not_called()

    def test_local_faq_fallback_uses_module_relative_path(self):
        expected_path = Path(faq_module.__file__).resolve().parent / "retrieval" / "faqs.json"
        self.assertEqual(faq_module.DEFAULT_FAQS_PATH, expected_path)
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                with patch.dict(os.environ, {"FAQS_URL": ""}):
                    loaded = faq_module.load_faqs()
            finally:
                os.chdir(original_directory)
        office_faq = next(item for item in loaded if item["id"] == "faq-03")
        self.assertIn("Ashaiman Roundabout", office_faq["answer"])

    def test_configured_faq_endpoint_failure_does_not_fall_back_to_file(self):
        with (
            patch.dict(os.environ, {"FAQS_URL": "https://cms.example.com/api/v1/faqs"}),
            patch.object(faq_module, "urlopen", side_effect=TimeoutError),
            patch.object(faq_module, "_load_local_faqs") as local_loader,
            self.assertRaises(FAQUnavailable),
        ):
            faq_module.load_faqs()
        local_loader.assert_not_called()

    def test_faq_loader_logs_failure_categories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.json"
            with (
                patch.dict(os.environ, {"FAQS_URL": ""}),
                patch.object(faq_module, "DEFAULT_FAQS_PATH", missing_path),
                self.assertLogs("backend.app.faq", level="WARNING") as missing_logs,
                self.assertRaises(FAQUnavailable),
            ):
                faq_module.load_faqs()
            self.assertIn("local FAQ file is missing", "\n".join(missing_logs.output))

            malformed_path = Path(temporary_directory) / "malformed.json"
            malformed_path.write_text("{", encoding="utf-8")
            with (
                patch.dict(os.environ, {"FAQS_URL": ""}),
                patch.object(faq_module, "DEFAULT_FAQS_PATH", malformed_path),
                self.assertLogs("backend.app.faq", level="WARNING") as malformed_logs,
                self.assertRaises(FAQUnavailable),
            ):
                faq_module.load_faqs()
            self.assertIn("malformed JSON", "\n".join(malformed_logs.output))

        with (
            patch.dict(os.environ, {"FAQS_URL": "https://cms.example.com/api/v1/faqs"}),
            patch.object(faq_module, "urlopen", side_effect=HTTPError(
                "https://cms.example.com/api/v1/faqs", 503, "Unavailable", None, None
            )),
            self.assertLogs("backend.app.faq", level="WARNING") as http_logs,
            self.assertRaises(FAQUnavailable),
        ):
            faq_module.load_faqs()
        self.assertIn("HTTP error status=503", "\n".join(http_logs.output))

        response = BytesIO(b"{}")
        response.status = 200
        response.url = "https://cms.example.com/api/v1/faqs"
        with (
            patch.dict(os.environ, {"FAQS_URL": response.url}),
            patch.object(faq_module, "urlopen", return_value=response),
            self.assertLogs("backend.app.faq", level="WARNING") as format_logs,
            self.assertRaises(FAQUnavailable),
        ):
            faq_module.load_faqs()
        self.assertIn("unexpected response format", "\n".join(format_logs.output))

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
