# Property Sales Chatbot

A Python chatbot that checks language, classifies intent, retrieves property
records, and optionally uses Gemini to write replies from those records.

| Path | Purpose |
| --- | --- |
| `training/data/intents.csv` | 154 synthetic visitor messages across seven intents, including Ghanaian English, Pidgin and mild profanity. |
| `training/data/intents_template.csv` | Blank two-column template for collecting more examples. |
| `training/notebooks/train_intent_classifier.ipynb` | Google Colab workflow for training and exporting. |
| `training/train.py` | Shared training script and evaluation logic. |
| `backend/app/moderation/` | Phrase rules for profanity, directed abuse and explicit threats. |
| `backend/app/intent/` | Classifier loading and confidence scores. |
| `backend/app/retrieval/` | Exact filters and TF-IDF ranking over property descriptions. |
| `backend/app/faq.py` | Validated FAQ endpoint loading and question matching. |
| `backend/app/generation/` | Gemini REST integration and template fallback. |
| `artifacts/` | Baseline model and evaluation metadata. |

## Train in Google Colab

1. Upload this ZIP to the root of Google Drive.
2. Extract the ZIP locally and upload or open
   `training/notebooks/train_intent_classifier.ipynb` in Colab.
3. Run the cells in order. The notebook extracts the archive into the Colab
   runtime, installs dependencies, evaluates the model, and writes the exported
   artifact and metadata to `MyDrive/property-sales-chatbot-artifacts/`.
4. For local use after retraining, copy both exported files into `artifacts/`
   and restart the backend. Load only model files you created or trust.

The synthetic holdout is a development check. FAQ and out-of-scope recall are
weak. Use independently collected examples before relying on the confidence
threshold or estimating live accuracy.

## Run the API locally

From the extracted root, use Python 3.11 or newer in a virtual environment.

```sh
python -m pip install -r requirements.txt
python training/train.py
python -m uvicorn backend.app.main:app --reload
```

Open http://127.0.0.1:8000/docs and try `POST /chat` with this JSON body.

```json
{"message":"Show me 3-bedroom houses for sale in Tema under GH₵900,000"}
```

When `LISTINGS_URL` is unset, the default sample catalogue is fictional and
returns listing P101 for this search.
The response includes intent, moderation decision, reply source and structured
listing records. To ask about “the second one,” pass the earlier result IDs in
`previous_listing_ids`. To try “What about East Legon?”, pass the earlier search in
`previous_query`. This starter API leaves conversation state with the caller.

To enable Gemini, set `GEMINI_API_KEY` and `GEMINI_MODEL` in the server environment,
then restart. Gemini receives the current question and matched listing records.
If Gemini is unavailable, the API returns the listing details from its template.
Keep the key on the backend. See [Google's generateContent reference](https://ai.google.dev/api/generate-content).

When `FAQS_URL` is unset, local FAQ testing reads
`backend/app/retrieval/faqs.json` relative to the FAQ module. Start the API with
the variable unset:

```powershell
Remove-Item Env:FAQS_URL -ErrorAction SilentlyContinue
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

If `FAQS_URL` is set, that API is authoritative; a failed request does not fall
back to the local JSON file. Plain HTTP is accepted only for loopback hosts
during local development. The deployed content backend URL must use HTTPS.

The moderation rules cover the example phrases and common variants. Extend
their coverage using reviewed Ghanaian English and Pidgin examples. Retrieval
ranks by word overlap after filtering. Preferences phrased in unrelated words
may need semantic embeddings.


## Live website integration

Set Render environment variables:

```
LISTINGS_URL=https://jaydeladem.github.io/prince-b-realty-consult/listings.json
FAQS_URL=https://<content-backend-host>/api/v1/faqs
CORS_ORIGINS=https://jaydeladem.github.io
GEMINI_API_KEY=<your backend-only key>
GEMINI_MODEL=<your supported Gemini model ID>
```

`FAQS_URL` is the full HTTPS URL of the public FAQ endpoint on the separate
website content backend; it is independent of `LISTINGS_URL`. The endpoint must
return a JSON array of published records with `id`, `category`, `question`,
`answer` and boolean `featured` fields. The chatbot fetches FAQs with an
8-second timeout and validates the response. It returns approved FAQ answers
directly, asks for clarification on uncertain matches, and keeps property
searches available if the FAQ service is unavailable. Set the placeholder host
to the content backend's actual public hostname in the existing chatbot Render
service.

The live catalogue is fetched for each request with an 8-second timeout, a 2 MB
limit and record validation. `LISTINGS_URL` selects live mode; if it is unset,
the local sample catalogue is used. A failed configured live fetch returns HTTP
503 and never falls back to fictional records. Keep the trained model artifact,
existing Render service and start command. No retraining is needed when the
catalogue changes. IDs may include letters, digits, underscores and hyphens.
Null values are unknown; only active listings appear in searches. The browser
sends `previous_query` and `previous_listing_ids`, and stores the returned
`search_query` for future requests. Never place `GEMINI_API_KEY` in website files.
