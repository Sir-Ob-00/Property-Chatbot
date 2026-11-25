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

The default catalogue is fictional and returns listing P101 for this search.
The response includes intent, moderation decision, reply source and structured
listing records. To ask about “the second one,” pass the earlier result IDs in
`previous_listing_ids`. To try “What about East Legon?”, pass the earlier search in
`previous_query`. This starter API leaves conversation state with the caller.

To enable Gemini, set `GEMINI_API_KEY` and `GEMINI_MODEL` in the server environment,
then restart. Gemini receives the current question and matched listing records.
If Gemini is unavailable, the API returns the listing details from its template.
Keep the key on the backend. See [Google's generateContent reference](https://ai.google.dev/api/generate-content).

Set `LISTINGS_FILE` to an absolute path for a JSON catalogue with the same fields
as `backend/app/retrieval/sample_listings.json`. Use lowercase property types and
statuses, prices in Ghana cedis, unique IDs, and null for bedrooms on land.
The starter uses P-number references such as P101. The site’s live API/database
adapter, chat widget and viewing submission are integration steps. The viewing
and contact routes direct visitors to the website’s contact page.

The moderation rules cover the example phrases and common variants. Extend
their coverage using reviewed Ghanaian English and Pidgin examples. Retrieval
ranks by word overlap after filtering. Preferences phrased in unrelated words
may need semantic embeddings.
