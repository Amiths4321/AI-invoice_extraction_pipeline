# Invoice Extraction Pipeline (Qwen, self-hosted)

Production-ready pipeline that converts raw invoice text (including OCR
output) into validated, structured JSON using a **Qwen model running on
your own remote server**. No Anthropic, OpenAI, or other third-party API
is used — the pipeline talks to your server over plain HTTP via the
OpenAI-compatible `/v1/chat/completions` request/response shape, which is
what vLLM, TGI, LM Studio, and Ollama's compatibility mode all expose.

## Project structure

```
invoice_extraction_pipeline/
├── README.md
├── requirements.txt
├── .env.example                  # copy to .env and fill in your server details
├── config.py                     # env-driven configuration
├── schema.py                     # pydantic schema + validation
├── extractor.py                  # core extraction logic (HTTP call, retries, parsing)
├── cli.py                        # command-line entry point (single + batch)
├── prompts/
│   └── system_prompt.txt         # the reusable extraction instructions
├── sample_invoices/
│   ├── sample_invoice_1.txt
│   └── sample_invoice_2.txt
└── tests/
    ├── test_schema.py
    └── test_extractor.py
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```
QWEN_BASE_URL=http://your-remote-server:8000   # no trailing slash, no /v1/... suffix
QWEN_MODEL=qwen2.5-instruct                     # must match your server's exact model name
QWEN_API_KEY=                                   # leave blank if your server has no auth
```

**`QWEN_MODEL` is the one setting most likely to need changing** — it must
exactly match whatever name your server was launched with:

| Server type | Where the model name comes from |
|---|---|
| vLLM | the `--served-model-name` flag (or the HF repo id if you didn't set one) |
| Ollama | the pulled tag, e.g. `qwen2.5:32b-instruct` |
| TGI | usually the HF repo id it was launched with |

If you're not sure what your server exposes, run:
```bash
curl http://your-remote-server:PORT/v1/models
```
which on an OpenAI-compatible server lists the available model name(s).

## Usage

Single invoice, printed to stdout:

```bash
python cli.py extract sample_invoices/sample_invoice_1.txt
```

Single invoice, written to a file:

```bash
python cli.py extract sample_invoices/sample_invoice_1.txt -o result.json
```

Batch — every `.txt` file in a directory, processed concurrently:

```bash
python cli.py batch sample_invoices/ -o output_dir/
```

Programmatic use:

```python
from extractor import extract_invoice

with open("invoice.txt") as f:
    raw_text = f.read()

result = extract_invoice(raw_text)   # dict matching the schema below
```

## Schema

```json
{
  "Vendor_Name": "string | null",
  "Invoice_Number": "string | null",
  "Invoice_Date": "YYYY-MM-DD | null",
  "Due_Date": "YYYY-MM-DD | null",
  "Line_Items": [
    {
      "Description": "string | null",
      "Quantity": "number | null",
      "Unit_Price": "number | null",
      "Total_Price": "number | null"
    }
  ],
  "Subtotal": "number | null",
  "Tax_Amount": "number | null",
  "Total_Amount": "number | null",
  "Currency": "string | null"
}
```

`extract_invoice()` returns a plain `dict` that has already passed pydantic
validation (`schema.Invoice`) — dates are guaranteed `YYYY-MM-DD`, currency
is a 3-letter uppercase code, and all monetary fields are floats or `null`.

## How it works

- **`prompts/system_prompt.txt`** holds the role definition, formatting
  rules, target schema, and one few-shot example — sent as the `system`
  role message on every request.
- **The raw invoice text** is sent as the `user` role message, wrapped in
  `<invoice_text>...</invoice_text>` tags, so the model can clearly tell
  static instructions apart from the document being processed.
- **`extractor.py`** posts directly to `{QWEN_BASE_URL}/v1/chat/completions`
  using `requests`, and handles two distinct failure modes separately:
  1. *Transient errors* (connection refused, timeout, HTTP 5xx, HTTP 429)
     — retried with exponential backoff, since the server is likely just
     busy or restarting.
  2. *Malformed model output* (invalid JSON, or JSON that fails schema
     validation) — retried with a corrective follow-up message that tells
     the model exactly what was wrong. Smaller self-hosted Qwen models are
     more likely to occasionally drift from strict JSON than a large
     hosted model, so this retry path matters more here than it would
     against a frontier API.
- If all retries are exhausted, `ExtractionError` is raised carrying the
  last raw response, so a calling system can log it, alert, or route the
  document to human review instead of silently failing.

## Testing

Tests mock `requests.post`, so they run fully offline — no remote server,
no network access required:

```bash
pytest tests/ -v
```

## Production hardening notes

- **Smaller/quantized Qwen models are more likely to violate the
  "JSON only" instruction occasionally.** If you see frequent retries in
  the logs, consider: a larger Qwen variant, a lower `temperature`
  (already defaulted to 0), or — if your server supports it — a
  constrained-decoding / guided-JSON feature (vLLM, TGI, and Outlines-based
  servers all have some form of this) so the server enforces the schema
  at generation time instead of relying on retries after the fact.
- **`REQUEST_TIMEOUT_SECONDS`** defaults to 120s because self-hosted
  inference on modest hardware can be considerably slower than a hosted
  API. Tune this to match your actual hardware/model size.
- **Observability**: the `logging` calls in `extractor.py` and `cli.py`
  are wired to a logger, not `print` — point them at your existing log
  aggregation rather than reading stdout in production.
- **Human-in-the-loop**: `ExtractionError.last_raw_response` is preserved
  specifically so a failed extraction can be routed to a review queue
  with full context, instead of being discarded.
- **Network**: since the server is remote, make sure `QWEN_BASE_URL` is
  reachable from wherever this pipeline actually runs (firewall rules,
  VPN, etc.) — connection errors surface as retries, then as a clear
  `ExtractionError` rather than a silent hang.
- **PII/financial data**: invoice text often contains names, addresses,
  and account-adjacent details. Make sure storage of raw text and
  extracted JSON complies with your data retention and access-control
  policies, same as you would for any hosted API.
