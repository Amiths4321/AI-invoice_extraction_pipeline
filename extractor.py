"""
Core invoice extraction logic.

Usage:
    from extractor import extract_invoice
    result = extract_invoice(raw_invoice_text)   # -> dict matching schema.Invoice

Design notes:
- Talks to a self-hosted Qwen model over plain HTTP, via the
  OpenAI-compatible /v1/chat/completions request/response shape that
  vLLM, TGI, LM Studio, and Ollama (compatibility mode) all implement.
  No Anthropic, OpenAI, or other third-party SDK is used — just
  `requests` against your own server.
- The reusable instructions/schema/few-shot example live in the *system*
  role message (prompts/system_prompt.txt). The variable, per-invoice
  text is sent as the *user* role message wrapped in <invoice_text> tags.
- Two independent failure modes are retried separately:
    1. Transient network/server errors (timeouts, connection drops, 5xx,
       429) -> retried with exponential backoff.
    2. The model returning text that isn't valid JSON, or JSON that
       doesn't match the schema -> retried with a corrective follow-up
       message telling the model exactly what was wrong.
- If all retries are exhausted, an ExtractionError is raised with enough
  context (last raw response, last error) for logging/alerting/human
  review routing.
"""
import json
import logging
import time
from typing import Optional

import requests
from pydantic import ValidationError

import config
from schema import Invoice, validate_invoice_payload

logger = logging.getLogger("invoice_pipeline.extractor")


class ExtractionError(Exception):
    """Raised when extraction fails after all retries are exhausted."""

    def __init__(self, message: str, last_raw_response: Optional[str] = None):
        super().__init__(message)
        self.last_raw_response = last_raw_response


def _load_system_prompt() -> str:
    with open(config.SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _strip_markdown_fences(text: str) -> str:
    """Defensive cleanup in case the model wraps output in ```json fences
    despite being told not to. Cheap insurance, not a substitute for the
    instruction itself."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _call_model(system_prompt: str, user_content: str) -> str:
    """Single request/retry cycle against the remote Qwen server.
    Returns the raw text of the model's reply."""
    url = f"{config.QWEN_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {config.QWEN_API_KEY}"

    payload = {
        "model": config.QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=config.REQUEST_TIMEOUT_SECONDS
            )

            if response.status_code == 429 or response.status_code >= 500:
                last_error = requests.exceptions.HTTPError(
                    f"{response.status_code}: {response.text[:300]}"
                )
                delay = config.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Transient server error (HTTP %d) on attempt %d/%d. Retrying in %.1fs.",
                    response.status_code, attempt, config.MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                # Non-retryable: bad request, model name mismatch, auth, etc.
                raise ExtractionError(
                    f"Non-retryable HTTP error {response.status_code} from {url}: "
                    f"{response.text[:500]}"
                )

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            delay = config.RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Could not reach Qwen server at %s on attempt %d/%d (%s). Retrying in %.1fs.",
                url, attempt, config.MAX_RETRIES, type(e).__name__, delay,
            )
            time.sleep(delay)

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            # Server responded but not in the expected OpenAI-compatible
            # shape — likely talking to the wrong endpoint/server type.
            raise ExtractionError(
                f"Unexpected response shape from {url}. Is QWEN_BASE_URL really an "
                f"OpenAI-compatible /v1/chat/completions endpoint? Raw error: {e}"
            ) from e

    raise ExtractionError(
        f"Could not get a response from Qwen server at {url} after "
        f"{config.MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def extract_invoice(raw_text: str) -> dict:
    """Extract structured invoice data from raw text (e.g. OCR output).

    Returns a dict matching the Invoice schema (schema.py).
    Raises ExtractionError if extraction fails after all retries.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text must be non-empty")

    system_prompt = _load_system_prompt()
    user_content = f"<invoice_text>\n{raw_text.strip()}\n</invoice_text>"

    last_raw_response: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt in range(1, config.MAX_RETRIES + 1):
        raw_response = _call_model(system_prompt, user_content)
        last_raw_response = raw_response
        cleaned = _strip_markdown_fences(raw_response)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Attempt %d/%d: model output was not valid JSON: %s", attempt, config.MAX_RETRIES, e)
            user_content = (
                f"{user_content}\n\n"
                f"Your previous response could not be parsed as JSON ({e}). "
                f"Respond again with ONLY a single valid JSON object matching the schema. "
                f"No markdown, no commentary."
            )
            continue

        try:
            validated: Invoice = validate_invoice_payload(parsed)
            return validated.model_dump()
        except ValidationError as e:
            last_error = e
            logger.warning("Attempt %d/%d: JSON did not match schema: %s", attempt, config.MAX_RETRIES, e)
            user_content = (
                f"{user_content}\n\n"
                f"Your previous JSON response did not match the required schema "
                f"({e}). Respond again with ONLY a single valid JSON object that "
                f"matches the schema exactly."
            )
            continue

    raise ExtractionError(
        f"Failed to obtain valid, schema-conformant JSON after {config.MAX_RETRIES} attempts: {last_error}",
        last_raw_response=last_raw_response,
    )
