import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extractor
from extractor import extract_invoice, ExtractionError

VALID_JSON = json.dumps({
    "Vendor_Name": "Acme Corp",
    "Invoice_Number": "INV-2026-001",
    "Invoice_Date": "2026-01-15",
    "Due_Date": "2026-02-15",
    "Line_Items": [
        {"Description": "Laptops", "Quantity": 2, "Unit_Price": 1200.0, "Total_Price": 2400.0}
    ],
    "Subtotal": 2400.0,
    "Tax_Amount": 240.0,
    "Total_Amount": 2640.0,
    "Currency": "USD",
})


def _mock_response(content: str, status_code: int = 200):
    """Build a fake requests.Response-like object carrying an
    OpenAI-compatible chat completion body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = content
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return resp


def test_extract_invoice_happy_path():
    with patch.object(extractor, "_call_model", return_value=VALID_JSON) as mock_call:
        result = extract_invoice("some raw invoice text")
    assert result["Vendor_Name"] == "Acme Corp"
    assert result["Total_Amount"] == 2640.0
    mock_call.assert_called_once()


def test_extract_invoice_strips_markdown_fences():
    fenced = f"```json\n{VALID_JSON}\n```"
    with patch.object(extractor, "_call_model", return_value=fenced):
        result = extract_invoice("some raw invoice text")
    assert result["Vendor_Name"] == "Acme Corp"


def test_extract_invoice_retries_on_invalid_json_then_succeeds():
    # First call returns garbage, second call returns valid JSON.
    with patch.object(extractor, "_call_model", side_effect=["not json at all", VALID_JSON]) as mock_call:
        result = extract_invoice("some raw invoice text")
    assert result["Vendor_Name"] == "Acme Corp"
    assert mock_call.call_count == 2


def test_extract_invoice_raises_after_exhausting_retries():
    extractor.config.MAX_RETRIES = 2
    with patch.object(extractor, "_call_model", return_value="still not json"):
        try:
            extract_invoice("some raw invoice text")
            assert False, "expected ExtractionError"
        except ExtractionError as e:
            assert e.last_raw_response == "still not json"
    extractor.config.MAX_RETRIES = 3  # restore default for other tests


def test_extract_invoice_rejects_empty_input():
    try:
        extract_invoice("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_call_model_parses_openai_compatible_response():
    """End-to-end check of the HTTP layer itself (requests.post is mocked,
    nothing leaves the test process)."""
    fake_response = _mock_response(VALID_JSON)
    with patch("extractor.requests.post", return_value=fake_response) as mock_post:
        text = extractor._call_model("system prompt", "<invoice_text>foo</invoice_text>")
    assert text == VALID_JSON
    called_url = mock_post.call_args.args[0]
    assert called_url.endswith("/v1/chat/completions")


def test_call_model_retries_on_500_then_succeeds():
    bad = _mock_response("server error", status_code=500)
    good = _mock_response(VALID_JSON, status_code=200)
    with patch("extractor.requests.post", side_effect=[bad, good]):
        with patch("extractor.time.sleep"):  # skip the real backoff delay in tests
            text = extractor._call_model("system prompt", "<invoice_text>foo</invoice_text>")
    assert text == VALID_JSON


def test_call_model_raises_on_non_retryable_4xx():
    bad = _mock_response("bad request", status_code=400)
    with patch("extractor.requests.post", return_value=bad):
        try:
            extractor._call_model("system prompt", "<invoice_text>foo</invoice_text>")
            assert False, "expected ExtractionError"
        except ExtractionError:
            pass
