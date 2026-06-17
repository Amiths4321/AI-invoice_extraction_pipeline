import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import validate_invoice_payload


def test_valid_payload_passes():
    payload = {
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
    }
    invoice = validate_invoice_payload(payload)
    assert invoice.Vendor_Name == "Acme Corp"
    assert len(invoice.Line_Items) == 1
    assert invoice.Line_Items[0].Total_Price == 2400.0


def test_null_fields_allowed():
    payload = {
        "Vendor_Name": "Acme Corp",
        "Invoice_Number": None,
        "Invoice_Date": None,
        "Due_Date": None,
        "Line_Items": [],
        "Subtotal": None,
        "Tax_Amount": None,
        "Total_Amount": None,
        "Currency": None,
    }
    invoice = validate_invoice_payload(payload)
    assert invoice.Invoice_Number is None
    assert invoice.Line_Items == []


def test_bad_date_format_rejected():
    payload = {
        "Vendor_Name": "Acme Corp",
        "Invoice_Number": "INV-1",
        "Invoice_Date": "01/15/2026",  # wrong format
        "Due_Date": None,
        "Line_Items": [],
        "Subtotal": None,
        "Tax_Amount": None,
        "Total_Amount": None,
        "Currency": "USD",
    }
    with pytest.raises(ValidationError):
        validate_invoice_payload(payload)


def test_bad_currency_code_rejected():
    payload = {
        "Vendor_Name": "Acme Corp",
        "Invoice_Number": "INV-1",
        "Invoice_Date": None,
        "Due_Date": None,
        "Line_Items": [],
        "Subtotal": None,
        "Tax_Amount": None,
        "Total_Amount": None,
        "Currency": "US Dollars",  # not a 3-letter code
    }
    with pytest.raises(ValidationError):
        validate_invoice_payload(payload)


def test_currency_is_uppercased():
    payload = {
        "Vendor_Name": "Acme Corp",
        "Invoice_Number": "INV-1",
        "Invoice_Date": None,
        "Due_Date": None,
        "Line_Items": [],
        "Subtotal": None,
        "Tax_Amount": None,
        "Total_Amount": None,
        "Currency": "usd",
    }
    invoice = validate_invoice_payload(payload)
    assert invoice.Currency == "USD"
