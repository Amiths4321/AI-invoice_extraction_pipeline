"""
Pydantic schema for validating extracted invoice JSON.

The LLM is instructed to follow this exact shape, but pydantic gives us a
hard guarantee before the data is allowed downstream into a database,
accounting system, etc. Anything that doesn't match raises a validation
error the caller can catch and handle (retry, route to a human review
queue, log and alert).
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    Description: Optional[str] = None
    Quantity: Optional[float] = None
    Unit_Price: Optional[float] = None
    Total_Price: Optional[float] = None


class Invoice(BaseModel):
    Vendor_Name: Optional[str] = None
    Invoice_Number: Optional[str] = None
    Invoice_Date: Optional[str] = None
    Due_Date: Optional[str] = None
    Line_Items: List[LineItem] = Field(default_factory=list)
    Subtotal: Optional[float] = None
    Tax_Amount: Optional[float] = None
    Total_Amount: Optional[float] = None
    Currency: Optional[str] = None

    @field_validator("Invoice_Date", "Due_Date")
    @classmethod
    def validate_date_format(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        import re
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"Date '{value}' is not in YYYY-MM-DD format")
        return value

    @field_validator("Currency")
    @classmethod
    def validate_currency_code(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not (value.isalpha() and len(value) == 3):
            raise ValueError(f"Currency '{value}' does not look like a 3-letter ISO code")
        return value.upper()


def validate_invoice_payload(data: dict) -> Invoice:
    """Validate a raw dict (already json.loads'd) against the Invoice schema.

    Raises pydantic.ValidationError if the shape or types are wrong.
    """
    return Invoice.model_validate(data)
