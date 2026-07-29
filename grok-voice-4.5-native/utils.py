"""Shared utilities.

Phone number normalization for the Plivo API. This example forwards Plivo's
mu-law 8 kHz audio straight to the xAI realtime API and back, so it needs no
audio conversion or voice-activity-detection utilities.
"""

from __future__ import annotations

import os

import phonenumbers
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")


def normalize_phone_number(phone: str, default_region: str = DEFAULT_COUNTRY_CODE) -> str:
    """Normalize phone number to E.164 format (digits only, no leading +)."""
    if not phone:
        return ""

    try:
        parsed = phonenumbers.parse(phone, default_region)
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164.lstrip("+")
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Failed to parse phone number '{phone}': {e}")
        return "".join(c for c in phone if c.isdigit())
