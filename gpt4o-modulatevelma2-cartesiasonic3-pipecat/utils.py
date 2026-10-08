"""Shared utilities: phone number normalization (E.164).

There are no audio helpers here. Pipecat's PlivoFrameSerializer does all the
μ-law decoding, encoding and resampling on the call path, so nothing in this
example converts audio itself (CLAUDE.md "utils.py Requirements": unused code
is not required). Tests that decode or build Plivo audio keep a small codec in
tests/helpers.py. VAD is handled by Pipecat (Silero analyzer on the user
aggregator), and the Modulate STT service lives in inbound/agent.py and
outbound/agent.py.
"""

from __future__ import annotations

import os

import phonenumbers
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# =============================================================================
# Configuration (only constants consumed by utility functions)
# =============================================================================

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# =============================================================================
# Phone Number Utilities
# =============================================================================


def normalize_phone_number(phone: str, default_region: str = DEFAULT_COUNTRY_CODE) -> str:
    """Normalize phone number to E.164 format (digits only, no leading +)."""
    if not phone:
        return ""

    try:
        parsed = phonenumbers.parse(phone, default_region)
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164.lstrip("+")
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Failed to parse phone number: {type(e).__name__}")
        return "".join(c for c in phone if c.isdigit())
