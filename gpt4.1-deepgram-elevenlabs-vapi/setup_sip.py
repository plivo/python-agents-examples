"""Set up Plivo SIP trunk with Vapi for voice agent telephony.

This script automates the entire SIP trunk setup:
1. Creates an IP Access Control List in Plivo with Vapi's signaling IPs
2. Creates an outbound Zentrunk (Vapi -> Plivo -> PSTN)
3. Creates an origination URI pointing to Vapi's SIP endpoint
4. Creates an inbound Zentrunk (PSTN -> Plivo -> Vapi)
5. Assigns the Plivo phone number to the inbound trunk
6. Registers the Plivo trunk as a BYO SIP credential in Vapi (using resolved IPs)
7. Imports the Plivo phone number into Vapi

API reference: https://www.plivo.com/docs/voice-agents/sip-trunking/api/sip-trunking

Prerequisites:
- Plivo account with Zentrunk enabled
- Vapi account with private API key
- .env file configured with PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, VAPI_PRIVATE_KEY

Usage:
    uv run python setup_sip.py
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")

PLIVO_BASE = f"https://api.plivo.com/v1/Account/{PLIVO_AUTH_ID}"
VAPI_BASE = "https://api.vapi.ai"

# Vapi's signaling IPs — must be whitelisted in Plivo's outbound trunk ACL
VAPI_SIGNALING_IPS = [
    "44.229.228.186",
    "44.238.177.138",
]

# Vapi's SIP endpoint for inbound origination
VAPI_SIP_URI = "sip.vapi.ai;transport=udp"


def check_env() -> bool:
    """Verify all required environment variables are set."""
    missing = []
    if not PLIVO_AUTH_ID:
        missing.append("PLIVO_AUTH_ID")
    if not PLIVO_AUTH_TOKEN:
        missing.append("PLIVO_AUTH_TOKEN")
    if not PLIVO_PHONE_NUMBER:
        missing.append("PLIVO_PHONE_NUMBER")
    if not VAPI_PRIVATE_KEY:
        missing.append("VAPI_PRIVATE_KEY")

    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        logger.info("Copy .env.example to .env and fill in the values.")
        return False
    return True


# =============================================================================
# Plivo Zentrunk Setup
# =============================================================================


def plivo_request(
    method: str, endpoint: str, json_data: dict | None = None
) -> requests.Response:
    """Make an authenticated request to the Plivo Zentrunk API."""
    url = f"{PLIVO_BASE}/{endpoint}"
    resp = requests.request(
        method, url, auth=(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN), json=json_data, timeout=30
    )
    return resp


def create_ip_acl() -> str | None:
    """Create an IP Access Control List with Vapi's signaling IPs.

    Plivo API: POST /Zentrunk/IPAccessControlList/
    Uses ip_addresses array (not ip_ranges).

    Returns the ipacl_uuid or None on failure.
    """
    logger.info("Creating IP ACL with Vapi signaling IPs...")

    resp = plivo_request(
        "POST",
        "Zentrunk/IPAccessControlList/",
        {
            "name": "Vapi Signaling IPs",
            "ip_addresses": VAPI_SIGNALING_IPS,
        },
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to create IP ACL: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    ipacl_uuid = data.get("ipacl_uuid", "")
    logger.info(f"Created IP ACL: {ipacl_uuid}")
    return ipacl_uuid


def create_outbound_trunk(ipacl_uuid: str) -> str | None:
    """Create an outbound Zentrunk (Vapi -> Plivo -> PSTN).

    Plivo API: POST /Zentrunk/Trunk/

    Returns the trunk ID or None.
    """
    logger.info("Creating outbound Zentrunk...")

    resp = plivo_request(
        "POST",
        "Zentrunk/Trunk/",
        {
            "name": "Vapi Outbound",
            "trunk_direction": "outbound",
            "trunk_status": "enabled",
            "secure": False,
            "ipacl_uuid": ipacl_uuid,
        },
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to create outbound trunk: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    trunk_id = data.get("trunk_id", "")
    logger.info(f"Created outbound trunk: {trunk_id}")

    # Fetch trunk details to get the SIP domain
    resp = plivo_request("GET", f"Zentrunk/Trunk/{trunk_id}/")
    if resp.status_code == 200:
        trunk_data = resp.json().get("object", resp.json())
        sip_domain = trunk_data.get("trunk_domain", f"{trunk_id}.zt.plivo.com")
    else:
        sip_domain = f"{trunk_id}.zt.plivo.com"

    logger.info(f"Termination SIP domain: {sip_domain}")
    return trunk_id


def create_origination_uri() -> str | None:
    """Create an origination URI pointing to Vapi's SIP endpoint.

    Plivo API: POST /Zentrunk/URI/

    Returns the uri_uuid or None.
    """
    logger.info(f"Creating origination URI for {VAPI_SIP_URI}...")

    resp = plivo_request(
        "POST",
        "Zentrunk/URI/",
        {
            "name": "Vapi Primary",
            "uri": VAPI_SIP_URI,
        },
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to create origination URI: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    uri_uuid = data.get("uri_uuid", "")
    logger.info(f"Created origination URI: {uri_uuid}")
    return uri_uuid


def create_inbound_trunk(primary_uri_uuid: str) -> str | None:
    """Create an inbound Zentrunk (PSTN -> Plivo -> Vapi).

    Plivo API: POST /Zentrunk/Trunk/

    Returns the trunk ID or None.
    """
    logger.info("Creating inbound Zentrunk...")

    resp = plivo_request(
        "POST",
        "Zentrunk/Trunk/",
        {
            "name": "Vapi Inbound",
            "trunk_direction": "inbound",
            "trunk_status": "enabled",
            "primary_uri_uuid": primary_uri_uuid,
        },
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to create inbound trunk: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    trunk_id = data.get("trunk_id", "")
    logger.info(f"Created inbound trunk: {trunk_id}")
    return trunk_id


def assign_phone_to_trunk(inbound_trunk_id: str) -> bool:
    """Assign the Plivo phone number to the inbound Zentrunk trunk.

    Plivo API: POST /Number/{number}/
    """
    logger.info(f"Assigning {PLIVO_PHONE_NUMBER} to inbound trunk {inbound_trunk_id}...")

    number = PLIVO_PHONE_NUMBER.lstrip("+")
    resp = plivo_request(
        "POST",
        f"Number/{number}/",
        {
            "app_id": inbound_trunk_id,
        },
    )

    if resp.status_code not in (200, 201, 202):
        logger.warning(
            f"Failed to assign phone to trunk: {resp.status_code} {resp.text}\n"
            "You may need to assign it manually in Plivo Console:\n"
            f"  Phone Numbers > {PLIVO_PHONE_NUMBER} > Voice Config >"
            f" SIP Trunk > {inbound_trunk_id}"
        )
        return False

    logger.info(f"Assigned {PLIVO_PHONE_NUMBER} to inbound trunk")
    return True


# =============================================================================
# Vapi Setup
# =============================================================================


def vapi_request(
    method: str, endpoint: str, json_data: dict | None = None
) -> requests.Response:
    """Make an authenticated request to the Vapi API."""
    url = f"{VAPI_BASE}/{endpoint}"
    resp = requests.request(
        method,
        url,
        headers={
            "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
            "Content-Type": "application/json",
        },
        json=json_data,
        timeout=30,
    )
    return resp


def register_sip_trunk(trunk_id: str) -> str | None:
    """Register the Plivo SIP trunk as a BYO credential in Vapi.

    Uses the Plivo Termination SIP Domain directly with inboundEnabled=false,
    per Vapi docs: https://docs.vapi.ai/advanced/sip/plivo

    Returns the credential ID or None.
    """
    sip_domain = f"{trunk_id}.zt.plivo.com"
    logger.info(f"Registering SIP trunk {sip_domain} with Vapi...")

    resp = vapi_request(
        "POST",
        "credential",
        {
            "provider": "byo-sip-trunk",
            "name": "Plivo Zentrunk",
            "gateways": [{"ip": sip_domain, "inboundEnabled": False}],
        },
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to register SIP trunk: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    credential_id = data.get("id", "")
    logger.info(f"Registered SIP trunk credential: {credential_id}")
    return credential_id


def import_phone_number(credential_id: str) -> str | None:
    """Import the Plivo phone number into Vapi as a BYO number.

    Returns the Vapi phone number ID or None.
    """
    logger.info(f"Importing {PLIVO_PHONE_NUMBER} into Vapi...")

    resp = vapi_request(
        "POST",
        "phone-number",
        {
            "provider": "byo-phone-number",
            "number": PLIVO_PHONE_NUMBER,
            "credentialId": credential_id,
            "name": "Plivo SIP Number",
            "numberE164CheckEnabled": False,
        },
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to import phone number: {resp.status_code} {resp.text}")
        return None

    data = resp.json()
    phone_id = data.get("id", "")
    logger.info(f"Imported phone number with Vapi ID: {phone_id}")
    return phone_id


def update_env_file(phone_number_id: str) -> None:
    """Update .env file with VAPI_PHONE_NUMBER_ID."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.exists(env_path):
        logger.warning(".env file not found — set VAPI_PHONE_NUMBER_ID manually")
        return

    with open(env_path) as f:
        content = f.read()

    if "VAPI_PHONE_NUMBER_ID=" in content:
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("VAPI_PHONE_NUMBER_ID="):
                lines[i] = f"VAPI_PHONE_NUMBER_ID={phone_number_id}"
                break
        content = "\n".join(lines)
    else:
        content += f"\nVAPI_PHONE_NUMBER_ID={phone_number_id}\n"

    with open(env_path, "w") as f:
        f.write(content)

    logger.info(f"Updated .env with VAPI_PHONE_NUMBER_ID={phone_number_id}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Run the full SIP trunk setup."""
    logger.info("=== Plivo + Vapi SIP Trunk Setup ===\n")

    if not check_env():
        return 1

    # Step 1: Plivo — Create IP ACL with Vapi's signaling IPs
    ipacl_uuid = create_ip_acl()
    if not ipacl_uuid:
        logger.error("Failed at Step 1: IP ACL creation.")
        return 1

    # Step 2: Plivo — Create outbound trunk
    outbound_trunk_id = create_outbound_trunk(ipacl_uuid)
    if not outbound_trunk_id:
        logger.error("Failed at Step 2: Outbound trunk creation.")
        return 1

    # Step 3: Plivo — Create origination URI for Vapi
    uri_uuid = create_origination_uri()
    if not uri_uuid:
        logger.error("Failed at Step 3: Origination URI creation.")
        return 1

    # Step 4: Plivo — Create inbound trunk
    inbound_trunk_id = create_inbound_trunk(uri_uuid)
    if not inbound_trunk_id:
        logger.warning("Inbound trunk creation failed — inbound calls won't work.")
        logger.warning("You can set this up manually later via Plivo Console.")
    else:
        # Step 5: Assign phone number to inbound trunk
        assign_phone_to_trunk(inbound_trunk_id)

    # Step 6: Vapi — Register SIP trunk credential (uses termination SIP domain)
    credential_id = register_sip_trunk(outbound_trunk_id)
    if not credential_id:
        logger.error("Failed at Step 6: Vapi credential registration.")
        return 1

    # Step 7: Vapi — Import phone number
    phone_number_id = import_phone_number(credential_id)
    if not phone_number_id:
        logger.error("Failed at Step 7: Vapi phone number import.")
        return 1

    # Step 8: Update .env
    update_env_file(phone_number_id)

    logger.info("\n=== Setup Complete! ===")
    logger.info(f"Outbound Trunk: {outbound_trunk_id}.zt.plivo.com")
    logger.info(f"Vapi Credential ID: {credential_id}")
    logger.info(f"Vapi Phone Number ID: {phone_number_id}")
    logger.info("\nYou can now start the server:")
    logger.info("  uv run python -m inbound.server   # For inbound calls")
    logger.info("  uv run python -m outbound.server   # For outbound calls")

    return 0


if __name__ == "__main__":
    sys.exit(main())
