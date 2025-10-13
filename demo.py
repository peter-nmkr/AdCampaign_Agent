import json
import os
import sys
import time
from typing import Dict

import requests


def wait_for_server(url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    last_err = None
    while time.time() < deadline:
        try:
            # GET root just to see if server is up (should return the HTML form)
            r = requests.get(url, timeout=3)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"Server not reachable at {url}: {last_err}")


def main() -> None:
    base_url = os.environ.get("DEMO_BASE_URL", "http://127.0.0.1:8014")
    endpoint = f"{base_url}/"

    # Ensure server is up
    print(f"Waiting for server at {endpoint} ...")
    wait_for_server(endpoint)
    print("Server is up. Sending request...")

    # Full form data (logo upload is optional and managed by the app's file system; skipping here)
    payload: Dict[str, object] = {
        "name": "Acme Dental Studio",
        "industry": "healthcare",
        "address": "123 Smile Ave, Tooth City",
        "website": "https://example.com",
        "uvp": "Premium, gentle dental care with same-day crowns",
        "problem": "Patients struggle to find trustworthy, pain-minimizing dental services",
        "audience": "Families and professionals in urban area",
        "goal": "Increase new patient bookings by 20% in 90 days",
        "language": "English",
        "offer": "Free whitening kit with first exam",
        "cta": "Book your appointment",
        "channels": "Website, Instagram, Email list",
        "budget": "$5,000/month",
    }

    headers = {"Content-Type": "application/json"}
    resp = requests.post(endpoint, data=json.dumps(payload), headers=headers, timeout=600)

    print(f"Response status: {resp.status_code}")
    out_path = os.path.join(os.getcwd(), "demo_output.html")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Wrote response HTML to: {out_path}")
    except Exception as e:
        print("Failed to write output:", e, file=sys.stderr)
        print("--- Response body (truncated) ---")
        print(resp.text[:1000])


if __name__ == "__main__":
    main()


