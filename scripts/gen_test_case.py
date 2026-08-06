#!/usr/bin/env python3
"""Assemble a regression test case from a payload and captured outputs.

Invoked by `make gen-tests`. Reads three environment variables:

- ``ENDPOINT``: the endpoint that was run (e.g. ``apply``).
- ``PAYLOAD_FILE``: path to the JSON file holding the input ``payload``.
- ``OUTPUTS``: the JSON string printed by ``tesseract run``.

Writes a complete test case (``endpoint`` + ``payload`` + ``expected_outputs``)
to stdout. The Makefile redirects that into the component's ``test_cases/``
directory, where ``make test`` later verifies it via
``tesseract run <image> test @<file>``. Review the result before committing:
tolerances (``atol``/``rtol``) and any non-deterministic outputs may need
hand-editing.
"""

import json
import os
import sys


def load_payload(path: str) -> dict:
    """Read the payload file and normalize it to a ``{"inputs": ...}`` dict."""
    with open(path) as f:
        raw = json.load(f)
    # Accept either a bare inputs mapping or a full ``{"payload": ...}`` wrapper.
    if "payload" in raw:
        return raw["payload"]
    if "inputs" in raw:
        return raw
    return {"inputs": raw}


def main() -> None:
    """Build the test case from the environment and print it to stdout."""
    endpoint = os.environ.get("ENDPOINT", "apply")
    payload_file = os.environ["PAYLOAD_FILE"]
    outputs_raw = os.environ["OUTPUTS"]

    try:
        expected_outputs = json.loads(outputs_raw)
    except json.JSONDecodeError:
        print(
            "ERROR: could not parse tesseract output as JSON. Raw output:\n"
            f"{outputs_raw}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    test_case = {
        "endpoint": endpoint,
        "payload": load_payload(payload_file),
        "expected_outputs": expected_outputs,
    }
    print(json.dumps(test_case, indent=4))


if __name__ == "__main__":
    main()
