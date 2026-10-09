"""Read-only SHA-256 check; no dependencies, simulations or patient data needed."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append((relative, "missing or invalid path"))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append((relative, "SHA-256 mismatch"))
    if failures:
        for relative, reason in failures:
            print(f"FAIL: {relative}: {reason}")
        return 1
    print(f"PASS: {len(manifest)} archived files match SHA256SUMS.json.")
    print("No experiments, downloads, model fits or file writes were performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
