"""Validate saved pipeline artifacts without credentials or network requests."""

import argparse
import sys
from pathlib import Path

from ticket_triage.validation import validate_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", type=Path, default=Path("tickets.json"))
    parser.add_argument("--schema", type=Path, default=Path("label_schema.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    errors = validate_artifacts(args.tickets, args.schema, args.output_dir)
    if errors:
        for error in errors:
            print(f"Validation error: {error}", file=sys.stderr)
        return 1
    print(f"All artifacts valid: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
