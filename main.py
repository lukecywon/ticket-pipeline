"""Run real OpenRouter inference against local JSON tickets."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ticket_triage.data import DataError, load_inputs
from ticket_triage.pipeline import run_pipeline
from ticket_triage.provider import ArtifactStore, OpenRouterClient
from ticket_triage.validation import validate_artifacts


DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--tickets", type=Path, default=Path("tickets.json"))
    command.add_argument("--schema", type=Path, default=Path("label_schema.json"))
    command.add_argument("--output-dir", type=Path, default=Path("outputs"))
    command.add_argument("--classification-model", help="Override CLASSIFICATION_MODEL.")
    command.add_argument("--reply-model", help="Override REPLY_MODEL.")
    command.add_argument("--overwrite", action="store_true", help="Replace known run artifacts in an existing output directory.")
    return command


def resolve_settings(args):
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise DataError("OPENROUTER_API_KEY is blank. Set it in .env or your environment before running the pipeline.")
    classification_model = args.classification_model if args.classification_model is not None else os.getenv("CLASSIFICATION_MODEL") or DEFAULT_MODEL
    reply_model = args.reply_model if args.reply_model is not None else os.getenv("REPLY_MODEL") or DEFAULT_MODEL
    if not classification_model.strip() or not reply_model.strip():
        raise DataError("Model IDs must not be blank.")
    return api_key, classification_model.strip(), reply_model.strip()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        # Complete preflight before creating or replacing any output artifacts.
        api_key, classification_model, reply_model = resolve_settings(args)
        tickets, schema = load_inputs(args.tickets, args.schema)
        store = ArtifactStore(args.output_dir, overwrite=args.overwrite)
        provider = OpenRouterClient(api_key, store)
        try:
            successful, report = run_pipeline(tickets, schema, store, provider, classification_model, reply_model)
        finally:
            provider.close()
        errors = validate_artifacts(args.tickets, args.schema, args.output_dir)
        print(f"Processed {report['total_tickets']} tickets; human review: {report['human_review_count']}; classification parse/validation failures: {report['parse_validation_failures']}; reply generation failures: {report['reply_generation_failures']}.")
        print(f"Artifacts: {args.output_dir.resolve()}")
        if not successful or errors:
            for error in errors:
                print(f"Validation error: {error}", file=sys.stderr)
            return 1
        print("Artifact validation passed.")
        return 0
    except (DataError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
