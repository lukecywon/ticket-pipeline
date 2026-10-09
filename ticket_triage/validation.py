"""Offline consistency checks for the complete artifact set."""

import re
from datetime import datetime
from pathlib import Path

from .data import DataError, OutputError, load_inputs, loads_json, parse_object, parse_reply, preprocess, read_json, stable_hash, validate_prediction, validate_reply
from .evaluation import evaluate, route_prediction
from .prompts import classification_request, reply_request


JSON_ARTIFACTS = (
    "tickets.json", "label_schema.json", "preprocessed_tickets.json",
    "routing_decisions.json", "triage_results.json", "prediction_comparison.json",
    "evaluation_report.json", "parsed_predictions.json", "reply_predictions.json",
    "errors.json", "run_metadata.json", "confusion_summary.json",
)


def _require(condition, message):
    if not condition:
        raise DataError(message)


def _same(actual, expected, message):
    _require(stable_hash(actual) == stable_hash(expected), message)


def _records(records, expected_ids, name):
    _require(isinstance(records, list), f"{name} must be an array.")
    _require(all(isinstance(record, dict) and isinstance(record.get("ticket_id"), str) for record in records), f"{name}: invalid ticket record.")
    _require([record["ticket_id"] for record in records] == expected_ids, f"{name}: missing, duplicate, unknown, or out-of-order ticket IDs.")
    return {record["ticket_id"]: record for record in records}


def _timestamp(value):
    _require(isinstance(value, str), "Timestamp must be a string.")
    try:
        date = datetime.fromisoformat(value)
        _require(date.utcoffset() is not None, "Timestamp must include a timezone.")
    except ValueError as exc:
        raise DataError("Invalid ISO-8601 timestamp.") from exc


def _artifact_path(root, relative):
    _require(isinstance(relative, str) and relative.startswith("raw_outputs/"), "Raw output reference must be a relative raw_outputs path.")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), "Raw output path escapes output directory.")
    return path


def validate_artifacts(tickets_path: Path, schema_path: Path, output_dir: Path):
    """Return a concise error list; never make model requests or require credentials."""
    try:
        return _validate_all(tickets_path, schema_path, Path(output_dir))
    except (DataError, OutputError, KeyError, TypeError, ValueError, IndexError, AttributeError, OSError) as exc:
        return [str(exc)]


def _validate_all(tickets_path, schema_path, root):
    tickets, schema = load_inputs(tickets_path, schema_path)
    missing = [name for name in (*JSON_ARTIFACTS, "llm_calls.jsonl") if not (root / name).is_file()]
    if missing:
        return [f"Missing artifact: {name}" for name in missing]
    artifacts = {name: read_json(root / name) for name in JSON_ARTIFACTS}
    _same(artifacts["tickets.json"], tickets, "Input ticket snapshot does not match supplied tickets.")
    _same(artifacts["label_schema.json"], schema, "Input label snapshot does not match supplied schema.")
    _same(artifacts["preprocessed_tickets.json"], preprocess(tickets), "Preprocessing does not preserve and normalize input as documented.")
    metadata = artifacts["run_metadata.json"]
    if metadata["status"] != "complete":
        failed_replies = [f"{record['ticket_id']} ({record['status']})" for record in artifacts["reply_predictions.json"] if record["status"] != "valid"]
        detail = f"; reply generation failed for {', '.join(failed_replies)}" if failed_replies else ""
        raise DataError(f"Run is incomplete{detail}; check errors.json and reply_predictions.json.")
    _require(metadata["tickets_hash"] == stable_hash(tickets) and metadata["schema_hash"] == stable_hash(schema), "Run metadata input hashes do not match.")
    _timestamp(metadata["started_at"])
    _timestamp(metadata["finished_at"])
    ids = [ticket["ticket_id"] for ticket in tickets]
    parsed = _records(artifacts["parsed_predictions.json"], ids, "parsed_predictions.json")
    routes = _records(artifacts["routing_decisions.json"], ids, "routing_decisions.json")
    results = _records(artifacts["triage_results.json"], ids, "triage_results.json")
    _records(artifacts["prediction_comparison.json"], ids, "prediction_comparison.json")
    for ticket_id in ids:
        record = parsed[ticket_id]
        _require(type(record["recovered"]) is bool, f"{ticket_id}: recovered must be boolean.")
        if record["status"] == "valid":
            _require(record["prediction"] is not None and record["error"] is None, f"{ticket_id}: valid prediction must not contain an error.")
            validate_prediction(record["prediction"], schema)
        else:
            _require(record["status"] in ("parse_error", "validation_error", "provider_error"), f"{ticket_id}: unknown prediction status.")
            _require(record["prediction"] is None and isinstance(record["error"], str) and record["error"].strip(), f"{ticket_id}: classification failure needs null prediction and an error.")
        _same(routes[ticket_id], route_prediction(ticket_id, record), f"{ticket_id}: routing decision violates deterministic rules.")
    auto_ids = [ticket_id for ticket_id in ids if routes[ticket_id]["route"] == "auto_triage"]
    replies = _records(artifacts["reply_predictions.json"], auto_ids, "reply_predictions.json")
    for ticket_id in auto_ids:
        record = replies[ticket_id]
        _require(record["status"] == "valid" and record["error"] is None, f"{ticket_id}: auto-triage reply generation failed.")
        _require(type(record["recovered"]) is bool, f"{ticket_id}: reply recovered must be boolean.")
        _, reply = validate_reply({"sentences": record["sentences"]})
        _same(record["customer_reply"], reply, f"{ticket_id}: saved reply differs from validated sentences.")
    for ticket_id in ids:
        prediction = parsed[ticket_id]["prediction"]
        route = routes[ticket_id]
        expected_result = {
            "ticket_id": ticket_id,
            "predicted_category": prediction["category"] if prediction else None,
            "predicted_urgency": prediction["urgency"] if prediction else None,
            "confidence": route["confidence"], "route": route["route"],
            "customer_reply": replies[ticket_id]["customer_reply"] if ticket_id in replies else None,
            "internal_note": None if ticket_id in replies else f"Escalated for human review. {route['routing_reason']}",
        }
        _same(results[ticket_id], expected_result, f"{ticket_id}: triage result labels, confidence, reply, or internal note is inconsistent.")

    calls = []
    try:
        lines = (root / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines, 1):
            _require(bool(line.strip()), f"llm_calls.jsonl line {index} is blank.")
            calls.append(loads_json(line))
    except (OSError, ValueError) as exc:
        raise DataError(f"Invalid llm_calls.jsonl: {exc}") from exc
    processed = {ticket["ticket_id"]: ticket for ticket in preprocess(tickets)}
    raw_by_path = {}
    classification_calls = {ticket_id: [] for ticket_id in ids}
    reply_calls = {ticket_id: [] for ticket_id in auto_ids}
    for call in calls:
        _require(isinstance(call, dict), "Each call log line must be a JSON object.")
        required = {"stage", "ticket_id", "timestamp", "provider", "model", "prompt_hash", "output_artifact"}
        _require(required <= call.keys(), "Call log is missing a required field.")
        ticket_id, stage = call["ticket_id"], call["stage"]
        _require(ticket_id in parsed, "Call log references an unknown ticket.")
        _require(stage in ("classification", "reply_generation"), "Call log has an invalid stage.")
        _timestamp(call["timestamp"])
        _require(call["provider"] == "openrouter", "Unexpected call provider.")
        _require(call["model"] == metadata["classification_model" if stage == "classification" else "reply_model"], "Logged model differs from run configuration.")
        _require(isinstance(call["prompt_hash"], str) and re.fullmatch(r"[0-9a-f]{64}", call["prompt_hash"]), "Invalid prompt hash.")
        relative = call["output_artifact"]
        _require(relative not in raw_by_path, "Multiple calls reference the same raw output.")
        _require(relative.startswith(f"raw_outputs/{metadata['run_id']}/"), "Call references another run's artifact.")
        raw = read_json(_artifact_path(root, relative))
        raw_by_path[relative] = raw
        _require(raw["ticket_id"] == ticket_id and raw["stage"] == stage and raw["timestamp"] == call["timestamp"], "Call log and raw artifact disagree.")
        _require(call["status"] == ("error" if raw["error"] else "success"), "Call log status disagrees with raw output.")
        request = raw["request"]
        _require(call["prompt_hash"] == stable_hash({"messages": request["messages"], "response_format": request["response_format"]}), "Prompt hash does not match saved request.")
        if stage == "classification":
            _same(request, classification_request(processed[ticket_id], schema, call["model"]), "Classification request does not match ticket and schema.")
            classification_calls[ticket_id].append(relative)
        else:
            _require(ticket_id in replies, "Human-review ticket received a reply-generation call.")
            _require(bool(classification_calls[ticket_id]), "Reply call occurred before classification.")
            options = [reply_request(processed[ticket_id], parsed[ticket_id]["prediction"], call["model"], retry=retry) for retry in (False, True)]
            _require(any(stable_hash(request) == stable_hash(option) for option in options), "Reply request does not match ticket and prediction.")
            reply_calls[ticket_id].append(relative)

    for ticket_id in ids:
        _require(bool(classification_calls[ticket_id]), f"{ticket_id}: no classification call was logged.")
        record = parsed[ticket_id]
        relative = record["output_artifact"]
        _require(relative == classification_calls[ticket_id][-1], f"{ticket_id}: classification does not reference its final call.")
        raw = raw_by_path[relative]
        if record["status"] == "provider_error":
            _require(bool(raw["error"]), f"{ticket_id}: provider failure lacks an error artifact.")
            _require(not record["recovered"] and record["recovery_details"] is None, f"{ticket_id}: provider failure cannot be a recovered output.")
            continue
        _require(not raw["error"], f"{ticket_id}: parsed output references a failed provider call.")
        body = loads_json(raw["response_text"])
        content = body["choices"][0]["message"]["content"]
        recovered, details = False, None
        try:
            value, recovered, details = parse_object(content)
            validate_prediction(value, schema)
        except OutputError as exc:
            _require(record["status"] == exc.kind and record["error"] == str(exc), f"{ticket_id}: saved classification error differs from raw response.")
        else:
            _require(record["status"] == "valid", f"{ticket_id}: successful raw classification saved as failure.")
            _same(value, record["prediction"], f"{ticket_id}: prediction differs from raw response.")
        _require(record["recovered"] == recovered and record["recovery_details"] == details, f"{ticket_id}: recovery metadata is incorrect.")
    for ticket_id in auto_ids:
        _require(bool(reply_calls[ticket_id]), f"{ticket_id}: no reply call logged.")
        record = replies[ticket_id]
        _require(record["output_artifact"] == reply_calls[ticket_id][-1], f"{ticket_id}: reply does not reference its final call.")
        raw = raw_by_path[record["output_artifact"]]
        _require(not raw["error"], f"{ticket_id}: reply references failed provider call.")
        body = loads_json(raw["response_text"])
        sentences, reply, recovered, details = parse_reply(body["choices"][0]["message"]["content"])
        _require(record["recovered"] == recovered, f"{ticket_id}: reply recovery metadata is incorrect.")
        if "recovery_details" in record:
            _require(record["recovery_details"] == details, f"{ticket_id}: reply recovery details are incorrect.")
        _same(record["sentences"], sentences, f"{ticket_id}: reply differs from raw output.")
        _same(record["customer_reply"], reply, f"{ticket_id}: reply text differs from raw output.")

    errors = artifacts["errors.json"]
    _require(isinstance(errors, list), "errors.json must be an array.")
    for error in errors:
        _require(isinstance(error, dict) and error["ticket_id"] in parsed, "Error log references an unknown ticket.")
        _require(error["kind"] in ("provider_error", "parse_error", "validation_error", "recovered_output"), "Unknown error-log kind.")
        _require(error["output_artifact"] in raw_by_path, "Error log references an unlogged call.")
        raw = raw_by_path[error["output_artifact"]]
        _require(error["stage"] == raw["stage"] and error["ticket_id"] == raw["ticket_id"], "Error log and raw output disagree.")
    _require(sum(error["kind"] == "provider_error" for error in errors) == sum(bool(raw["error"]) for raw in raw_by_path.values()), "Provider error log is inconsistent with raw calls.")
    report, comparisons, confusion = evaluate(tickets, list(parsed.values()), list(routes.values()), list(replies.values()), errors, schema)
    _same(artifacts["evaluation_report.json"], report, "Evaluation metrics do not match recomputed metrics.")
    _same(artifacts["prediction_comparison.json"], comparisons, "Per-ticket comparisons do not match recomputed comparisons.")
    _same(artifacts["confusion_summary.json"], confusion, "Confusion summary does not match predictions.")
    return []
