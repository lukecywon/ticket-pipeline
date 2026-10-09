"""Input validation, strict JSON parsing, and model output validation."""

import hashlib
import json
import math
import re
from pathlib import Path

from jsonschema import Draft202012Validator


class DataError(ValueError):
    """Invalid input or artifact."""


class OutputError(ValueError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds the finite floating-point range.")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _check_nesting(text: str):
    # Bound nesting independently of Python version/parser recursion behavior.
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > 64:
                raise OutputError("parse_error", "JSON exceeds the maximum nesting depth of 64.")
        elif character in "]}":
            depth -= 1


def loads_json(text: str):
    _check_nesting(text)
    try:
        return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float, object_pairs_hook=_unique_object)
    except RecursionError as exc:
        raise ValueError("JSON nesting limit exceeded.") from exc


def read_json(path: Path):
    try:
        return loads_json(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise DataError(f"Cannot read valid JSON from {path}: {exc}") from exc


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_hash(data) -> str:
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_schema(schema):
    if not isinstance(schema, dict):
        raise DataError("Label schema must be a JSON object.")
    for key in ("categories", "urgency_levels"):
        labels = schema.get(key)
        if not isinstance(labels, list) or not labels:
            raise DataError(f"{key} must be a nonempty list.")
        if any(not isinstance(label, str) or not label.strip() or label != label.strip() for label in labels):
            raise DataError(f"{key} must contain nonempty, trimmed strings.")
        if len(set(labels)) != len(labels):
            raise DataError(f"{key} contains duplicate labels.")


def validate_tickets(tickets, schema):
    if not isinstance(tickets, list):
        raise DataError("Tickets must be a JSON array.")
    seen = set()
    for index, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            raise DataError(f"Ticket {index} must be an object.")
        for field in ("ticket_id", "customer_message"):
            if not isinstance(ticket.get(field), str) or not ticket[field].strip():
                raise DataError(f"Ticket {index}: {field} must be a nonempty string.")
        if ticket["ticket_id"] in seen:
            raise DataError(f"Duplicate ticket_id: {ticket['ticket_id']}")
        seen.add(ticket["ticket_id"])
        for field, labels in (("expected_category", "categories"), ("expected_urgency", "urgency_levels")):
            if ticket.get(field) not in schema[labels]:
                raise DataError(f"Ticket {ticket['ticket_id']}: {field} must belong to {labels}.")


def load_inputs(tickets_path: Path, schema_path: Path):
    schema = read_json(schema_path)
    validate_schema(schema)
    tickets = read_json(tickets_path)
    validate_tickets(tickets, schema)
    return tickets, schema


def preprocess(tickets):
    return [dict(ticket, normalized_message=ticket["customer_message"].replace("\r\n", "\n").replace("\r", "\n").strip()) for ticket in tickets]


def classification_schema(labels):
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": labels["categories"]},
            "urgency": {"type": "string", "enum": labels["urgency_levels"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning_summary": {"type": "string", "minLength": 1, "maxLength": 600},
            "needs_human_review": {"type": "boolean"},
        },
        "required": ["category", "urgency", "confidence", "reasoning_summary", "needs_human_review"],
        "additionalProperties": False,
    }


REPLY_SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "minItems": 2, "maxItems": 4, "items": {"type": "string", "minLength": 1}}},
    "required": ["sentences"],
    "additionalProperties": False,
}


def parse_object(text: str, *, recover: bool = True):
    """Recover only one unambiguous top-level object; never repair field values."""
    try:
        parsed = loads_json(text)
    except OutputError:
        raise
    except (ValueError, TypeError) as original:
        if not recover or not isinstance(text, str):
            raise OutputError("parse_error", f"Invalid JSON: {original}") from original
        decoder = json.JSONDecoder(parse_constant=_reject_constant, parse_float=_finite_float, object_pairs_hook=_unique_object)
        candidates = []
        offset = 0
        # Skip the entire decoded value to avoid treating its nested objects as candidates.
        for_match = re.compile(r"[\{\[]")
        while match := for_match.search(text, offset):
            try:
                value, end = decoder.raw_decode(text, match.start())
                _check_nesting(text[match.start():end])
            except OutputError:
                raise
            except RecursionError as exc:
                raise OutputError("parse_error", "JSON nesting limit exceeded.") from exc
            except ValueError:
                offset = match.start() + 1
                continue
            offset = end
            if isinstance(value, dict):
                candidates.append(value)
        if len(candidates) != 1:
            raise OutputError("parse_error", "Invalid JSON; recovery requires exactly one complete JSON object.") from original
        return candidates[0], True, "Extracted one JSON object from surrounding text."
    if not isinstance(parsed, dict):
        raise OutputError("validation_error", "Model output must be a JSON object.")
    return parsed, False, None


def _validate(value, schema):
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: str(list(error.path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.path)) or "output"
        raise OutputError("validation_error", f"{location}: {error.message}")


def validate_prediction(prediction, labels):
    _validate(prediction, classification_schema(labels))
    confidence = prediction["confidence"]
    if isinstance(confidence, bool) or not math.isfinite(confidence):
        raise OutputError("validation_error", "confidence must be a finite number, not a boolean.")
    if not prediction["reasoning_summary"].strip():
        raise OutputError("validation_error", "reasoning_summary must not be blank.")
    return prediction


def split_sentences(text):
    """Split reply prose without treating common abbreviations/decimals as endings."""
    text = re.sub(r"\s+", " ", text).strip()
    endings = re.compile(r'''[.!?]+["'\u201d\u2019)\]]*(?=\s|$)''')
    abbreviations = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "e.g.", "i.e.", "etc.", "a.m.", "p.m."}
    sentences, start = [], 0
    for ending in endings.finditer(text):
        punctuation = ending.group().rstrip('"\'\u201d\u2019)]')
        remaining = text[ending.end():].lstrip()
        prefix = text[:ending.start() + len(punctuation)]
        word = re.search(r"([A-Za-z]+(?:\.[A-Za-z]+)*\.)$", prefix)
        token = word.group(1) if word else ""
        if punctuation == "." and remaining:
            if token.lower() in abbreviations or re.fullmatch(r"(?:[A-Za-z]\.){2,}", token):
                continue
            if re.fullmatch(r"[A-Z]\.", token) and remaining[0].isupper():
                continue
        if punctuation == "..." and remaining and remaining[0].islower():
            continue
        sentences.append(text[start:ending.end()].strip())
        start = ending.end()
    if text[start:].strip():
        sentences.append(text[start:].strip())
    return sentences


def validate_reply(value):
    """Validate the canonical representation: 2-4 items, one sentence per item."""
    _validate(value, REPLY_SCHEMA)
    sentences = [sentence.strip() for sentence in value["sentences"]]
    for sentence in sentences:
        if not sentence or not re.search(r'''[.!?]["'\u201d\u2019)\]]*$''', sentence) or "\n" in sentence or "\r" in sentence:
            raise OutputError("validation_error", "Each reply item must be one nonempty sentence ending in punctuation.")
        if len(split_sentences(sentence)) != 1:
            raise OutputError("validation_error", "Each reply array item must contain only one sentence.")
    return sentences, " ".join(sentences)


def parse_reply(text):
    """Recover paragraph-style array items, preserving words and sentence limits."""
    value, recovered, details = parse_object(text)
    try:
        sentences, reply = validate_reply(value)
    except OutputError:
        input_schema = {
            **REPLY_SCHEMA,
            "properties": {"sentences": {**REPLY_SCHEMA["properties"]["sentences"], "minItems": 1}},
        }
        _validate(value, input_schema)
        if any(not item.strip() for item in value["sentences"]):
            raise OutputError("validation_error", "Reply items must not be blank.")
        normalized = [sentence for item in value["sentences"] for sentence in split_sentences(item)]
        # This retains the original 2-4 sentence requirement after normalization.
        sentences, reply = validate_reply({"sentences": normalized})
        normalization = "Split paragraph-style reply items into individual sentences and normalized whitespace."
        recovered = True
        details = f"{details} {normalization}" if details else normalization
    return sentences, reply, recovered, details
