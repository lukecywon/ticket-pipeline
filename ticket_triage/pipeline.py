"""Sequential pipeline: classification -> deterministic routing -> reply or note."""

from .data import OutputError, parse_object, parse_reply, preprocess, stable_hash, validate_prediction
from .evaluation import evaluate, route_prediction
from .prompts import classification_request, reply_request
from .provider import ProviderError, utc_now


def classify(ticket, schema, model, provider, store):
    record = {
        "ticket_id": ticket["ticket_id"], "status": "valid", "prediction": None,
        "recovered": False, "recovery_details": None, "error": None, "output_artifact": None,
    }
    try:
        completion = provider.complete(classification_request(ticket, schema, model), ticket["ticket_id"], "classification")
        record["output_artifact"] = completion.output_artifact
        value, recovered, details = parse_object(completion.content)
        record.update(recovered=recovered, recovery_details=details)
        record["prediction"] = validate_prediction(value, schema)
        if recovered:
            store.error(ticket["ticket_id"], "classification", "recovered_output", details, output_artifact=completion.output_artifact)
    except OutputError as exc:
        record.update(status=exc.kind, error=str(exc))
        store.error(ticket["ticket_id"], "classification", exc.kind, str(exc), output_artifact=record["output_artifact"])
    except ProviderError as exc:
        record.update(status="provider_error", error=str(exc), output_artifact=exc.output_artifact)
    return record


def generate_reply(ticket, prediction, model, provider, store):
    record = {
        "ticket_id": ticket["ticket_id"], "status": "valid", "sentences": None,
        "customer_reply": None, "recovered": False, "recovery_details": None, "error": None, "output_artifact": None,
    }
    # The provider handles transport retries. Only invalid content triggers this retry.
    for attempt in range(2):
        try:
            completion = provider.complete(reply_request(ticket, prediction, model, retry=bool(attempt)), ticket["ticket_id"], "reply_generation")
            record["output_artifact"] = completion.output_artifact
            sentences, reply, recovered, details = parse_reply(completion.content)
            record.update(status="valid", sentences=sentences, customer_reply=reply, recovered=recovered, recovery_details=details, error=None)
            if recovered:
                store.error(ticket["ticket_id"], "reply_generation", "recovered_output", details, output_artifact=completion.output_artifact)
            return record
        except OutputError as exc:
            record.update(status=exc.kind, error=str(exc))
            store.error(ticket["ticket_id"], "reply_generation", exc.kind, str(exc), output_artifact=record["output_artifact"], attempt=attempt + 1)
        except ProviderError as exc:
            record.update(status="provider_error", error=str(exc), output_artifact=exc.output_artifact)
            return record
    return record


def run_pipeline(tickets, schema, store, provider, classification_model, reply_model):
    processed = preprocess(tickets)
    metadata = {
        "run_id": store.run_id, "started_at": utc_now(), "finished_at": None, "status": "running",
        "classification_model": classification_model, "reply_model": reply_model,
        "tickets_hash": stable_hash(tickets), "schema_hash": stable_hash(schema),
    }
    store.write("tickets.json", tickets)
    store.write("label_schema.json", schema)
    store.write("preprocessed_tickets.json", processed)
    store.write("run_metadata.json", metadata)
    store.write("errors.json", [])
    predictions, routes, results, replies = [], [], [], []
    for ticket in processed:
        parsed = classify(ticket, schema, classification_model, provider, store)
        predictions.append(parsed)
        route = route_prediction(ticket["ticket_id"], parsed)
        routes.append(route)
        prediction = parsed["prediction"]
        reply = None
        note = None
        if route["route"] == "auto_triage":
            generated = generate_reply(ticket, prediction, reply_model, provider, store)
            replies.append(generated)
            reply = generated["customer_reply"]
        else:
            note = f"Escalated for human review. {route['routing_reason']}"
        results.append({
            "ticket_id": ticket["ticket_id"],
            "predicted_category": prediction["category"] if prediction else None,
            "predicted_urgency": prediction["urgency"] if prediction else None,
            "confidence": route["confidence"], "route": route["route"],
            "customer_reply": reply, "internal_note": note,
        })
        # Checkpoint after each ticket so a later failure does not lose prior work.
        store.write("parsed_predictions.json", predictions)
        store.write("routing_decisions.json", routes)
        store.write("triage_results.json", results)
        store.write("reply_predictions.json", replies)
    # Also write these for empty input.
    for name, records in (("parsed_predictions.json", predictions), ("routing_decisions.json", routes), ("triage_results.json", results), ("reply_predictions.json", replies)):
        store.write(name, records)
    report, comparisons, confusion = evaluate(tickets, predictions, routes, replies, store.errors, schema)
    store.write("evaluation_report.json", report)
    store.write("prediction_comparison.json", comparisons)
    store.write("confusion_summary.json", confusion)
    successful = not (report["classification_provider_failures"] or report["reply_generation_failures"])
    metadata.update(finished_at=utc_now(), status="complete" if successful else "incomplete")
    store.write("run_metadata.json", metadata)
    return successful, report
