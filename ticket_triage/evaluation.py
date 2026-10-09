"""Pure routing and metric functions shared by the runner and artifact validator."""


THRESHOLD = 0.65


def route_prediction(ticket_id, parsed):
    prediction = parsed["prediction"]
    if prediction is None:
        return {
            "ticket_id": ticket_id, "route": "human_review", "confidence": 0.0,
            "routing_reason": f"Classification failed ({parsed['status']}): {parsed['error']}",
        }
    confidence = prediction["confidence"]
    human = confidence < THRESHOLD
    return {
        "ticket_id": ticket_id, "route": "human_review" if human else "auto_triage",
        "confidence": confidence,
        "routing_reason": f"Confidence {confidence:.4f} {'<' if human else '>='} {THRESHOLD:.2f}.",
    }


def evaluate(tickets, predictions, routes, replies, errors, schema):
    prediction_map = {record["ticket_id"]: record for record in predictions}
    route_map = {record["ticket_id"]: record for record in routes}
    comparison = []
    matrix = {expected: {predicted: 0 for predicted in schema["categories"]} for expected in schema["categories"]}
    unclassified = {label: 0 for label in schema["categories"]}
    for ticket in tickets:
        ticket_id = ticket["ticket_id"]
        parsed = prediction_map[ticket_id]
        prediction = parsed["prediction"]
        category = prediction["category"] if prediction else None
        urgency = prediction["urgency"] if prediction else None
        comparison.append({
            "ticket_id": ticket_id,
            "expected_category": ticket["expected_category"], "predicted_category": category,
            "category_correct": category == ticket["expected_category"],
            "expected_urgency": ticket["expected_urgency"], "predicted_urgency": urgency,
            "urgency_correct": urgency == ticket["expected_urgency"],
            "classification_status": parsed["status"], "route": route_map[ticket_id]["route"],
        })
        if category is None:
            unclassified[ticket["expected_category"]] += 1
        else:
            matrix[ticket["expected_category"]][category] += 1
    count = len(tickets)
    category_correct = sum(record["category_correct"] for record in comparison)
    urgency_correct = sum(record["urgency_correct"] for record in comparison)
    report = {
        "total_tickets": count,
        "category_correct": category_correct, "category_denominator": count,
        "category_accuracy": category_correct / count if count else None,
        "urgency_correct": urgency_correct, "urgency_denominator": count,
        "urgency_accuracy": urgency_correct / count if count else None,
        "human_review_count": sum(route["route"] == "human_review" for route in routes),
        "parse_validation_failures": sum(record["status"] in ("parse_error", "validation_error") for record in predictions),
        "classification_provider_failures": sum(record["status"] == "provider_error" for record in predictions),
        "provider_error_attempts": sum(error["kind"] == "provider_error" for error in errors),
        "recovered_output_count": sum(record["status"] == "valid" and record["recovered"] for record in predictions) + sum(record["status"] == "valid" and record["recovered"] for record in replies),
        "reply_generation_failures": sum(record["status"] != "valid" for record in replies),
    }
    mixups = [{"expected_category": expected, "predicted_category": predicted, "count": frequency}
              for expected, row in matrix.items() for predicted, frequency in row.items() if expected != predicted and frequency]
    mixups.sort(key=lambda item: (-item["count"], item["expected_category"], item["predicted_category"]))
    confusion = {"matrix": matrix, "most_common_mixups": mixups, "unclassified_by_expected_category": unclassified}
    return report, comparison, confusion
