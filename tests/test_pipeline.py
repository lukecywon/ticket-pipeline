"""Offline tests use the actual HTTP adapter with an httpx mock transport."""

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from dotenv import load_dotenv

import main as cli
import validate as validator_cli
from ticket_triage.data import DataError, OutputError, classification_schema, loads_json, parse_object, parse_reply, preprocess, read_json, stable_hash, validate_prediction, validate_reply, validate_schema, validate_tickets, write_json
from ticket_triage.evaluation import evaluate, route_prediction
from ticket_triage.pipeline import run_pipeline
from ticket_triage.provider import ArtifactStore, OpenRouterClient
from ticket_triage.validation import JSON_ARTIFACTS, validate_artifacts


SCHEMA = {"categories": ["billing", "login_access", "other"], "urgency_levels": ["low", "medium", "high"]}
MODEL = "anthropic/claude-haiku-5.5"


def ticket(ticket_id="X", category="billing", urgency="high", message="My payment is missing."):
    return {"ticket_id": ticket_id, "customer_message": message, "expected_category": category, "expected_urgency": urgency}


def prediction(confidence=0.9, category="billing", urgency="high", review=False):
    return {"category": category, "urgency": urgency, "confidence": confidence, "reasoning_summary": "The customer reports a missing payment.", "needs_human_review": review}


def reply():
    return {"sentences": ["I understand that your payment has not appeared.", "Could you share the transaction reference so support can better understand the issue?"]}


def envelope(content):
    if not isinstance(content, str):
        content = json.dumps(content)
    return httpx.Response(200, json={"model": MODEL, "provider": "Anthropic", "choices": [{"finish_reason": "stop", "message": {"content": content}}]})


class CoreTests(unittest.TestCase):
    def test_schema_and_input_validation(self):
        validate_schema(SCHEMA)
        validate_tickets([ticket()], SCHEMA)
        for bad in ({}, {"categories": [], "urgency_levels": ["high"]}, {"categories": ["billing", "billing"], "urgency_levels": ["high"]}, {"categories": [True], "urgency_levels": ["high"]}):
            with self.subTest(schema=bad), self.assertRaises(DataError):
                validate_schema(bad)
        for bad in ([ticket(), ticket()], [dict(ticket(), customer_message="  ")], [dict(ticket(), ticket_id=1)], [dict(ticket(), expected_category="unknown")], [dict(ticket(), expected_urgency=None)]):
            with self.subTest(tickets=bad), self.assertRaises(DataError):
                validate_tickets(bad, SCHEMA)

    def test_preprocessing_preserves_original_and_order(self):
        original = [ticket("B", message=" \r\nI cannot log in.\rDo not reset my account. \r\n"), ticket("A")]
        before = copy.deepcopy(original)
        processed = preprocess(original)
        self.assertEqual(original, before)
        self.assertEqual(processed[0]["customer_message"], original[0]["customer_message"])
        self.assertEqual(processed[0]["normalized_message"], "I cannot log in.\nDo not reset my account.")
        self.assertEqual([row["ticket_id"] for row in processed], ["B", "A"])

    def test_prediction_validation_rejects_bad_fields(self):
        self.assertEqual(validate_prediction(prediction(), SCHEMA), prediction())
        for field, values in {
            "category": ["unknown", None, 7], "urgency": ["urgent", None],
            "confidence": [-0.1, 1.1, True, "0.8", float("nan"), float("inf")],
            "reasoning_summary": ["", "  ", 3], "needs_human_review": ["false", 0],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(OutputError):
                    validate_prediction(dict(prediction(), **{field: value}), SCHEMA)
        for bad in ({"category": "billing"}, dict(prediction(), extra="unexpected")):
            with self.assertRaises(OutputError):
                validate_prediction(bad, SCHEMA)
        self.assertEqual(classification_schema(SCHEMA)["properties"]["category"]["enum"], SCHEMA["categories"])

    def test_json_recovery_and_rejection(self):
        data = prediction()
        value, recovered, details = parse_object("Here is your output:\n```json\n" + json.dumps(data) + "\n``` Done.")
        self.assertEqual(value, data)
        self.assertTrue(recovered)
        self.assertTrue(details)
        self.assertFalse(parse_object(json.dumps(data))[1])
        for text in ("not JSON", "{broken", json.dumps(data) + json.dumps(data), "[]", "null", '[{"category":"billing"}]'):
            with self.subTest(text=text), self.assertRaises(OutputError):
                parse_object(text)
        value, _, _ = parse_object("noise " + json.dumps(dict(data, category="unknown")))
        with self.assertRaises(OutputError):
            validate_prediction(value, SCHEMA)
        for text in ('{"a": 1, "a": 2}', '{"x": NaN}', '{"x": 1e999}'):
            with self.assertRaises(ValueError):
                loads_json(text)

    def test_threshold_ignores_model_flag(self):
        for confidence, expected in ((0, "human_review"), (0.649999, "human_review"), (0.65, "auto_triage"), (1, "auto_triage")):
            for flag in (False, True):
                with self.subTest(confidence=confidence, flag=flag):
                    route = route_prediction("X", {"prediction": prediction(confidence, review=flag), "status": "valid", "error": None})
                    self.assertEqual(route["route"], expected)
        failure = route_prediction("X", {"prediction": None, "status": "parse_error", "error": "Bad JSON"})
        self.assertEqual(failure["route"], "human_review")
        self.assertIn("Bad JSON", failure["routing_reason"])

    def test_reply_sentence_validation(self):
        self.assertEqual(validate_reply(reply())[1], " ".join(reply()["sentences"]))
        for bad in ({"sentences": ["Only one sentence."]}, {"sentences": ["A."] * 5}, {"sentences": ["", "B."]}, {"sentences": ["First. Second.", "Third."]}, {"sentences": ["No punctuation", "B."]}):
            with self.subTest(reply=bad), self.assertRaises(OutputError):
                validate_reply(bad)

    def test_reply_paragraph_recovery_preserves_text_and_sentence_count(self):
        paragraph = (
            "I'm sorry you're still unable to log in after updating your password, especially since the reset email is working on your end. "
            "Invalid credential errors after a reset can sometimes happen when an older saved password or an autofill entry is being used, so please try typing the new password manually in a private browser window. "
            "If the problem continues, reply with the exact error message you see and the time you last attempted to log in so our team can review the situation."
        )
        sentences, text, recovered, details = parse_reply(json.dumps({"sentences": [paragraph]}))
        self.assertEqual(len(sentences), 3)
        self.assertEqual(text, paragraph)
        self.assertTrue(recovered)
        self.assertIn("paragraph-style", details)
        sentences, text, recovered, _ = parse_reply(json.dumps({"sentences": ["First sentence. Second sentence.", "Third sentence."]}))
        self.assertEqual(sentences, ["First sentence.", "Second sentence.", "Third sentence."])
        self.assertTrue(recovered)

    def test_sentence_recovery_handles_abbreviations_numbers_urls_and_quotes(self):
        for paragraph, expected in (
            ("Please contact Dr. Smith. The amount shown is $10.50.", 2),
            ("Please ask A. Smith to check. Use https://support.example.com/help for details.", 2),
            ('The screen says "Invalid credentials." Can you confirm the exact error?', 2),
            ("Check another browser, e.g. Firefox. Please share the error message.", 2),
            ("Please confirm the U.S. billing address. Share only the last four digits.", 2),
            ("We understand the issue.\n\nPlease share the error.\r\nCan you confirm the browser?", 3),
        ):
            with self.subTest(paragraph=paragraph):
                sentences, _, recovered, _ = parse_reply(json.dumps({"sentences": [paragraph]}))
                self.assertEqual(len(sentences), expected)
                self.assertTrue(recovered)

    def test_reply_recovery_does_not_weaken_sentence_or_type_constraints(self):
        for value in (
            {"sentences": ["Only one actual sentence."]},
            {"sentences": ["One. Two. Three. Four. Five."]},
            {"sentences": ["Good sentence.", "   "]},
            {"sentences": "First sentence. Second sentence."},
            {"sentences": [1, "Another sentence."]},
            {"sentences": ["First sentence.", "Second sentence without punctuation"]},
            {"sentences": ["First. Second."], "unexpected": True},
        ):
            with self.subTest(value=value), self.assertRaises(OutputError):
                parse_reply(json.dumps(value))
        sentences, text, recovered, details = parse_reply(json.dumps(reply()))
        self.assertEqual(sentences, reply()["sentences"])
        self.assertEqual(text, " ".join(reply()["sentences"]))
        self.assertFalse(recovered)
        self.assertIsNone(details)

    def test_noisy_json_and_paragraph_reply_recoveries_are_both_reported(self):
        sentences, _, recovered, details = parse_reply('```json\n{"sentences": ["First sentence. Second sentence."]}\n```')
        self.assertEqual(len(sentences), 2)
        self.assertTrue(recovered)
        self.assertIn("Extracted", details)
        self.assertIn("paragraph-style", details)

    def test_excessive_json_nesting_is_a_parse_error(self):
        nested = "[" * 2000 + "0" + "]" * 2000
        with self.assertRaises(ValueError):
            loads_json(nested)
        with self.assertRaises(OutputError) as raised:
            parse_object(nested)
        self.assertEqual(raised.exception.kind, "parse_error")

    def test_accuracy_counts_failures_and_correct_review_predictions(self):
        tickets = [ticket("A"), ticket("B"), ticket("C")]
        predictions = [
            {"ticket_id": "A", "status": "valid", "prediction": prediction(0.5), "recovered": False},
            {"ticket_id": "B", "status": "valid", "prediction": prediction(0.7, urgency="low"), "recovered": False},
            {"ticket_id": "C", "status": "validation_error", "prediction": None, "error": "bad label", "recovered": False},
        ]
        routes = [route_prediction(row["ticket_id"], row) for row in predictions]
        report, comparisons, confusion = evaluate(tickets, predictions, routes, [], [], SCHEMA)
        self.assertEqual(report["category_accuracy"], 2 / 3)
        self.assertEqual(report["urgency_accuracy"], 1 / 3)
        self.assertEqual(report["human_review_count"], 2)
        self.assertEqual(report["parse_validation_failures"], 1)
        self.assertEqual(len(comparisons), 3)
        self.assertEqual(confusion["unclassified_by_expected_category"]["billing"], 1)
        report, _, _ = evaluate([], [], [], [], [], SCHEMA)
        self.assertIsNone(report["category_accuracy"])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "outputs"
        self.tickets_path = self.root / "tickets.json"
        self.schema_path = self.root / "label_schema.json"
        self.requests = []

    def execute(self, responses, tickets=None, schema=None):
        tickets = [ticket()] if tickets is None else tickets
        schema = SCHEMA if schema is None else schema
        write_json(self.tickets_path, tickets)
        write_json(self.schema_path, schema)
        iterator = iter(responses)

        def handler(request):
            self.requests.append(loads_json(request.content.decode()))
            value = next(iterator)
            if isinstance(value, Exception):
                raise value
            return value if isinstance(value, httpx.Response) else envelope(value)

        store = ArtifactStore(self.output)
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            provider = OpenRouterClient("offline-test-key", store, client=client, sleep=lambda _: None)
            return run_pipeline(tickets, schema, store, provider, MODEL, MODEL)

    def validate(self):
        return validate_artifacts(self.tickets_path, self.schema_path, self.output)

    def test_complete_run_regenerates_every_artifact_and_validates(self):
        ok, report = self.execute([prediction(0.65, review=True), reply(), prediction(0.5)], [ticket("first"), ticket("second")])
        self.assertTrue(ok)
        self.assertEqual(report["human_review_count"], 1)
        self.assertEqual(self.validate(), [])
        for name in (*JSON_ARTIFACTS, "llm_calls.jsonl"):
            self.assertTrue((self.output / name).is_file(), name)
        records = read_json(self.output / "triage_results.json")
        self.assertIsNone(records[0]["internal_note"])
        self.assertTrue(records[0]["customer_reply"])
        self.assertIsNone(records[1]["customer_reply"])
        self.assertIn("0.5000", records[1]["internal_note"])
        calls = [loads_json(line) for line in (self.output / "llm_calls.jsonl").read_text().splitlines()]
        self.assertEqual([call["stage"] for call in calls], ["classification", "reply_generation", "classification"])
        self.assertEqual(len(calls), 3)
        for request in self.requests:
            self.assertTrue(request["response_format"]["json_schema"]["strict"])
            self.assertTrue(request["provider"]["require_parameters"])
            self.assertNotIn("expected_category", json.dumps(request))
            self.assertNotIn("expected_urgency", json.dumps(request))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validator_cli.main(["--tickets", str(self.tickets_path), "--schema", str(self.schema_path), "--output-dir", str(self.output)]), 0)

    def test_unknown_labels_and_malformed_output_route_to_review_without_replies(self):
        ok, report = self.execute([prediction(category="unknown"), "not JSON"], [ticket("A"), ticket("B")])
        self.assertTrue(ok)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(report["parse_validation_failures"], 2)
        self.assertEqual(report["category_accuracy"], 0)
        self.assertEqual(self.validate(), [])
        for row in read_json(self.output / "triage_results.json"):
            self.assertEqual(row["route"], "human_review")
            self.assertIsNone(row["predicted_category"])
            self.assertTrue(row["internal_note"])

    def test_recovery_is_visible_in_predictions_errors_and_metrics(self):
        ok, report = self.execute(["```json\n" + json.dumps(prediction(0.4)) + "\n```"])
        self.assertTrue(ok)
        self.assertEqual(report["recovered_output_count"], 1)
        self.assertEqual(report["parse_validation_failures"], 0)
        self.assertTrue(read_json(self.output / "parsed_predictions.json")[0]["recovered"])
        self.assertEqual(read_json(self.output / "errors.json")[0]["kind"], "recovered_output")
        self.assertEqual(self.validate(), [])

    def test_transient_transport_and_http_failure_retry_is_logged(self):
        ok, report = self.execute([httpx.ReadTimeout("timeout"), prediction(0.4), httpx.Response(429, text="slow down"), prediction(0.4)], [ticket("A"), ticket("B")])
        self.assertTrue(ok)
        self.assertEqual(report["provider_error_attempts"], 2)
        self.assertEqual(report["classification_provider_failures"], 0)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(self.validate(), [])

    def test_auth_failure_not_retried_and_later_ticket_continues(self):
        ok, report = self.execute([httpx.Response(401, text="offline-test-key"), prediction(0.4)], [ticket("A"), ticket("B")])
        self.assertFalse(ok)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(report["classification_provider_failures"], 1)
        self.assertIn("incomplete", self.validate()[0])
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertNotIn("offline-test-key", path.read_text(encoding="utf-8"))

    def test_invalid_envelope_is_saved_and_escalated(self):
        ok, report = self.execute([httpx.Response(200, text="not even an API response")])
        self.assertFalse(ok)
        self.assertEqual(report["human_review_count"], 1)
        self.assertEqual(report["classification_provider_failures"], 1)
        self.assertEqual(read_json(self.output / "triage_results.json")[0]["route"], "human_review")

    def test_truncated_response_is_not_accepted(self):
        response = httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": json.dumps(prediction())}}]})
        ok, report = self.execute([response])
        self.assertFalse(ok)
        self.assertEqual(report["classification_provider_failures"], 1)
        self.assertEqual(len(self.requests), 1)

    def test_provider_retry_is_bounded_and_continues(self):
        ok, report = self.execute([httpx.Response(500), httpx.Response(503), prediction(0.4)], [ticket("A"), ticket("B")])
        self.assertFalse(ok)
        self.assertEqual(report["classification_provider_failures"], 1)
        self.assertEqual(report["provider_error_attempts"], 2)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(read_json(self.output / "triage_results.json")), 2)

    def test_noisy_invalid_labels_are_not_counted_as_successful_recovery(self):
        ok, report = self.execute(["Here is the JSON: " + json.dumps(prediction(category="unknown"))])
        self.assertTrue(ok)
        self.assertEqual(report["parse_validation_failures"], 1)
        self.assertEqual(report["recovered_output_count"], 0)
        self.assertEqual(self.validate(), [])

    def test_reply_validation_retries_once(self):
        ok, report = self.execute([prediction(), {"sentences": ["Too short."]}, reply()])
        self.assertTrue(ok)
        self.assertEqual(len(self.requests), 3)
        self.assertIn("Previous response failed", self.requests[-1]["messages"][0]["content"])
        self.assertEqual(report["reply_generation_failures"], 0)
        self.assertEqual(self.validate(), [])

    def test_paragraph_reply_recovery_avoids_retry_and_is_auditable(self):
        paragraph = "I understand that the payment is missing. Please share the transaction reference. Could you confirm when you submitted it?"
        ok, report = self.execute([prediction(), {"sentences": [paragraph]}])
        self.assertTrue(ok)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(report["reply_generation_failures"], 0)
        self.assertEqual(report["recovered_output_count"], 1)
        saved = read_json(self.output / "reply_predictions.json")[0]
        self.assertEqual(len(saved["sentences"]), 3)
        self.assertEqual(saved["customer_reply"], paragraph)
        self.assertTrue(saved["recovered"])
        self.assertIn("paragraph-style", saved["recovery_details"])
        self.assertEqual(read_json(self.output / "errors.json")[0]["kind"], "recovered_output")
        self.assertEqual(self.validate(), [])
        path = self.output / "reply_predictions.json"
        write_json(path, [dict(saved, recovered=False)])
        self.assertTrue(self.validate())

    def test_failed_reply_keeps_route_and_marks_run_incomplete(self):
        ok, report = self.execute([prediction(), "invalid", "still invalid", prediction(0.4)], [ticket("A"), ticket("B")])
        self.assertFalse(ok)
        self.assertEqual(report["reply_generation_failures"], 1)
        results = read_json(self.output / "triage_results.json")
        self.assertEqual(results[0]["route"], "auto_triage")
        self.assertIsNone(results[0]["customer_reply"])
        self.assertEqual(results[1]["route"], "human_review")
        self.assertTrue(self.validate())

    def test_empty_input_creates_valid_empty_artifacts(self):
        ok, report = self.execute([], [])
        self.assertTrue(ok)
        self.assertEqual(report["total_tickets"], 0)
        self.assertIsNone(report["urgency_accuracy"])
        self.assertEqual(self.validate(), [])
        self.assertEqual((self.output / "llm_calls.jsonl").read_text(), "")

    def test_replacement_fixture_dynamic_labels_and_arbitrary_ticket_id(self):
        schema = {"categories": ["custom_a", "custom_b"], "urgency_levels": ["soon", "routine"]}
        fixture = [ticket("../../outside", "custom_b", "soon", "Help with this case.")]
        ok, _ = self.execute([prediction(0.4, "custom_b", "soon")], fixture, schema)
        self.assertTrue(ok)
        self.assertEqual(self.validate(), [])
        self.assertIn("custom_b", self.requests[0]["messages"][0]["content"])
        self.assertEqual(self.requests[0]["response_format"]["json_schema"]["schema"]["properties"]["urgency"]["enum"], ["soon", "routine"])

    def test_validator_rejects_tampered_artifacts(self):
        self.execute([prediction(), reply()])
        for filename, modify in (
            ("routing_decisions.json", lambda data: data + data),
            ("triage_results.json", lambda data: [dict(data[0], customer_reply=None)]),
            ("triage_results.json", lambda data: [dict(data[0], predicted_category="unknown")]),
            ("evaluation_report.json", lambda data: dict(data, category_accuracy=0.17)),
            ("parsed_predictions.json", lambda data: [dict(data[0], prediction=dict(data[0]["prediction"], category="other"))]),
            ("reply_predictions.json", lambda data: [dict(data[0], sentences=["Forged sentence.", "Another sentence."]) ]),
        ):
            path = self.output / filename
            original = read_json(path)
            with self.subTest(filename=filename):
                write_json(path, modify(copy.deepcopy(original)))
                self.assertTrue(self.validate())
                write_json(path, original)
                self.assertEqual(self.validate(), [])
        path = self.output / "evaluation_report.json"
        original = path.read_text()
        path.write_text("broken JSON")
        self.assertTrue(self.validate())
        path.write_text(original)
        path.unlink()
        self.assertIn("Missing artifact", self.validate()[0])

    def test_validator_rejects_raw_call_tampering(self):
        self.execute([prediction(0.4)])
        path = next((self.output / "raw_outputs").rglob("*.json"))
        raw = read_json(path)
        raw["request"]["messages"][0]["content"] += " Unexpected change."
        write_json(path, raw)
        self.assertTrue(self.validate())

    def test_validator_rejects_malformed_and_dangling_call_logs(self):
        self.execute([prediction(0.4)])
        log_path = self.output / "llm_calls.jsonl"
        original = log_path.read_text()
        log_path.write_text("not JSON\n")
        self.assertTrue(self.validate())
        record = loads_json(original)
        record["output_artifact"] = "raw_outputs/../../outside.json"
        log_path.write_text(json.dumps(record) + "\n")
        self.assertTrue(self.validate())
        log_path.write_text(original)
        self.assertEqual(self.validate(), [])

    def test_overwrite_requires_explicit_flag_and_preserves_unrelated_files(self):
        self.execute([prediction(0.4)])
        with self.assertRaises(DataError):
            ArtifactStore(self.output)
        unrelated = self.output / "keep.txt"
        unrelated.write_text("Keep this file.")
        new_store = ArtifactStore(self.output, overwrite=True)
        self.assertEqual(unrelated.read_text(), "Keep this file.")
        self.assertEqual((new_store.root / "llm_calls.jsonl").read_text(), "")


class CLITests(unittest.TestCase):
    def test_cli_names_ticket_with_failed_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tickets_path, schema_path = root / "tickets.json", root / "label_schema.json"
            write_json(tickets_path, [ticket("T1")])
            write_json(schema_path, SCHEMA)
            responses = iter([prediction(), "invalid", "invalid"])
            with httpx.Client(transport=httpx.MockTransport(lambda request: envelope(next(responses)))) as http_client:
                def provider_factory(api_key, store):
                    return OpenRouterClient(api_key, store, client=http_client, sleep=lambda _: None)

                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-key"}, clear=True), patch("main.load_dotenv"), patch("main.OpenRouterClient", side_effect=provider_factory), contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
                    code = cli.main(["--tickets", str(tickets_path), "--schema", str(schema_path), "--output-dir", str(root / "output")])
            self.assertEqual(code, 1)
            self.assertIn("reply generation failures: 1", stdout.getvalue())
            self.assertIn("reply generation failed for T1", stderr.getvalue())

    def test_main_successfully_runs_and_validates_mocked_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tickets_path, schema_path = root / "tickets.json", root / "label_schema.json"
            write_json(tickets_path, [ticket()])
            write_json(schema_path, SCHEMA)
            responses = iter([prediction(), reply()])
            with httpx.Client(transport=httpx.MockTransport(lambda request: envelope(next(responses)))) as http_client:
                def provider_factory(api_key, store):
                    return OpenRouterClient(api_key, store, client=http_client, sleep=lambda _: None)

                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-key"}, clear=True), patch("main.load_dotenv"), patch("main.OpenRouterClient", side_effect=provider_factory), contextlib.redirect_stdout(io.StringIO()) as stdout:
                    code = cli.main(["--tickets", str(tickets_path), "--schema", str(schema_path), "--output-dir", str(root / "output")])
            self.assertEqual(code, 0)
            self.assertIn("Artifact validation passed", stdout.getvalue())

    def test_invalid_input_preflight_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "tickets.json", [ticket(), ticket()])
            write_json(root / "label_schema.json", SCHEMA)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-key"}, clear=True), patch("main.load_dotenv"), patch("httpx.Client.post") as post, contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["--tickets", str(root / "tickets.json"), "--schema", str(root / "label_schema.json"), "--output-dir", str(root / "output")])
            self.assertEqual(code, 2)
            self.assertFalse((root / "output").exists())
            post.assert_not_called()

    def test_missing_credentials_do_not_replace_output_or_make_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            sentinel = output / "llm_calls.jsonl"
            sentinel.write_text("preserve me")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=True), patch("main.load_dotenv"), patch("httpx.Client.post") as post, contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = cli.main(["--output-dir", str(output), "--overwrite"])
            self.assertEqual(code, 2)
            self.assertIn("OPENROUTER_API_KEY is blank", stderr.getvalue())
            self.assertEqual(sentinel.read_text(), "preserve me")
            post.assert_not_called()

    def test_cli_environment_dotenv_and_default_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv_file = Path(directory) / ".env"
            dotenv_file.write_text("OPENROUTER_API_KEY=file-key\nCLASSIFICATION_MODEL=file-classifier\nREPLY_MODEL=file-reply\n")
            actual_loader = load_dotenv
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key", "CLASSIFICATION_MODEL": "env-classifier"}, clear=True), patch("main.load_dotenv", side_effect=lambda *args, **kwargs: actual_loader(dotenv_file, override=False)):
                settings = cli.resolve_settings(cli.parser().parse_args(["--classification-model", "cli-classifier"]))
                self.assertEqual(settings, ("env-key", "cli-classifier", "file-reply"))
                settings = cli.resolve_settings(cli.parser().parse_args([]))
                self.assertEqual(settings[1], "env-classifier")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=True), patch("main.load_dotenv"):
                self.assertEqual(cli.resolve_settings(cli.parser().parse_args([])), ("key", MODEL, MODEL))

    def test_validator_needs_no_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), patch("httpx.Client.post") as post, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(validator_cli.main(["--output-dir", directory]), 1)
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
