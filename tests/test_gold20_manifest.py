from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import easy_agentic_data.evaluation as evaluation_module
import easy_agentic_data.gold20_runtime as gold20_runtime
import easy_agentic_data.registry as registry_module
import easy_agentic_data.sandbox.docker as docker_sandbox
import easy_agentic_data.scenarios as scenarios_module
from easy_agentic_data.cli import main
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.gold20 import (
    GOLD20_CONTAINER_REPLAY_SCHEMA_VERSION,
    GOLD20_FREEZE_CONFIG_SCHEMA_VERSION,
    GOLD20_MANIFEST_SCHEMA_VERSION,
    GOLD20_MATERIALIZATION_SCHEMA_VERSION,
    GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION,
    GOLD20_RUNTIME_PLATFORM,
    GOLD20_RUNTIME_RANDOM_SEED,
    GOLD_REPAIR_VALIDATION_SCHEMA_VERSION,
    HIDDEN_TEST_PATCH_VALIDATION_SCHEMA_VERSION,
    freeze_gold20,
)
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.sandbox import SandboxLimits
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seed_corpus import REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class Gold20ManifestTests(unittest.TestCase):
    def test_freeze_writes_metadata_only_deterministic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            manifest = freeze_gold20(fixture["config"])
            repeated = freeze_gold20(fixture["config"])

            self.assertTrue(manifest["valid"], manifest["issues"])
            self.assertEqual(manifest["schema_version"], GOLD20_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["corpus_id"], repeated["corpus_id"])
            self.assertEqual(
                [record["record_sha256"] for record in manifest["records"]],
                [record["record_sha256"] for record in repeated["records"]],
            )
            self.assertEqual(len(manifest["records"]), 20)
            self.assertEqual(manifest["coverage"], {"repositories": 8, "languages": 2})
            self.assertEqual(manifest["records"][0]["repository"].split("/")[0][:6], "owner-")
            self.assertRegex(
                manifest["records"][0]["hashes"]["workspace_tree_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(
                manifest["evidence"]["reference_repair_evidence_sha256"],
                _file_sha256(fixture["reference_repairs"]),
            )
            self.assertEqual(manifest["audits"]["resolved_finding_count"], 3)
            self.assertEqual(manifest["audits"]["unresolved_finding_count"], 0)
            self.assertEqual(
                json.loads(fixture["manifest"].read_text(encoding="utf-8")),
                repeated,
            )

            serialized = json.dumps(manifest, sort_keys=True)
            for secret in fixture["private_canaries"]:
                self.assertNotIn(secret, serialized)
            self.assertNotIn(str(fixture["root"]), serialized)
            self.assertNotIn("workspace-private", serialized)
            self.assertNotIn("evaluator-private", serialized)
            self.assertNotIn("https://github.com/", serialized)

    def test_freeze_binds_container_replay_artifact_and_record_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))

            manifest = freeze_gold20(fixture["config"])

            self.assertTrue(manifest["valid"], manifest["issues"])
            self.assertEqual(
                manifest["evidence"]["container_replay_sha256"],
                _file_sha256(fixture["container_replay"]),
            )
            self.assertEqual(
                manifest["evidence"]["runtime_build_spec_sha256"],
                sorted(_file_sha256(path) for path in fixture["runtime_build_specs"]),
            )
            replay_records = {
                record["source_instance_id"]: record
                for record in _read_json(fixture["container_replay"])["records"]
            }
            for record in manifest["records"]:
                replay = replay_records[record["source_instance_id"]]
                self.assertEqual(
                    record["hashes"]["container_replay_record_sha256"],
                    _stable_json_sha256(_safe_container_replay_record(replay)),
                )

    def test_freeze_rejects_container_image_policy_and_exit_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            replay = _read_json(fixture["container_replay"])
            replay["records"][0]["image_digest"] = f"sha256:{'f' * 64}"
            replay["records"][1]["sandbox_policy"]["network_mode"] = "bridge"
            replay["records"][2]["repaired_hidden_test_exit_codes"] = [1]
            _write_json(fixture["container_replay"], replay)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertFalse(manifest["valid"])
            self.assertIn("container_replay_image_mismatch", codes)
            self.assertIn("container_replay_sandbox_policy_mismatch", codes)
            self.assertIn("container_replay_repair_did_not_pass", codes)

    def test_freeze_requires_disabled_scenario_network_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            scenario = _read_json(scenario_path)
            scenario["environment"]["network_policy"] = "enabled"
            _write_json(scenario_path, scenario)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("scenario_network_policy_not_disabled", _issue_codes(manifest))

    def test_freeze_rejects_unsupported_evaluator_surfaces(self) -> None:
        cases = (
            ("required_state", {"private": "required"}),
            ("forbidden_state", {"private": "forbidden"}),
            ("retrieval_requirements", ["inspect a file"]),
            ("trace_quality_rubric", ["use a tool"]),
        )
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = _build_fixture(Path(directory))
                scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
                scenario = _read_json(scenario_path)
                if field in {"required_state", "forbidden_state"}:
                    scenario["hidden_evaluator"][field] = value
                else:
                    scenario["hidden_evaluator"]["metadata"][field] = value
                _write_json(scenario_path, scenario)

                manifest = freeze_gold20(fixture["config"])

                self.assertFalse(manifest["valid"])
                self.assertIn(
                    "scenario_unsupported_evaluator_surface",
                    _issue_codes(manifest),
                )

    def test_freeze_requires_exact_runtime_image_id_and_verification_mode(self) -> None:
        cases = (
            ("image_id", None, "container_replay_runtime_image_id_mismatch"),
            (
                "image_id",
                f"sha256:{'f' * 64}",
                "container_replay_runtime_image_id_mismatch",
            ),
            (
                "build_verification_mode",
                None,
                "container_replay_runtime_build_verification_mode_invalid",
            ),
            (
                "build_verification_mode",
                "declared_spec_only",
                "container_replay_runtime_build_verification_mode_invalid",
            ),
        )
        for field, value, issue_code in cases:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                fixture = _build_fixture(Path(directory))
                replay = _read_json(fixture["container_replay"])
                if value is None:
                    replay["runtime_builds"][0].pop(field)
                else:
                    replay["runtime_builds"][0][field] = value
                _write_json(fixture["container_replay"], replay)

                manifest = freeze_gold20(fixture["config"])

                self.assertFalse(manifest["valid"])
                self.assertIn(issue_code, _issue_codes(manifest))

    def test_freeze_binds_runtime_producer_component_hashes(self) -> None:
        cases = ("missing", "tampered")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = _build_fixture(Path(directory))
                replay = _read_json(fixture["container_replay"])
                if case == "missing":
                    replay["producer"].pop("component_sha256s")
                else:
                    replay["producer"]["component_sha256s"]["evaluation.py"] = "f" * 64
                _write_json(fixture["container_replay"], replay)

                manifest = freeze_gold20(fixture["config"])

                self.assertFalse(manifest["valid"])
                self.assertIn(
                    "container_replay_producer_component_hash_mismatch",
                    _issue_codes(manifest),
                )

    def test_freeze_recomputes_hidden_test_semantic_result_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            replay = _read_json(fixture["container_replay"])
            replay["records"][0]["base_hidden_test_result_sha256s"] = ["f" * 64]
            _write_json(fixture["container_replay"], replay)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn(
                "container_replay_hidden_test_result_hash_invalid",
                _issue_codes(manifest),
            )

    def test_freeze_rejects_mismatched_hidden_result_vector_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            replay = _read_json(fixture["container_replay"])
            replay["records"][0]["base_hidden_test_exit_codes"] = [1, 2]
            _write_json(fixture["container_replay"], replay)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn(
                "container_replay_hidden_test_result_hash_invalid",
                _issue_codes(manifest),
            )

    def test_freeze_rejects_non_exact_container_replay_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            replay = _read_json(fixture["container_replay"])
            replay["records"].pop()
            replay["counts"] = {"records": 20, "valid": 20, "invalid": 0}
            _write_json(fixture["container_replay"], replay)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertFalse(manifest["valid"])
            self.assertIn("container_replay_record_set_mismatch", codes)
            self.assertIn("container_replay_scenario_set_mismatch", codes)
            self.assertIn("container_replay_counts_mismatch", codes)
            self.assertIn("container_replay_artifact_invalid", codes)

    def test_freeze_redacts_private_container_replay_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            replay = _read_json(fixture["container_replay"])
            artifact_secret = "container-artifact-private-canary"
            record_secret = "container-record-private-canary"
            replay["private_runtime_debug"] = artifact_secret
            replay["records"][0]["private_runtime_debug"] = record_secret
            _write_json(fixture["container_replay"], replay)

            manifest = freeze_gold20(fixture["config"])

            self.assertTrue(manifest["valid"], manifest["issues"])
            serialized = json.dumps(manifest, sort_keys=True)
            self.assertNotIn(artifact_secret, serialized)
            self.assertNotIn(record_secret, serialized)

    def test_corpus_id_binds_holdout_and_decontamination_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            original = freeze_gold20(fixture["config"])
            holdout_registry = ScenarioRegistry(fixture["holdout"])
            holdout_seed = QuerySeed(
                public=PublicTaskContext(
                    query="Second held-out benchmark query",
                    context={
                        "repository": "holdout/second",
                        "source_instance_id": "holdout-2",
                    },
                ),
                provenance="swe_bench:holdout-2",
                license="MIT",
                split="evaluation",
                task_family="bug_repair",
                source_method="external_issue_workspace",
                contamination_tags=["benchmark_source"],
                verifier_types=["hidden_test_patch"],
                coverage_tags=["language:python"],
                metadata={"source_name": "swe_bench", "source_instance_id": "holdout-2"},
            )
            holdout_environment = EnvironmentSpec(
                name="holdout-second",
                version="1",
                image_digest=f"sha256:{'c' * 64}",
                source_uri="https://github.com/holdout/second.git",
                source_revision="d" * 40,
                health_check=["python -c 'pass'"],
                metadata={"repository": "holdout/second", "language": "python"},
            )
            second_holdout_patch = "diff --git a/second b/second\n+holdout-second\n"
            holdout_registry.add_scenario(
                Scenario(
                    query_seed=holdout_seed,
                    environment=holdout_environment,
                    hidden_evaluator=HiddenEvaluatorContext(
                        reference_artifacts=["private://holdout-second-artifact"],
                        hidden_tests=["python -m pytest holdout-second-only-test"],
                        metadata={
                            "test_patch": second_holdout_patch,
                            "test_patch_sha256": _text_sha256(second_holdout_patch),
                        },
                    ),
                    metadata={
                        "source_name": "swe_bench",
                        "source_instance_id": "holdout-2",
                    },
                )
            )

            changed = freeze_gold20(fixture["config"])

            self.assertTrue(original["valid"], original["issues"])
            self.assertTrue(changed["valid"], changed["issues"])
            self.assertNotEqual(original["corpus_id"], changed["corpus_id"])

    def test_cli_freezes_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "registry",
                        "freeze-gold-20",
                        "--config",
                        str(fixture["config"]),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(json.loads(output.getvalue())["valid"])

    def test_cli_returns_two_when_freeze_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            evidence = _read_json(fixture["materialization"])
            evidence["valid"] = False
            _write_json(fixture["materialization"], evidence)
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "registry",
                        "freeze-gold-20",
                        "--config",
                        str(fixture["config"]),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertFalse(json.loads(output.getvalue())["valid"])

    def test_invalid_freeze_does_not_replace_valid_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            valid_manifest = freeze_gold20(fixture["config"])
            original_bytes = fixture["manifest"].read_bytes()
            materialization = _read_json(fixture["materialization"])
            materialization["valid"] = False
            _write_json(fixture["materialization"], materialization)

            invalid_manifest = freeze_gold20(fixture["config"])

            self.assertTrue(valid_manifest["valid"])
            self.assertFalse(invalid_manifest["valid"])
            self.assertEqual(fixture["manifest"].read_bytes(), original_bytes)

    def test_freeze_rejects_non_reproducible_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            evidence = _read_json(fixture["materialization"])
            evidence["records"][0]["attempts"] = 1
            evidence["records"][0]["workspace_tree_hashes"] = ["1" * 64, "2" * 64]
            _write_json(fixture["materialization"], evidence)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("materialization_attempts_insufficient", _issue_codes(manifest))
            self.assertIn("materialization_tree_hash_mismatch", _issue_codes(manifest))
            self.assertTrue(all(not record["valid"] for record in manifest["records"]))

    def test_freeze_binds_repair_validation_to_materialized_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            evidence = _read_json(fixture["materialization"])
            evidence["records"][0]["workspace_tree_hashes"] = ["f" * 64, "f" * 64]
            for result in evidence["records"][0]["attempt_results"]:
                result["workspace_tree_sha256"] = "f" * 64
            _write_json(fixture["materialization"], evidence)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertNotIn("materialization_tree_hash_mismatch", _issue_codes(manifest))
            self.assertIn(
                "repair_validation_materialization_tree_mismatch",
                _issue_codes(manifest),
            )

    def test_freeze_derives_rehearsal_failure_from_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            rehearsal = _read_json(fixture["rehearsal"])
            for result in rehearsal["hidden_test_patch_rehearsal"]["results"]:
                result["command_results"][0]["exit_code"] = 0
            _write_json(fixture["rehearsal"], rehearsal)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("hidden_patch_rehearsal_original_did_not_fail", codes)
            self.assertIn("hidden_patch_rehearsal_outcome_label_mismatch", codes)

    def test_freeze_requires_health_check_and_content_addressed_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            scenario = _read_json(scenario_path)
            scenario["environment"]["health_check"] = []
            scenario["environment"]["image_digest"] = "example/runtime:latest"
            _write_json(scenario_path, scenario)
            environment_path = (
                fixture["registry"]
                / "environments"
                / f"{scenario['environment']['environment_id']}.json"
            )
            environment = _read_json(environment_path)
            environment["health_check"] = []
            environment["image_digest"] = "example/runtime:latest"
            _write_json(environment_path, environment)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("scenario_health_check_empty", _issue_codes(manifest))
            self.assertIn("scenario_image_not_content_addressed", _issue_codes(manifest))

    def test_freeze_requires_patch_fail_repair_validation_triad(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            validation_path = fixture["validations"][0]
            validation = _read_json(validation_path)
            validation["patch_check_exit"] = False
            validation["original_hidden_test_exit"] = 0
            validation["fixed_hidden_test_exit"] = 1
            validation["validated_repair_sha256"] = ""
            _write_json(validation_path, validation)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("repair_validation_patch_check_failed", codes)
            self.assertIn("repair_validation_original_did_not_fail", codes)
            self.assertIn("repair_validation_repair_did_not_pass", codes)
            self.assertIn("repair_validation_repair_hash_missing", codes)

    def test_freeze_binds_validation_to_reference_repair_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            validation_path = fixture["validations"][0]
            validation = _read_json(validation_path)
            validation["validated_repair_sha256"] = "f" * 64
            _write_json(validation_path, validation)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn(
                "repair_validation_reference_repair_hash_mismatch",
                _issue_codes(manifest),
            )

    def test_freeze_recomputes_reference_repair_hash_from_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            bundle = _read_json(fixture["reference_repairs"])
            bundle["records"][0]["repair_patch"] += "# tampered\n"
            _write_json(fixture["reference_repairs"], bundle)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("reference_repair_patch_hash_mismatch", _issue_codes(manifest))
            serialized = json.dumps(manifest, sort_keys=True)
            self.assertNotIn("# tampered", serialized)

    def test_freeze_binds_validation_and_repair_bundle_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            validation_path = fixture["validations"][0]
            validation = _read_json(validation_path)
            validation["scenario_id"] = "scenario_wrong"
            validation["seed_id"] = "seed_wrong"
            validation["environment_id"] = "env_wrong"
            _write_json(validation_path, validation)
            bundle = _read_json(fixture["reference_repairs"])
            bundle["records"][1]["scenario_id"] = "scenario_wrong"
            bundle["records"][1]["seed_id"] = "seed_wrong"
            bundle["records"][1]["environment_id"] = "env_wrong"
            _write_json(fixture["reference_repairs"], bundle)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("repair_validation_scenario_id_mismatch", codes)
            self.assertIn("repair_validation_seed_id_mismatch", codes)
            self.assertIn("repair_validation_environment_id_mismatch", codes)
            self.assertIn("reference_repair_scenario_id_mismatch", codes)
            self.assertIn("reference_repair_seed_id_mismatch", codes)
            self.assertIn("reference_repair_environment_id_mismatch", codes)

    def test_freeze_accepts_missing_optional_apply_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            for validation_path in fixture["validations"]:
                validation = _read_json(validation_path)
                validation.pop("repair_check_exit")
                validation.pop("repaired_patch_check_exit")
                _write_json(validation_path, validation)

            manifest = freeze_gold20(fixture["config"])

            self.assertTrue(manifest["valid"], manifest["issues"])

    def test_freeze_rejects_nonzero_optional_apply_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            repair_validation = _read_json(fixture["validations"][0])
            repair_validation["repair_check_exit"] = 1
            _write_json(fixture["validations"][0], repair_validation)
            hidden_patch_validation = _read_json(fixture["validations"][1])
            hidden_patch_validation["repaired_patch_check_exit"] = 1
            _write_json(fixture["validations"][1], hidden_patch_validation)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("repair_validation_repair_check_failed", codes)
            self.assertIn("repair_validation_repaired_patch_check_failed", codes)

    def test_freeze_rejects_invalid_aggregate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            aggregate = fixture["root"] / "invalid-aggregate.json"
            records = [_read_json(path) for path in fixture["validations"]]
            _write_json(
                aggregate,
                {
                    "schema_version": GOLD_REPAIR_VALIDATION_SCHEMA_VERSION,
                    "counts": {"records": 20, "valid": 0, "invalid": 20},
                    "records": [*records, None],
                    "valid": False,
                },
            )
            config = _read_json(fixture["config"])
            config["repair_validation_evidence"] = [str(aggregate)]
            _write_json(fixture["config"], config)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("repair_validation_aggregate_invalid", codes)
            self.assertIn("repair_validation_aggregate_counts_mismatch", codes)
            self.assertIn("repair_validation_aggregate_records_invalid", codes)

    def test_freeze_requires_exact_repair_validation_command_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            validation_path = fixture["validations"][0]
            validation = _read_json(validation_path)
            command_hash = validation.pop("hidden_test_command_sha256")
            validation["hidden_test_command_sha256s"] = [command_hash, "f" * 64]
            _write_json(validation_path, validation)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("repair_validation_command_hash_mismatch", _issue_codes(manifest))

    def test_freeze_requires_schema_for_standalone_repair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            validation_path = fixture["validations"][0]
            validation = _read_json(validation_path)
            validation.pop("schema_version")
            _write_json(validation_path, validation)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("repair_validation_schema_invalid", _issue_codes(manifest))

    def test_freeze_accepts_aggregated_standardized_repair_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            records = [_read_json(path) for path in fixture["validations"]]
            for record in records:
                record.pop("schema_version")
                record["repair_patch_sha256"] = record.pop("validated_repair_sha256")
                record["base"] = {
                    "hidden_patch_check": {"exit_code": record.pop("patch_check_exit")},
                    "hidden_patch_apply": {"exit_code": record.pop("patch_apply_exit")},
                    "hidden_test_run": {"exit_code": record.pop("original_hidden_test_exit")},
                }
                record["repaired"] = {
                    "repair_patch_check": {"exit_code": record.pop("repair_check_exit")},
                    "repair_patch_apply": {"exit_code": record.pop("repair_apply_exit")},
                    "hidden_patch_check": {"exit_code": record.pop("repaired_patch_check_exit")},
                    "hidden_patch_apply": {"exit_code": record.pop("repaired_patch_apply_exit")},
                    "hidden_test_run": {"exit_code": record.pop("fixed_hidden_test_exit")},
                }
            aggregate = fixture["root"] / "standardized-validations.json"
            _write_json(
                aggregate,
                {
                    "schema_version": GOLD_REPAIR_VALIDATION_SCHEMA_VERSION,
                    "counts": {"records": 20, "valid": 20, "invalid": 0},
                    "records": records,
                    "valid": True,
                },
            )
            config = _read_json(fixture["config"])
            config["repair_validation_evidence"] = [str(aggregate)]
            _write_json(fixture["config"], config)

            manifest = freeze_gold20(fixture["config"])

            self.assertTrue(manifest["valid"], manifest["issues"])
            self.assertEqual(
                manifest["evidence"]["repair_validation_artifact_sha256"],
                [_file_sha256(aggregate)],
            )

    def test_freeze_rejects_source_set_drift_and_non_exact_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            lines = fixture["source"].read_text(encoding="utf-8").splitlines()
            fixture["source"].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            scenario_path.unlink()

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("gold20_scenario_count_not_exact", codes)
            self.assertIn("source_record_count_not_exact", codes)
            self.assertIn("source_record_set_mismatch", codes)

    def test_freeze_binds_public_query_and_workspace_recipe_to_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            source_records = [
                json.loads(line)
                for line in fixture["source"].read_text(encoding="utf-8").splitlines()
                if line
            ]
            source_records[0]["problem_statement"] = "A different public task"
            source_records[0]["setup_commands"] = ["python -m pip install hidden-package"]
            source_records[0]["health_check"] = ["python -c 'raise SystemExit(1)'"]
            source_records[0]["image_digest"] = f"sha256:{'f' * 64}"
            source_records[0]["source_uri"] = "file:///different/workspace"
            source_records[0]["workspace_original_source_uri"] = (
                "https://github.com/different/repository.git"
            )
            fixture["source"].write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in source_records),
                encoding="utf-8",
            )
            rehearsal = _read_json(fixture["rehearsal"])
            rehearsal["source"]["sha256"] = _file_sha256(fixture["source"])
            _write_json(fixture["rehearsal"], rehearsal)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("source_query_registry_mismatch", codes)
            self.assertIn("source_setup_commands_mismatch", codes)
            self.assertIn("source_health_check_mismatch", codes)
            self.assertIn("source_image_digest_mismatch", codes)
            self.assertIn("source_workspace_uri_mismatch", codes)
            self.assertIn("source_original_workspace_uri_mismatch", codes)

    def test_freeze_requires_explicit_resolution_for_warning_only_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            config = _read_json(fixture["config"])
            config.pop("resolved_contamination_findings")
            _write_json(fixture["config"], config)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("seed_audit_holdout_repository_overlap", _issue_codes(manifest))
            self.assertEqual(manifest["audits"]["resolved_finding_count"], 0)
            self.assertEqual(manifest["audits"]["unresolved_finding_count"], 3)

    def test_freeze_rejects_unmatched_contamination_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            config = _read_json(fixture["config"])
            config["resolved_contamination_findings"].append(
                {
                    "audit": "seed_library",
                    "code": "holdout_repository_overlap",
                    "entry_id": "seed_missing",
                    "rationale": "This resolution does not identify a reported warning.",
                }
            )
            _write_json(fixture["config"], config)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("contamination_resolution_unmatched", _issue_codes(manifest))

    def test_freeze_rejects_and_redacts_private_source_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            source_records = [
                json.loads(line)
                for line in fixture["source"].read_text(encoding="utf-8").splitlines()
                if line
            ]
            secret_url = "https://private.corp.example/internal?token=TOPSECRET"
            source_records[0]["source_url"] = secret_url
            fixture["source"].write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in source_records),
                encoding="utf-8",
            )
            rehearsal = _read_json(fixture["rehearsal"])
            rehearsal["source"]["sha256"] = _file_sha256(fixture["source"])
            _write_json(fixture["rehearsal"], rehearsal)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("source_url_missing_or_untrusted", _issue_codes(manifest))
            self.assertNotIn(secret_url, json.dumps(manifest, sort_keys=True))
            self.assertNotIn("TOPSECRET", json.dumps(manifest, sort_keys=True))

    def test_freeze_binds_github_source_url_to_registry_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            source_records = [
                json.loads(line)
                for line in fixture["source"].read_text(encoding="utf-8").splitlines()
                if line
            ]
            repository = source_records[0]["repo"]
            source_records[0]["source_url"] = f"https://github.com/{repository}/pull/999999"
            fixture["source"].write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in source_records),
                encoding="utf-8",
            )
            rehearsal = _read_json(fixture["rehearsal"])
            rehearsal["source"]["sha256"] = _file_sha256(fixture["source"])
            _write_json(fixture["rehearsal"], rehearsal)

            manifest = freeze_gold20(fixture["config"])

            codes = _issue_codes(manifest)
            self.assertIn("source_url_registry_mismatch", codes)
            self.assertIn("source_instance_url_identity_mismatch", codes)

    def test_freeze_verifies_holdout_oracle_hash_before_decontamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            holdout_path = next((fixture["holdout"] / "scenarios").glob("*.json"))
            holdout = _read_json(holdout_path)
            holdout["hidden_evaluator"]["metadata"]["test_patch_sha256"] = "f" * 64
            _write_json(holdout_path, holdout)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("holdout_hidden_test_patch_hash_mismatch", _issue_codes(manifest))

    def test_freeze_requires_a_benchmark_scenario_in_holdout_basis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            holdout_path = next((fixture["holdout"] / "scenarios").glob("*.json"))
            holdout = _read_json(holdout_path)
            holdout["query_seed"]["provenance"] = "curated_holdout:holdout-1"
            holdout["query_seed"]["contamination_tags"] = []
            holdout["query_seed"]["metadata"]["source_name"] = "curated_holdout"
            holdout["metadata"]["source_name"] = "curated_holdout"
            _write_json(holdout_path, holdout)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("holdout_registry_has_no_benchmark_scenarios", _issue_codes(manifest))

    def test_freeze_rejects_hidden_evaluator_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            scenario = _read_json(scenario_path)
            patch = scenario["hidden_evaluator"]["metadata"]["test_patch"]
            hidden_filename = patch.split(" a/", 1)[1].split(" ", 1)[0]
            scenario["query_seed"]["public"]["query"] += f"\nInspect {hidden_filename}."
            _write_json(scenario_path, scenario)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("hidden_evaluator_leaked_to_public_task", _issue_codes(manifest))

    def test_freeze_rejects_cross_scenario_evaluator_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_paths = sorted((fixture["registry"] / "scenarios").glob("*.json"))
            hidden = _read_json(scenario_paths[0])
            public = _read_json(scenario_paths[1])
            patch = hidden["hidden_evaluator"]["metadata"]["test_patch"]
            hidden_filename = patch.split(" a/", 1)[1].split(" ", 1)[0]
            public["query_seed"]["public"]["query"] += f"\nInspect {hidden_filename}."
            _write_json(scenario_paths[1], public)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn(
                "hidden_evaluator_leaked_across_public_tasks",
                _issue_codes(manifest),
            )

    def test_freeze_rejects_short_private_canary_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            scenario = _read_json(scenario_path)
            scenario["hidden_evaluator"]["reference_answer"] = "ZXQ1234"
            scenario["query_seed"]["public"]["query"] += "\nLeaked: ZXQ1234."
            _write_json(scenario_path, scenario)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("hidden_evaluator_leaked_to_public_task", _issue_codes(manifest))

    def test_freeze_redacts_private_canary_from_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            scenario = _read_json(scenario_path)
            secret = scenario["hidden_evaluator"]["reference_answer"]
            scenario["query_seed"]["metadata"]["permitted_use"] = secret
            scenario["environment"]["metadata"]["permitted_use"] = secret
            _write_json(scenario_path, scenario)
            seed_path = fixture["registry"] / "seeds" / f"{scenario['query_seed']['seed_id']}.json"
            seed = _read_json(seed_path)
            seed["metadata"]["permitted_use"] = secret
            _write_json(seed_path, seed)
            environment_path = (
                fixture["registry"]
                / "environments"
                / f"{scenario['environment']['environment_id']}.json"
            )
            environment = _read_json(environment_path)
            environment["metadata"]["permitted_use"] = secret
            _write_json(environment_path, environment)
            source_records = [
                json.loads(line)
                for line in fixture["source"].read_text(encoding="utf-8").splitlines()
                if line
            ]
            target_source_id = scenario["metadata"]["source_instance_id"]
            for source_record in source_records:
                if source_record["instance_id"] == target_source_id:
                    source_record["permitted_use"] = secret
            fixture["source"].write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in source_records),
                encoding="utf-8",
            )
            rehearsal = _read_json(fixture["rehearsal"])
            rehearsal["source"]["sha256"] = _file_sha256(fixture["source"])
            _write_json(fixture["rehearsal"], rehearsal)

            manifest = freeze_gold20(fixture["config"])

            self.assertIn("manifest_metadata_contains_private_canary", _issue_codes(manifest))
            self.assertNotIn(secret, json.dumps(manifest, sort_keys=True))

    def test_freeze_redacts_cross_scenario_canary_from_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            scenario_paths = sorted((fixture["registry"] / "scenarios").glob("*.json"))
            hidden_scenario = _read_json(scenario_paths[1])
            hidden_scenario["hidden_evaluator"]["reference_answer"] = "research"
            _write_json(scenario_paths[1], hidden_scenario)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn("manifest_metadata_contains_private_canary", _issue_codes(manifest))
            self.assertNotIn(
                '"permitted_use": "research"',
                json.dumps(manifest, sort_keys=True),
            )

    def test_freeze_recomputes_scenario_decontamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            train_path = sorted((fixture["registry"] / "scenarios").glob("*.json"))[0]
            train_command = _read_json(train_path)["hidden_evaluator"]["hidden_tests"][0]
            holdout_path = next((fixture["holdout"] / "scenarios").glob("*.json"))
            holdout = _read_json(holdout_path)
            holdout["hidden_evaluator"]["hidden_tests"] = [train_command]
            _write_json(holdout_path, holdout)
            config = _read_json(fixture["config"])
            config["resolved_contamination_findings"].append(
                {
                    "audit": "scenario_decontamination",
                    "code": "holdout_hidden_test_overlap",
                    "entry_id": _read_json(train_path)["scenario_id"],
                    "rationale": "An error-level oracle overlap must remain impossible to waive.",
                }
            )
            _write_json(fixture["config"], config)

            manifest = freeze_gold20(fixture["config"])

            self.assertFalse(manifest["valid"])
            self.assertIn(
                "scenario_audit_holdout_hidden_test_overlap",
                _issue_codes(manifest),
            )
            self.assertIn(
                "contamination_resolution_cannot_waive_error",
                _issue_codes(manifest),
            )


def _build_fixture(root: Path) -> dict[str, object]:
    registry_root = root / "registry"
    registry = ScenarioRegistry(registry_root)
    source_records = []
    scenarios = []
    private_canaries = []
    for index in range(20):
        repository = f"owner-{index % 8}/repo-{index % 8}"
        language = "python" if index % 2 == 0 else "javascript"
        source_id = f"owner-{index % 8}__repo-{index % 8}-pr-{1000 + index}"
        source_url = f"https://github.com/{repository}/pull/{1000 + index}"
        public_query = f"Repair public regression {index}"
        revision = hashlib.sha1(f"revision-{index}".encode()).hexdigest()
        patch = f"diff --git a/private-{index}.txt b/private-{index}.txt\n+secret-{index}\n"
        command = f"python -m pytest evaluator-private-{index}"
        patch_sha = _text_sha256(patch)
        seed = QuerySeed(
            public=PublicTaskContext(
                query=public_query,
                context={
                    "repository": repository,
                    "source_instance_id": source_id,
                    "source_type": "public_pr",
                    "source_url": source_url,
                },
            ),
            category="software_engineering",
            difficulty=3,
            provenance=f"gold20_public:{source_id}",
            license="MIT",
            split="train",
            task_family="bug_repair",
            source_method="external_issue_workspace",
            train_eligible=True,
            verifier_types=["hidden_command", "hidden_test_patch"],
            coverage_tags=[f"language:{language}", f"repo:{repository}"],
            metadata={
                "source_name": "gold20_public",
                "source_instance_id": source_id,
                "source_type": "public_pr",
                "source_url": source_url,
                "permitted_use": "research",
                "language": language,
            },
        )
        image = f"ghcr.io/example/runtime@sha256:{hashlib.sha256(language.encode()).hexdigest()}"
        environment = EnvironmentSpec(
            name=f"gold20-{index}",
            version="1",
            image_digest=image,
            source_uri=f"https://github.com/{repository}.git",
            source_revision=revision,
            setup_commands=[],
            health_check=["python -c 'pass'"],
            evaluator_refs=[f"private://evaluator-private-{index}"],
            metadata={
                "repository": repository,
                "language": language,
                "permitted_use": "research",
                "source_url": source_url,
            },
        )
        scenario = Scenario(
            query_seed=seed,
            environment=environment,
            hidden_evaluator=HiddenEvaluatorContext(
                reference_answer=f"evaluator-private-answer-{index}",
                reference_artifacts=[f"private://evaluator-private-artifact-{index}"],
                hidden_tests=[command],
                metadata={"test_patch": patch, "test_patch_sha256": patch_sha},
            ),
            metadata={"source_name": "gold20_public", "source_instance_id": source_id},
        )
        registry.add_scenario(scenario)
        scenarios.append(scenario)
        source_records.append(
            {
                "instance_id": source_id,
                "source_name": "gold20_public",
                "source_url": source_url,
                "type": "pull_request",
                "repo": repository,
                "base_commit": revision,
                "license": "MIT",
                "permitted_use": "research",
                "language": language,
                "problem_statement": public_query,
                "source_uri": environment.source_uri,
                "workspace_original_source_uri": f"https://github.com/{repository}.git",
                "image_digest": environment.image_digest,
                "setup_commands": environment.setup_commands,
                "health_check": environment.health_check,
                "test_patch": patch,
                "test_patch_sha256": patch_sha,
            }
        )
        private_canaries.extend([patch, command, f"evaluator-private-answer-{index}"])

    source_path = root / "source.jsonl"
    source_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in source_records),
        encoding="utf-8",
    )
    runtime_build_specs: dict[str, Path] = {}
    for scenario in scenarios:
        image_digest = scenario.environment.image_digest
        if image_digest in runtime_build_specs:
            continue
        language = _scenario_language(scenario)
        build_spec = root / "runtime-builds" / f"{language}.Dockerfile"
        build_spec.parent.mkdir(parents=True, exist_ok=True)
        build_spec.write_text(
            f"FROM scratch\nLABEL easy_agentic_data.runtime={language}\n",
            encoding="utf-8",
        )
        runtime_build_specs[image_digest] = build_spec
    zero_hash = _text_sha256("")
    rehearsal_results = []
    materialization_records = []
    validation_paths = []
    reference_repair_records = []
    container_replay_records = []
    for index, scenario in enumerate(scenarios):
        patch = scenario.hidden_evaluator.metadata["test_patch"]
        command = scenario.hidden_evaluator.hidden_tests[0]
        tree_hash = _text_sha256(f"tree-{index}")
        repair_patch = (
            f"diff --git a/service-{index}.py b/service-{index}.py\n"
            f"--- a/service-{index}.py\n"
            f"+++ b/service-{index}.py\n"
            "@@ -1 +1 @@\n"
            f"-broken-{index}\n"
            f"+fixed-{index}\n"
        )
        repair_sha256 = _text_sha256(repair_patch)
        private_canaries.append(repair_patch)
        rehearsal_results.append(
            {
                "scenario_id": scenario.scenario_id,
                "environment_id": scenario.environment.environment_id,
                "workspace": str(root / "workspace-private" / str(index)),
                "source_uri": scenario.environment.source_uri,
                "source_revision": scenario.environment.source_revision,
                "test_patch_sha256": _text_sha256(patch),
                "hidden_command_count": 1,
                "patch_check_exit_code": 0,
                "patch_apply_exit_code": 0,
                "hidden_commands_ran": True,
                "commands_run": 1,
                "expected_outcome": "fail",
                "command_outcome": "fail",
                "command_results": [
                    {
                        "command_sha256": _text_sha256(command),
                        "exit_code": 1,
                        "stdout_sha256": zero_hash,
                        "stderr_sha256": zero_hash,
                    }
                ],
                "valid": True,
            }
        )
        materialization_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "environment_id": scenario.environment.environment_id,
                "source_revision": scenario.environment.source_revision,
                "image_digest": scenario.environment.image_digest,
                "setup_commands_sha256": _stable_json_sha256([]),
                "health_check_sha256": _stable_json_sha256(scenario.environment.health_check),
                "attempts": 2,
                "workspace_tree_hashes": [tree_hash, tree_hash],
                "attempt_results": [
                    {
                        "attempt": attempt,
                        "workspace_tree_sha256": tree_hash,
                        "setup_exit_codes": [],
                        "health_check_exit_codes": [0],
                        "valid": True,
                    }
                    for attempt in range(2)
                ],
                "health_checks_passed": True,
                "valid": True,
            }
        )
        validation_path = root / "validations" / f"{index:02d}.json"
        _write_json(
            validation_path,
            {
                "schema_version": HIDDEN_TEST_PATCH_VALIDATION_SCHEMA_VERSION,
                "source_instance_id": _source_id(scenario),
                "scenario_id": scenario.scenario_id,
                "seed_id": scenario.query_seed.seed_id,
                "environment_id": scenario.environment.environment_id,
                "repository": _repository(scenario),
                "source_revision": scenario.environment.source_revision,
                "hidden_test_patch_sha256": _text_sha256(patch),
                "hidden_test_command_sha256": _text_sha256(command),
                "patch_check_exit": 0,
                "patch_apply_exit": 0,
                "original_hidden_test_exit": 1,
                "fixed_hidden_test_exit": 0,
                "repair_check_exit": 0,
                "repair_apply_exit": 0,
                "repaired_patch_check_exit": 0,
                "repaired_patch_apply_exit": 0,
                "workspace_revision_matches": True,
                "materialization_tree_sha256": tree_hash,
                "image_digest": scenario.environment.image_digest,
                "setup_commands_sha256": _stable_json_sha256([]),
                "health_check_sha256": _stable_json_sha256(scenario.environment.health_check),
                "validated_repair_sha256": repair_sha256,
                "valid": True,
            },
        )
        validation_paths.append(validation_path)
        reference_repair_records.append(
            {
                "source_instance_id": _source_id(scenario),
                "scenario_id": scenario.scenario_id,
                "seed_id": scenario.query_seed.seed_id,
                "environment_id": scenario.environment.environment_id,
                "source_revision": scenario.environment.source_revision,
                "repair_patch": repair_patch,
                "repair_patch_sha256": repair_sha256,
            }
        )
        initial_state_sha256 = _text_sha256(f"initial-state-{index}")
        limits = SandboxLimits(**scenario.environment.resource_limits)
        health_result_sha256 = _stable_json_sha256(
            {
                "exit_code": 0,
                "stderr_sha256": zero_hash,
                "stdout_sha256": zero_hash,
                "truncated": False,
            }
        )
        container_replay_records.append(
            {
                "source_instance_id": _source_id(scenario),
                "scenario_id": scenario.scenario_id,
                "seed_id": scenario.query_seed.seed_id,
                "environment_id": scenario.environment.environment_id,
                "scenario_instance_id": ScenarioInstance.materialize(
                    scenario,
                    random_seed=GOLD20_RUNTIME_RANDOM_SEED,
                    initial_state_hash=initial_state_sha256,
                ).instance_id,
                "source_revision": scenario.environment.source_revision,
                "image_digest": scenario.environment.image_digest,
                "runtime_build_spec_sha256": _file_sha256(
                    runtime_build_specs[scenario.environment.image_digest]
                ),
                "sandbox_policy": _container_sandbox_policy(
                    limits,
                    scenario.environment.image_digest,
                ),
                "limits": asdict(limits),
                "base_workspace_tree_sha256": tree_hash,
                "repaired_workspace_tree_sha256": tree_hash,
                "base_initial_state_sha256": initial_state_sha256,
                "repaired_initial_state_sha256": initial_state_sha256,
                "hidden_test_patch_sha256": _text_sha256(patch),
                "hidden_test_command_sha256s": [_text_sha256(command)],
                "validated_repair_sha256": repair_sha256,
                "setup_commands_sha256": _stable_json_sha256([]),
                "health_check_sha256": _stable_json_sha256(scenario.environment.health_check),
                "base_setup_exit_codes": [],
                "repaired_setup_exit_codes": [],
                "base_health_exit_codes": [0],
                "repaired_health_exit_codes": [0],
                "base_post_health_exit_codes": [0],
                "repaired_post_health_exit_codes": [0],
                "base_health_result_sha256s": [health_result_sha256],
                "repaired_health_result_sha256s": [health_result_sha256],
                "base_hidden_patch_exit": 0,
                "repaired_hidden_patch_exit": 0,
                "base_hidden_patch_infrastructure_failure": False,
                "repaired_hidden_patch_infrastructure_failure": False,
                "repair_check_exit": 0,
                "repair_apply_exit": 0,
                "base_hidden_test_exit_codes": [1],
                "repaired_hidden_test_exit_codes": [0],
                "base_hidden_test_infrastructure_failures": [False],
                "repaired_hidden_test_infrastructure_failures": [False],
                "base_hidden_test_result_sha256s": [
                    _stable_json_sha256(
                        {
                            "exit_code": 1,
                            "passed": False,
                            "infrastructure_failure": False,
                        }
                    )
                ],
                "repaired_hidden_test_result_sha256s": [
                    _stable_json_sha256(
                        {
                            "exit_code": 0,
                            "passed": True,
                            "infrastructure_failure": False,
                        }
                    )
                ],
                "valid": True,
            }
        )

    rehearsal_path = root / "rehearsal.json"
    _write_json(
        rehearsal_path,
        {
            "schema_version": REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION,
            "source": {"sha256": _file_sha256(source_path)},
            "import": {
                "imported": 20,
                "scenario_ids": [scenario.scenario_id for scenario in scenarios],
            },
            "hidden_test_patch_rehearsal": {
                "enabled": True,
                "requested": 20,
                "sampled": 20,
                "expected_outcome": "fail",
                "results": rehearsal_results,
                "valid": True,
            },
            "valid": True,
        },
    )
    materialization_path = root / "materialization.json"
    _write_json(
        materialization_path,
        {
            "schema_version": GOLD20_MATERIALIZATION_SCHEMA_VERSION,
            "records": materialization_records,
            "valid": True,
        },
    )
    reference_repair_path = root / "reference-repairs.json"
    _write_json(
        reference_repair_path,
        {
            "schema_version": GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION,
            "records": reference_repair_records,
        },
    )
    container_replay_path = root / "container-replay.json"
    _write_json(
        container_replay_path,
        {
            "schema_version": GOLD20_CONTAINER_REPLAY_SCHEMA_VERSION,
            "producer": {
                "module_sha256": _file_sha256(Path(gold20_runtime.__file__)),
                "sandbox_backend_sha256": _file_sha256(Path(docker_sandbox.__file__)),
                "component_sha256s": {
                    "evaluation.py": _file_sha256(Path(evaluation_module.__file__)),
                    "registry.py": _file_sha256(Path(registry_module.__file__)),
                    "scenarios.py": _file_sha256(Path(scenarios_module.__file__)),
                },
            },
            "execution": {
                "backend": "DockerSandbox",
                "docker_server_version": "test-docker-server",
                "platform": GOLD20_RUNTIME_PLATFORM,
                "random_seed": GOLD20_RUNTIME_RANDOM_SEED,
            },
            "runtime_builds": [
                {
                    "image_digest": image_digest,
                    "image_id": image_digest.rsplit("@", 1)[-1],
                    "platform": GOLD20_RUNTIME_PLATFORM,
                    "image_size_bytes": 1024 + index,
                    "build_spec_sha256": _file_sha256(build_spec),
                    "build_verification_mode": "local_image_id_plus_declared_spec",
                }
                for index, (image_digest, build_spec) in enumerate(
                    sorted(runtime_build_specs.items())
                )
            ],
            "counts": {"records": 20, "valid": 20, "invalid": 0},
            "records": container_replay_records,
            "valid": True,
        },
    )
    holdout_root = root / "holdout"
    holdout_registry = ScenarioRegistry(holdout_root)
    holdout_patch = "diff --git a/holdout b/holdout\n+holdout\n"
    holdout_seed = QuerySeed(
        public=PublicTaskContext(
            query="Held-out benchmark query",
            context={"repository": "owner-0/repo-0", "source_instance_id": "holdout-1"},
        ),
        provenance="swe_bench:holdout-1",
        license="MIT",
        split="evaluation",
        task_family="bug_repair",
        source_method="external_issue_workspace",
        contamination_tags=["benchmark_source"],
        verifier_types=["hidden_test_patch"],
        coverage_tags=["language:python"],
        metadata={
            "source_name": "swe_bench",
            "source_instance_id": "holdout-1",
            "repository": "owner-0/repo-0",
        },
    )
    holdout_environment = EnvironmentSpec(
        name="holdout",
        version="1",
        image_digest=f"sha256:{'a' * 64}",
        source_uri="https://github.com/owner-0/repo-0.git",
        source_revision="b" * 40,
        health_check=["python -c 'pass'"],
        metadata={"repository": "owner-0/repo-0", "language": "python"},
    )
    holdout_registry.add_scenario(
        Scenario(
            query_seed=holdout_seed,
            environment=holdout_environment,
            hidden_evaluator=HiddenEvaluatorContext(
                reference_artifacts=["private://holdout-artifact"],
                hidden_tests=["python -m pytest holdout-only-test"],
                metadata={
                    "test_patch": holdout_patch,
                    "test_patch_sha256": _text_sha256(holdout_patch),
                },
            ),
            metadata={"source_name": "swe_bench", "source_instance_id": "holdout-1"},
        )
    )

    manifest_path = root / "gold20-manifest.json"
    config_path = root / "freeze-config.json"
    _write_json(
        config_path,
        {
            "schema_version": GOLD20_FREEZE_CONFIG_SCHEMA_VERSION,
            "registry_root": str(registry_root),
            "source_snapshot": str(source_path),
            "hidden_patch_rehearsal": str(rehearsal_path),
            "materialization_reset_evidence": str(materialization_path),
            "repair_validation_evidence": [str(path) for path in validation_paths],
            "reference_repair_evidence": str(reference_repair_path),
            "container_runtime_builds": [
                {
                    "image_digest": image_digest,
                    "platform": GOLD20_RUNTIME_PLATFORM,
                    "build_spec": str(build_spec),
                }
                for image_digest, build_spec in sorted(runtime_build_specs.items())
            ],
            "container_replay_evidence": str(container_replay_path),
            "holdout_registry_roots": [str(holdout_root)],
            "resolved_contamination_findings": [
                {
                    "audit": "seed_library",
                    "code": "holdout_repository_overlap",
                    "entry_id": scenario.query_seed.seed_id,
                    "rationale": (
                        "Repository overlap was reviewed; query, provenance, source instance, "
                        "hidden command, reference artifact, and oracle hashes are disjoint."
                    ),
                }
                for scenario in scenarios
                if _repository(scenario) == "owner-0/repo-0"
            ],
            "manifest_output": str(manifest_path),
        },
    )
    return {
        "root": root,
        "config": config_path,
        "registry": registry_root,
        "source": source_path,
        "rehearsal": rehearsal_path,
        "materialization": materialization_path,
        "validations": validation_paths,
        "reference_repairs": reference_repair_path,
        "runtime_build_specs": list(runtime_build_specs.values()),
        "container_replay": container_replay_path,
        "holdout": holdout_root,
        "manifest": manifest_path,
        "private_canaries": private_canaries,
    }


def _source_id(scenario: Scenario) -> str:
    return str(scenario.metadata["source_instance_id"])


def _repository(scenario: Scenario) -> str:
    return str(scenario.query_seed.public.context["repository"])


def _scenario_language(scenario: Scenario) -> str:
    return str(scenario.environment.metadata["language"])


def _container_sandbox_policy(
    limits: SandboxLimits,
    image_digest: str,
) -> dict[str, object]:
    return {
        "image_id": image_digest.rsplit("@", 1)[-1],
        "user": "65532:65532",
        "network_mode": "none",
        "rootfs_read_only": True,
        "privileged": False,
        "workspace_mount_type": "volume",
        "workspace_mount_read_write": True,
        "docker_socket_mounted": False,
        "tmpfs": "rw,noexec,nosuid,size=64m",
        "memory_bytes": 1024**3,
        "nano_cpus": int(limits.cpus * 1_000_000_000),
        "pids_limit": limits.pids,
    }


def _safe_container_replay_record(value: dict[str, object]) -> dict[str, object]:
    return {
        key: value.get(key)
        for key in (
            "source_instance_id",
            "scenario_id",
            "seed_id",
            "environment_id",
            "scenario_instance_id",
            "source_revision",
            "image_digest",
            "runtime_build_spec_sha256",
            "sandbox_policy",
            "limits",
            "base_workspace_tree_sha256",
            "repaired_workspace_tree_sha256",
            "base_initial_state_sha256",
            "repaired_initial_state_sha256",
            "hidden_test_patch_sha256",
            "hidden_test_command_sha256s",
            "validated_repair_sha256",
            "setup_commands_sha256",
            "health_check_sha256",
            "base_setup_exit_codes",
            "repaired_setup_exit_codes",
            "base_health_exit_codes",
            "repaired_health_exit_codes",
            "base_post_health_exit_codes",
            "repaired_post_health_exit_codes",
            "base_health_result_sha256s",
            "repaired_health_result_sha256s",
            "base_hidden_patch_exit",
            "repaired_hidden_patch_exit",
            "base_hidden_patch_infrastructure_failure",
            "repaired_hidden_patch_infrastructure_failure",
            "repair_check_exit",
            "repair_apply_exit",
            "base_hidden_test_exit_codes",
            "repaired_hidden_test_exit_codes",
            "base_hidden_test_infrastructure_failures",
            "repaired_hidden_test_infrastructure_failures",
            "base_hidden_test_result_sha256s",
            "repaired_hidden_test_result_sha256s",
            "valid",
        )
    }


def _issue_codes(manifest: dict[str, object]) -> set[str]:
    return {str(issue["code"]) for issue in manifest["issues"]}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


if __name__ == "__main__":
    unittest.main()
