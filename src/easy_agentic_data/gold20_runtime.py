from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from easy_agentic_data.evaluation import HiddenCommandEvaluator, HiddenTestPatchEvaluator
from easy_agentic_data.gold20 import (
    EXPECTED_SEED_COUNT,
    GOLD20_FREEZE_CONFIG_SCHEMA_VERSION,
    GOLD20_MATERIALIZATION_SCHEMA_VERSION,
    GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION,
)
from easy_agentic_data.registry import (
    ScenarioRegistry,
    _workspace_tree_hash,
    materialize_environment_source,
)
from easy_agentic_data.sandbox import DockerSandbox, SandboxLimits
from easy_agentic_data.scenarios import Scenario

GOLD20_CONTAINER_REPLAY_SCHEMA_VERSION = "easy_agentic_data.gold20_container_replay.v1"
GOLD20_RUNTIME_PLATFORM = "linux/arm64"
GOLD20_RUNTIME_RANDOM_SEED = 0
GOLD20_BUILD_VERIFICATION_MODE = "local_image_id_plus_declared_spec"


def replay_gold20_runtime(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Replay the Gold-20 repair oracle through the production Docker sandbox."""

    config_file = Path(config_path).expanduser().resolve()
    config = _read_object(config_file)
    if config.get("schema_version") != GOLD20_FREEZE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Gold-20 config must use schema version {GOLD20_FREEZE_CONFIG_SCHEMA_VERSION}"
        )
    config_dir = config_file.parent
    registry_root = _required_path(config, "registry_root", config_dir)
    materialization_path = _required_path(
        config,
        "materialization_reset_evidence",
        config_dir,
    )
    reference_repair_path = _required_path(
        config,
        "reference_repair_evidence",
        config_dir,
    )
    output_path = _resolve_output_path(config, config_dir, output)
    runtime_builds = _load_runtime_builds(config, config_dir)

    registry = ScenarioRegistry(registry_root)
    scenarios = [
        registry.get_scenario(row["scenario_id"])
        for row in registry.list_scenarios()
    ]
    if len(scenarios) != EXPECTED_SEED_COUNT:
        raise ValueError(
            f"Gold-20 runtime replay requires exactly {EXPECTED_SEED_COUNT} scenarios"
        )
    source_ids = {_source_instance_id(scenario) for scenario in scenarios}
    if "" in source_ids or len(source_ids) != EXPECTED_SEED_COUNT:
        raise ValueError("Gold-20 runtime replay requires unique source instance IDs")

    materialization = _read_object(materialization_path)
    if materialization.get("schema_version") != GOLD20_MATERIALIZATION_SCHEMA_VERSION:
        raise ValueError("Gold-20 materialization evidence schema is invalid")
    if materialization.get("valid") is not True:
        raise ValueError("Gold-20 materialization evidence is not valid")
    materialization_records = _index_records(
        materialization.get("records"),
        "scenario_id",
    )

    repairs = _read_object(reference_repair_path)
    if repairs.get("schema_version") != GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION:
        raise ValueError("Gold-20 reference repair evidence schema is invalid")
    repair_records = _index_records(repairs.get("records"), "source_instance_id")
    if set(repair_records) != source_ids:
        raise ValueError("Gold-20 runtime replay repair set does not match the registry")

    image_metadata = {
        build["image_digest"]: _inspect_runtime_image(build)
        for build in runtime_builds
    }
    scenario_images = {scenario.environment.image_digest for scenario in scenarios}
    if set(image_metadata) != scenario_images:
        raise ValueError("Gold-20 runtime build image set does not match the registry")

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ead-gold20-runtime-") as directory:
        replay_root = Path(directory)
        for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
            source_id = _source_instance_id(scenario)
            materialization_record = materialization_records.get(scenario.scenario_id)
            repair_record = repair_records.get(source_id)
            if materialization_record is None or repair_record is None:
                raise ValueError(f"Gold-20 runtime replay evidence is missing {source_id}")
            record = _replay_scenario(
                registry=registry,
                scenario=scenario,
                source_id=source_id,
                materialization_record=materialization_record,
                repair_record=repair_record,
                replay_root=replay_root,
                image_metadata=image_metadata[scenario.environment.image_digest],
            )
            records.append(record)

    valid_count = sum(record["valid"] is True for record in records)
    payload = {
        "schema_version": GOLD20_CONTAINER_REPLAY_SCHEMA_VERSION,
        "producer": {
            "module_sha256": _file_sha256(Path(__file__)),
            "sandbox_backend_sha256": _file_sha256(
                Path(__file__).with_name("sandbox") / "docker.py"
            ),
            "component_sha256s": _producer_component_sha256s(),
        },
        "execution": {
            "backend": "DockerSandbox",
            "docker_server_version": _docker_server_version(),
            "platform": GOLD20_RUNTIME_PLATFORM,
            "random_seed": GOLD20_RUNTIME_RANDOM_SEED,
        },
        "runtime_builds": sorted(
            image_metadata.values(),
            key=lambda item: item["image_digest"],
        ),
        "counts": {
            "records": len(records),
            "valid": valid_count,
            "invalid": len(records) - valid_count,
        },
        "records": records,
        "valid": len(records) == EXPECTED_SEED_COUNT
        and valid_count == EXPECTED_SEED_COUNT,
    }
    if payload["valid"]:
        _write_json_atomically(output_path, payload)
    return payload


def _replay_scenario(
    *,
    registry: ScenarioRegistry,
    scenario: Scenario,
    source_id: str,
    materialization_record: dict[str, Any],
    repair_record: dict[str, Any],
    replay_root: Path,
    image_metadata: dict[str, Any],
) -> dict[str, Any]:
    _require_supported_scenario(scenario)
    expected_tree = _stable_materialization_tree(materialization_record)
    repair_patch = str(repair_record.get("repair_patch") or "")
    hidden_patch = str(scenario.hidden_evaluator.metadata.get("test_patch") or "")
    limits = SandboxLimits(**scenario.environment.resource_limits)
    scenario_root = replay_root / scenario.scenario_id
    base_source = materialize_environment_source(
        scenario.environment,
        scenario_root / "base",
        run_health_checks=False,
    )
    repaired_source = materialize_environment_source(
        scenario.environment,
        scenario_root / "repaired",
        run_health_checks=False,
    )
    base_tree = _workspace_tree_hash(base_source)
    repaired_tree = _workspace_tree_hash(repaired_source)

    base = _execute_variant(
        registry=registry,
        scenario=scenario,
        source=base_source,
        limits=limits,
        repair_patch=None,
    )
    repaired = _execute_variant(
        registry=registry,
        scenario=scenario,
        source=repaired_source,
        limits=limits,
        repair_patch=repair_patch,
    )
    repair_sha256 = _text_sha256(repair_patch)
    command_hashes = sorted(
        _text_sha256(command) for command in scenario.hidden_evaluator.hidden_tests
    )
    policy_expected = _expected_sandbox_policy(limits, image_metadata["image_digest"])
    valid = (
        bool(repair_patch)
        and repair_record.get("repair_patch_sha256") == repair_sha256
        and repair_record.get("scenario_id") == scenario.scenario_id
        and repair_record.get("seed_id") == scenario.query_seed.seed_id
        and repair_record.get("environment_id") == scenario.environment.environment_id
        and repair_record.get("source_revision") == scenario.environment.source_revision
        and base_tree == repaired_tree == expected_tree
        and base["initial_state_sha256"] == repaired["initial_state_sha256"]
        and base["sandbox_policy"] == repaired["sandbox_policy"] == policy_expected
        and _all_zero(base["setup_exit_codes"])
        and _all_zero(repaired["setup_exit_codes"])
        and _all_zero(base["health_exit_codes"])
        and _all_zero(repaired["health_exit_codes"])
        and _all_zero(base["post_health_exit_codes"])
        and _all_zero(repaired["post_health_exit_codes"])
        and base["health_result_sha256s"] == repaired["health_result_sha256s"]
        and base["hidden_patch_exit"] == 0
        and repaired["hidden_patch_exit"] == 0
        and base["hidden_patch_infrastructure_failure"] is False
        and repaired["hidden_patch_infrastructure_failure"] is False
        and base["repair_check_exit"] is None
        and base["repair_apply_exit"] is None
        and repaired["repair_check_exit"] == 0
        and repaired["repair_apply_exit"] == 0
        and len(base["hidden_test_exit_codes"])
        == len(repaired["hidden_test_exit_codes"])
        == len(scenario.hidden_evaluator.hidden_tests)
        and any(code != 0 for code in base["hidden_test_exit_codes"])
        and _all_zero(repaired["hidden_test_exit_codes"])
        and not any(base["hidden_test_infrastructure_failures"])
        and not any(repaired["hidden_test_infrastructure_failures"])
        and image_metadata["image_digest"] == scenario.environment.image_digest
        and image_metadata["platform"] == GOLD20_RUNTIME_PLATFORM
    )
    return {
        "source_instance_id": source_id,
        "scenario_id": scenario.scenario_id,
        "seed_id": scenario.query_seed.seed_id,
        "environment_id": scenario.environment.environment_id,
        "scenario_instance_id": base["scenario_instance_id"],
        "source_revision": scenario.environment.source_revision,
        "image_digest": scenario.environment.image_digest,
        "runtime_build_spec_sha256": image_metadata["build_spec_sha256"],
        "sandbox_policy": base["sandbox_policy"],
        "limits": asdict(limits),
        "base_workspace_tree_sha256": base_tree,
        "repaired_workspace_tree_sha256": repaired_tree,
        "base_initial_state_sha256": base["initial_state_sha256"],
        "repaired_initial_state_sha256": repaired["initial_state_sha256"],
        "hidden_test_patch_sha256": _text_sha256(hidden_patch),
        "hidden_test_command_sha256s": command_hashes,
        "validated_repair_sha256": repair_sha256,
        "setup_commands_sha256": _stable_json_sha256(scenario.environment.setup_commands),
        "health_check_sha256": _stable_json_sha256(scenario.environment.health_check),
        "base_setup_exit_codes": base["setup_exit_codes"],
        "repaired_setup_exit_codes": repaired["setup_exit_codes"],
        "base_health_exit_codes": base["health_exit_codes"],
        "repaired_health_exit_codes": repaired["health_exit_codes"],
        "base_post_health_exit_codes": base["post_health_exit_codes"],
        "repaired_post_health_exit_codes": repaired["post_health_exit_codes"],
        "base_health_result_sha256s": base["health_result_sha256s"],
        "repaired_health_result_sha256s": repaired["health_result_sha256s"],
        "base_hidden_patch_exit": base["hidden_patch_exit"],
        "repaired_hidden_patch_exit": repaired["hidden_patch_exit"],
        "base_hidden_patch_infrastructure_failure": base[
            "hidden_patch_infrastructure_failure"
        ],
        "repaired_hidden_patch_infrastructure_failure": repaired[
            "hidden_patch_infrastructure_failure"
        ],
        "repair_check_exit": repaired["repair_check_exit"],
        "repair_apply_exit": repaired["repair_apply_exit"],
        "base_hidden_test_exit_codes": base["hidden_test_exit_codes"],
        "repaired_hidden_test_exit_codes": repaired["hidden_test_exit_codes"],
        "base_hidden_test_infrastructure_failures": base[
            "hidden_test_infrastructure_failures"
        ],
        "repaired_hidden_test_infrastructure_failures": repaired[
            "hidden_test_infrastructure_failures"
        ],
        "base_hidden_test_result_sha256s": base["hidden_test_result_sha256s"],
        "repaired_hidden_test_result_sha256s": repaired[
            "hidden_test_result_sha256s"
        ],
        "valid": valid,
    }


def _execute_variant(
    *,
    registry: ScenarioRegistry,
    scenario: Scenario,
    source: Path,
    limits: SandboxLimits,
    repair_patch: str | None,
) -> dict[str, Any]:
    sandbox = DockerSandbox(
        image_digest=scenario.environment.image_digest,
        source_directory=source,
        limits=limits,
        network_enabled=False,
    )
    sandbox.create()
    try:
        policy = _inspect_sandbox_policy(sandbox)
        setup_results = [
            _command_result(sandbox.execute_as_root(shlex.split(command)))
            for command in scenario.environment.setup_commands
        ]
        health_results = [
            _command_result(sandbox.execute(shlex.split(command)))
            for command in scenario.environment.health_check
        ]
        initial_state = sandbox.state_hash()
        instance = registry.materialize(
            scenario.scenario_id,
            random_seed=GOLD20_RUNTIME_RANDOM_SEED,
            initial_state_hash=initial_state,
        )
        repair_check_exit: int | None = None
        repair_apply_exit: int | None = None
        if repair_patch is not None:
            sandbox.write(".ead_reference_repair.patch", repair_patch)
            repair_check_exit = sandbox.execute(
                ["git", "apply", "--check", ".ead_reference_repair.patch"]
            ).exit_code
            repair_apply_exit = sandbox.execute(
                ["git", "apply", ".ead_reference_repair.patch"]
            ).exit_code
        patch_result = HiddenTestPatchEvaluator().evaluate(sandbox, instance)
        hidden_results = [
            HiddenCommandEvaluator(shlex.split(command)).evaluate(sandbox, instance)
            for command in scenario.hidden_evaluator.hidden_tests
        ]
        post_health_results = [
            _command_result(sandbox.execute(shlex.split(command)))
            for command in scenario.environment.health_check
        ]
        return {
            "scenario_instance_id": instance.instance_id,
            "sandbox_policy": policy,
            "initial_state_sha256": initial_state,
            "setup_exit_codes": [result["exit_code"] for result in setup_results],
            "health_exit_codes": [result["exit_code"] for result in health_results],
            "post_health_exit_codes": [
                result["exit_code"] for result in post_health_results
            ],
            "health_result_sha256s": [
                result["result_sha256"] for result in health_results
            ],
            "repair_check_exit": repair_check_exit,
            "repair_apply_exit": repair_apply_exit,
            "hidden_patch_exit": int(patch_result.evidence.get("exit_code", -1)),
            "hidden_patch_infrastructure_failure": patch_result.infrastructure_failure,
            "hidden_test_exit_codes": [
                int(result.evidence.get("exit_code", -1)) for result in hidden_results
            ],
            "hidden_test_infrastructure_failures": [
                result.infrastructure_failure for result in hidden_results
            ],
            "hidden_test_result_sha256s": [
                _hidden_test_result_sha256(
                    exit_code=int(result.evidence.get("exit_code", -1)),
                    infrastructure_failure=result.infrastructure_failure,
                )
                for result in hidden_results
            ],
        }
    finally:
        sandbox.destroy()


def _command_result(result: Any) -> dict[str, Any]:
    payload = {
        "exit_code": int(result.exit_code),
        "stdout_sha256": _text_sha256(result.stdout),
        "stderr_sha256": _text_sha256(result.stderr),
        "truncated": bool(result.truncated),
    }
    payload["result_sha256"] = _stable_json_sha256(payload)
    return payload


def _inspect_sandbox_policy(sandbox: DockerSandbox) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "inspect", sandbox.container_name],
        text=True,
        capture_output=True,
        check=True,
    )
    values = json.loads(completed.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError("Docker inspect returned an unexpected container payload")
    value = values[0]
    host = value["HostConfig"]
    mounts = value.get("Mounts", [])
    workspace_mount = next(
        (
            mount
            for mount in mounts
            if isinstance(mount, dict) and mount.get("Destination") == "/workspace"
        ),
        {},
    )
    socket_mounted = any(
        isinstance(mount, dict)
        and mount.get("Destination") == "/var/run/docker.sock"
        for mount in mounts
    )
    return {
        "image_id": str(value.get("Image") or ""),
        "user": str(value.get("Config", {}).get("User") or ""),
        "network_mode": str(host.get("NetworkMode") or ""),
        "rootfs_read_only": host.get("ReadonlyRootfs") is True,
        "privileged": host.get("Privileged") is True,
        "workspace_mount_type": str(workspace_mount.get("Type") or ""),
        "workspace_mount_read_write": workspace_mount.get("RW") is True,
        "docker_socket_mounted": socket_mounted,
        "tmpfs": str((host.get("Tmpfs") or {}).get("/tmp") or ""),
        "memory_bytes": int(host.get("Memory") or 0),
        "nano_cpus": int(host.get("NanoCpus") or 0),
        "pids_limit": int(host.get("PidsLimit") or 0),
    }


def _expected_sandbox_policy(limits: SandboxLimits, image_digest: str) -> dict[str, Any]:
    return {
        "image_id": image_digest,
        "user": "65532:65532",
        "network_mode": "none",
        "rootfs_read_only": True,
        "privileged": False,
        "workspace_mount_type": "volume",
        "workspace_mount_read_write": True,
        "docker_socket_mounted": False,
        "tmpfs": "rw,noexec,nosuid,size=64m",
        "memory_bytes": _memory_bytes(limits.memory),
        "nano_cpus": int(limits.cpus * 1_000_000_000),
        "pids_limit": limits.pids,
    }


def _load_runtime_builds(
    config: dict[str, Any],
    config_dir: Path,
) -> list[dict[str, Any]]:
    raw = config.get("container_runtime_builds")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Gold-20 config requires container_runtime_builds")
    builds = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Gold-20 container runtime build entries must be objects")
        image = str(item.get("image_digest") or "").strip()
        platform = str(item.get("platform") or "").strip()
        build_spec_value = str(item.get("build_spec") or "").strip()
        if not image or image in seen or platform != GOLD20_RUNTIME_PLATFORM:
            raise ValueError("Gold-20 container runtime build identity is invalid")
        build_spec = _resolve_path(config_dir, build_spec_value)
        if not build_spec.is_file():
            raise ValueError(f"Gold-20 runtime build spec does not exist: {build_spec}")
        seen.add(image)
        builds.append(
            {
                "image_digest": image,
                "platform": platform,
                "build_spec": build_spec,
                "build_spec_sha256": _file_sha256(build_spec),
            }
        )
    return builds


def _inspect_runtime_image(build: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "image", "inspect", build["image_digest"]],
        text=True,
        capture_output=True,
        check=True,
    )
    values = json.loads(completed.stdout)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError("Docker inspect returned an unexpected image payload")
    value = values[0]
    actual_id = str(value.get("Id") or "")
    platform = f"{value.get('Os')}/{value.get('Architecture')}"
    image_size = int(value.get("Size") or 0)
    if actual_id != build["image_digest"] or platform != build["platform"]:
        raise ValueError("Gold-20 runtime image identity does not match its declaration")
    if image_size <= 0:
        raise ValueError("Gold-20 runtime image size is invalid")
    return {
        "image_digest": actual_id,
        "image_id": actual_id,
        "platform": platform,
        "image_size_bytes": image_size,
        "build_spec_sha256": build["build_spec_sha256"],
        "build_verification_mode": GOLD20_BUILD_VERIFICATION_MODE,
    }


def _require_supported_scenario(scenario: Scenario) -> None:
    evaluator = scenario.hidden_evaluator
    if scenario.environment.network_policy != "disabled":
        raise ValueError("Gold-20 runtime replay requires disabled network access")
    if (
        evaluator.required_state
        or evaluator.forbidden_state
        or evaluator.metadata.get("retrieval_requirements")
        or evaluator.metadata.get("trace_quality_rubric")
    ):
        raise ValueError("Gold-20 runtime replay has an unsupported evaluator surface")


def _producer_component_sha256s() -> dict[str, str]:
    package_root = Path(__file__).parent
    return {
        name: _file_sha256(package_root / name)
        for name in ("evaluation.py", "registry.py", "scenarios.py")
    }


def _hidden_test_result_sha256(
    *,
    exit_code: int,
    infrastructure_failure: bool,
) -> str:
    return _stable_json_sha256(
        {
            "exit_code": exit_code,
            "passed": exit_code == 0 and not infrastructure_failure,
            "infrastructure_failure": infrastructure_failure,
        }
    )


def _docker_server_version() -> str:
    completed = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        text=True,
        capture_output=True,
        check=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError("Docker server version is unavailable")
    return value


def _stable_materialization_tree(record: dict[str, Any]) -> str:
    values = record.get("workspace_tree_hashes")
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("Gold-20 materialization record lacks reset hashes")
    hashes = [str(value) for value in values]
    if len(set(hashes)) != 1:
        raise ValueError("Gold-20 materialization reset hashes do not match")
    return hashes[0]


def _index_records(value: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Gold-20 runtime replay records must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Gold-20 runtime replay records must be objects")
        item_id = str(item.get(key) or "")
        if not item_id or item_id in indexed:
            raise ValueError(f"Gold-20 runtime replay has an invalid {key}")
        indexed[item_id] = item
    return indexed


def _source_instance_id(scenario: Scenario) -> str:
    return str(scenario.metadata.get("source_instance_id") or "").strip()


def _all_zero(values: Any) -> bool:
    return isinstance(values, list) and all(value == 0 for value in values)


def _memory_bytes(value: str) -> int:
    normalized = value.strip().lower()
    multipliers = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
    suffix = normalized[-1]
    if suffix not in multipliers:
        raise ValueError(f"Unsupported Docker memory limit: {value}")
    return int(float(normalized[:-1]) * multipliers[suffix])


def _required_path(config: dict[str, Any], key: str, config_dir: Path) -> Path:
    value = str(config.get(key) or "").strip()
    if not value:
        raise ValueError(f"Gold-20 config requires {key}")
    path = _resolve_path(config_dir, value)
    if not path.exists():
        raise ValueError(f"Gold-20 input does not exist for {key}: {path}")
    return path


def _resolve_output_path(
    config: dict[str, Any],
    config_dir: Path,
    output: str | Path | None,
) -> Path:
    value = output or config.get("container_replay_evidence")
    if not value:
        raise ValueError("Gold-20 config requires container_replay_evidence")
    return _resolve_path(config_dir, value)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)
