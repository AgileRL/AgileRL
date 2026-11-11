# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Job entrypoint: log a merged checkpoint to the Model Registry and serve it on vLLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import transformers

from agilerl.arena.snowflake.publish import (
    MERGED_EXPORT_REQUIRED,
    PublishDocument,
)
from agilerl.arena.snowflake.session import open_session
from agilerl.arena.snowflake.source import SnowflakeSession

if TYPE_CHECKING:
    from snowflake.ml.model import ModelVersion


def main(argv: list[str] | None = None) -> int:
    """Run the publish request named on the command line.

    :param argv: Arguments after the module name. ``None`` reads ``sys.argv``.
    :return: ``0`` once the model is logged and any service is up.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        msg = (
            "usage: python -m agilerl.arena.snowflake.publish_job "
            "/arena/run/publish-<job>.json"
        )
        raise SystemExit(msg)
    endpoint = run_publish(Path(args[0]))
    if endpoint is not None:
        print(f"endpoint: {endpoint}")
    return 0


def run_publish(
    request_path: Path, session: SnowflakeSession | None = None
) -> str | None:
    """Log the requested checkpoint and start its service when one is named.

    :param request_path: ``publish-<job>.json`` in the run directory on the stage mount.
    :param session: Snowpark session. ``None`` opens the in-job session.
    :return: The service's ingress endpoint, or ``None`` when no service was asked for.
    :raises ValueError: The checkpoint has no ``merged/.complete``.
    """
    request = cast(
        "PublishDocument", json.loads(request_path.read_text(encoding="utf-8"))
    )
    run_dir = request_path.parent
    model_dir = run_dir / request["checkpoint"] / "merged"
    if not (model_dir / ".complete").is_file():
        msg = f"{model_dir} has no .complete. {MERGED_EXPORT_REQUIRED}"
        raise ValueError(msg)
    metrics = checkpoint_metrics(run_dir / "metrics.jsonl", request["step"])
    pipeline = load_text_pipeline(model_dir)
    active = open_session(None) if session is None else session
    try:
        version = log_chat_model(active, pipeline, request, metrics)
        if request["service_name"] is None:
            return None
        return start_vllm_service(version, request)
    finally:
        active.close()


def checkpoint_metrics(path: Path, step: int) -> dict[str, float] | None:
    """Metric values from the last ``metrics.jsonl`` row at ``step``.

    :param path: The run's metrics file.
    :param step: Published checkpoint step, matched against ``agent_<n>/steps``.
    :return: The row's non-null metrics, or ``None`` when no row matches.
    """
    if not path.is_file():
        return None
    rows = [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if any(
            name.endswith("/steps") and value == step
            for name, value in row["metrics"].items()
        )
    ]
    if not rows:
        return None
    # Non-finite values are written as null.
    return {
        name: value for name, value in rows[-1]["metrics"].items() if value is not None
    }


def load_text_pipeline(model_dir: Path) -> transformers.Pipeline:
    """Load the merged export as a text-generation pipeline in its saved dtype.

    The weights stay in host memory, about the model's size in bf16.

    :param model_dir: ``checkpoint_<n>/merged`` on the stage mount.
    :return: The pipeline.
    """
    return transformers.pipeline(
        task="text-generation", model=str(model_dir), dtype="auto"
    )


def log_chat_model(
    session: SnowflakeSession,
    pipeline: transformers.Pipeline,
    request: PublishDocument,
    metrics: dict[str, float] | None,
) -> ModelVersion:
    """Log ``pipeline`` with the OpenAI chat signature vLLM serving needs.

    :param session: Snowpark session.
    :param pipeline: Loaded merged export.
    :param request: Parsed request file.
    :param metrics: Version metrics, or ``None``.
    :return: The new model version.
    """
    # Deferred optional extra: snowflake-job, installed only in the job image.
    from snowflake.ml.model.openai_signatures import OPENAI_CHAT_SIGNATURE

    # Deferred optional extra: snowflake-job, installed only in the job image.
    from snowflake.ml.registry import Registry

    registry = Registry(
        session, database_name=request["database"], schema_name=request["schema"]
    )
    return registry.log_model(
        pipeline,
        model_name=request["model_name"],
        version_name=request["version_name"],
        comment=f"run_id={request['run_id']} checkpoint={request['checkpoint']}",
        metrics=metrics,
        signatures=OPENAI_CHAT_SIGNATURE,
        target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
    )


def start_vllm_service(version: ModelVersion, request: PublishDocument) -> str | None:
    """Create the vLLM service and read its endpoint by service name.

    :param version: Logged model version.
    :param request: Parsed request file with ``service_name`` set.
    :return: The ingress endpoint, or ``None`` while Snowflake provisions it.
    :raises ValueError: The version does not list the new service.
    """
    # Deferred optional extra: snowflake-job, installed only in the job image.
    from snowflake.ml.model.inference_engine import InferenceEngine

    target = f"{request['database']}.{request['schema']}.{request['service_name']}"
    engine_args = []
    if request["gpu_count"] > 1:
        engine_args.append(f"--tensor-parallel-size={request['gpu_count']}")
    if request["max_model_len"] is not None:
        engine_args.append(f"--max-model-len={request['max_model_len']}")
    version.create_service(
        service_name=target,
        service_compute_pool=request["compute_pool"],
        ingress_enabled=True,
        gpu_requests=str(request["gpu_count"]),
        inference_engine_options={
            "engine": InferenceEngine.VLLM,
            "engine_args_override": engine_args,
        },
    )
    for service in version.list_services().to_dict("records"):
        if str(service["name"]).upper() == target.upper():
            return service["inference_endpoint"]
    msg = f"Model version does not list service {target}"
    raise ValueError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
