# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Call records as Parquet."""

from __future__ import annotations

import pyarrow as pa

__all__ = ["CALL_RECORD_SCHEMA"]

CALL_RECORD_SCHEMA = pa.schema(
    [
        ("call_id", pa.string()),
        ("request_id", pa.string()),
        ("schema_version", pa.int32()),
        ("organization_id", pa.string()),
        ("deployment_id", pa.string()),
        ("instance_id", pa.string()),
        ("user_id", pa.string()),
        ("end_user", pa.string()),
        ("client_session_id", pa.string()),
        ("metadata", pa.string()),
        ("base_model", pa.string()),
        ("checkpoint_id", pa.string()),
        ("model_version", pa.string()),
        ("chat_template_hash", pa.string()),
        ("sampling", pa.string()),
        ("received_at", pa.timestamp("us", tz="UTC")),
        ("queue_ms", pa.int64()),
        ("first_token_ms", pa.int64()),
        ("total_ms", pa.int64()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("cached_tokens", pa.int64()),
        ("finish_reason", pa.string()),
        ("error", pa.string()),
        ("stream", pa.bool_()),
        ("prompt_prefix_hashes", pa.list_(pa.string())),
        ("response_prefix_hash", pa.string()),
        ("messages", pa.string()),
        ("tools", pa.string()),
        ("tool_choice", pa.string()),
        ("response_message", pa.string()),
        ("raw_completion", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("completion_token_ids", pa.list_(pa.int32())),
        ("completion_logprobs", pa.list_(pa.float32())),
    ]
)
"""Arrow schema of a ``calls/`` Parquet file, one row per :class:`CallRecord`."""
