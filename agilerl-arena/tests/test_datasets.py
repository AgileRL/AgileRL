# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArenaClient dataset methods."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import ArenaFileNotFoundError, ArenaValidationError


@pytest.fixture
def api_key_client():
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        return ArenaClient(api_key="test-key")


def _multipart_text(files: dict, key: str) -> str:
    part = files[key]
    assert part[0] is None
    return part[1]


class TestListDatasets:
    def test_list_by_name(self, api_key_client):
        api_key_client._request = MagicMock(return_value=[{"name": "my-data"}])
        result = api_key_client.list_datasets(name="my-data")
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/datasets",
            params={"name": "my-data"},
        )
        assert result == [{"name": "my-data", "hf_dataset_id": None}]

    def test_search(self, api_key_client):
        api_key_client._request = MagicMock(return_value=[{"hf_dataset_id": "hf/foo"}])
        result = api_key_client.list_datasets(search="countdown")
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/datasets",
            params={"search": "countdown"},
        )
        assert list(result[0].keys()) == ["name", "hf_dataset_id"]
        assert result[0]["name"] is None
        assert result[0]["hf_dataset_id"] == "hf/foo"

    def test_list_no_params(self, api_key_client):
        api_key_client._request = MagicMock(return_value=[])
        api_key_client.list_datasets()
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/datasets",
            params=None,
        )

    def test_list_non_list_response_passthrough(self, api_key_client):
        payload = {"results": [{"name": "hf-ds", "downloads": 100}]}
        api_key_client._request = MagicMock(return_value=payload)
        result = api_key_client.list_datasets(search="countdown")
        assert result is payload

    def test_list_orders_fields(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value=[
                {
                    "id": 1,
                    "category": "sft",
                    "name": "my-data",
                    "hf_dataset_id": "org/ds",
                },
            ],
        )
        result = api_key_client.list_datasets()
        assert list(result[0].keys()) == ["name", "hf_dataset_id", "id", "category"]
        assert result[0]["hf_dataset_id"] == "org/ds"


class TestDatasetExists:
    def test_exists(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={
                "exists": True,
                "id": 7,
                "datasetType": "reasoning",
            }
        )
        result = api_key_client.dataset_exists("my-dataset")
        api_key_client._request.assert_called_once_with(
            "GET",
            "/api/cli/v1/datasets/exists",
            params={"name": "my-dataset"},
        )
        assert result["exists"] is True
        assert result["id"] == 7


class TestCreateDataset:
    def test_create_with_dict_mapping_and_file(self, api_key_client, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        api_key_client._request = MagicMock(return_value={"name": "ds1", "id": 1})

        result = api_key_client.create_dataset(
            name="ds1",
            category="reasoning",
            column_mapping={"prompt": "question"},
            file=csv_path,
        )

        api_key_client._request.assert_called_once()
        call = api_key_client._request.call_args
        assert call[0] == ("POST", "/api/cli/v1/datasets/create")
        assert call[1]["timeout"] == api_key_client._upload_timeout
        files = call[1]["files"]
        assert "data" not in call[1]
        assert _multipart_text(files, "name") == "ds1"
        assert _multipart_text(files, "category") == "reasoning"
        assert json.loads(_multipart_text(files, "column_mapping")) == {
            "prompt": "question",
        }
        assert files["file"][0] == "data.csv"
        assert result["id"] == 1

    def test_create_hf_import(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"name": "hf-ds"})

        api_key_client.create_dataset(
            name="hf-ds",
            category="preference",
            column_mapping='{"prompt": "q"}',
            hf_dataset_name="org/dataset",
            hf_config="default",
            hf_split="train",
        )

        call = api_key_client._request.call_args
        files = call[1]["files"]
        assert "data" not in call[1]
        assert _multipart_text(files, "hf_dataset_name") == "org/dataset"
        assert _multipart_text(files, "hf_config") == "default"
        assert _multipart_text(files, "hf_split") == "train"

    def test_create_logs_success_when_ready_and_uploaded(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={
                "name": "ds1",
                "is_ready": True,
                "uploaded": True,
            },
        )
        with patch("agilerl.arena.client.logger") as mock_logger:
            api_key_client.create_dataset(
                name="ds1",
                category="sft",
                column_mapping={},
            )
        mock_logger.info.assert_called_once_with(
            "Dataset %s created successfully.",
            "ds1",
        )

    def test_create_metadata_only(self, api_key_client):
        api_key_client._request = MagicMock(return_value={"name": "meta"})

        api_key_client.create_dataset(
            name="meta",
            category="sft",
            column_mapping={"text": "content"},
        )

        call = api_key_client._request.call_args
        files = call[1]["files"]
        assert "data" not in call[1]
        assert _multipart_text(files, "name") == "meta"
        assert "file" not in files

    def test_missing_file_raises(self, api_key_client):
        with pytest.raises(ArenaFileNotFoundError, match="Upload file not found"):
            api_key_client.create_dataset(
                name="ds",
                category="reasoning",
                column_mapping={},
                file="/no/such/file.csv",
            )

    def test_invalid_category_raises(self, api_key_client):
        with pytest.raises(ArenaValidationError, match="Invalid dataset category"):
            api_key_client.create_dataset(
                name="ds",
                category="multiturn",
                column_mapping={},
            )


class TestBuildSubmitExperimentMultipart:
    def test_omits_blank_completion(self):
        files = ArenaClient._build_submit_experiment_multipart(
            manifest={"algorithm": {"name": "GRPO"}},
            project="proj",
            resource_id="arena-medium",
            num_nodes=2,
            experiment_name="exp",
            reward_file=b"def reward(q, a, c):\n    return 1.0\n",
            completion=None,
        )
        assert "completion" not in files
        assert files["reward_file"][0] == "reward.py"


class TestBuildCreateDatasetMultipart:
    def test_string_mapping_passthrough(self):
        data, files = ArenaClient._build_create_dataset_multipart(
            name="n",
            category="reasoning",
            column_mapping='{"a": "b"}',
        )
        assert data["column_mapping"] == '{"a": "b"}'
        assert files == {}

    def test_category_normalized_to_lowercase(self):
        data, _ = ArenaClient._build_create_dataset_multipart(
            name="n",
            category="SFT",
            column_mapping={},
        )
        assert data["category"] == "sft"

    def test_bytes_file(self):
        _, files = ArenaClient._build_create_dataset_multipart(
            name="n",
            category="reasoning",
            column_mapping={},
            file=b"col\n1\n",
        )
        assert files["file"] == ("dataset.csv", b"col\n1\n", "text/csv")


class TestDeleteDataset:
    def test_delete_skips_prompt_when_confirmed(self, api_key_client):
        api_key_client._request = MagicMock(
            return_value={"name": "old-ds", "archived": True}
        )
        result = api_key_client.delete_dataset("old-ds", confirm=True)
        api_key_client._request.assert_called_once_with(
            "DELETE",
            "/api/cli/v1/datasets/delete",
            json={"name": "old-ds"},
        )
        assert result["archived"] is True

    def test_delete_prompts_by_default(self, api_key_client):
        api_key_client._request = MagicMock()
        with patch("builtins.input", return_value="n"):
            result = api_key_client.delete_dataset("old-ds")
        api_key_client._request.assert_not_called()
        assert result is None
