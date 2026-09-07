# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ArenaClient dataset methods."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agilerl.arena.client import (
    CSV_CONTENT_TYPE,
    PARQUET_CONTENT_TYPE,
    ArenaClient,
)
from agilerl.arena.exceptions import ArenaFileNotFoundError, ArenaValidationError


@pytest.fixture
def api_key_client():
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        return ArenaClient(api_key="test-key")


def _multipart_items(files) -> list[tuple]:
    if isinstance(files, dict):
        return list(files.items())
    return list(files)


def _multipart_text(files, key: str) -> str:
    for name, part in _multipart_items(files):
        if name == key:
            assert part[0] is None
            return part[1]
    raise KeyError(key)


def _file_uploads(files) -> list[tuple]:
    return [part for name, part in _multipart_items(files) if name == "file"]


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
        assert list(result[0].keys()) == ["name", "hf_dataset_id", "category"]
        assert result[0]["hf_dataset_id"] == "org/ds"
        assert "id" not in result[0]


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
        assert result["datasetType"] == "reasoning"
        assert "id" not in result


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
        uploads = _file_uploads(files)
        assert len(uploads) == 1
        assert uploads[0][0] == "data.csv"
        assert uploads[0][2] == CSV_CONTENT_TYPE
        assert result == {"name": "ds1"}
        assert "id" not in result

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
        assert _file_uploads(files) == []

    def test_create_parquet_file(self, api_key_client, tmp_path):
        parquet_path = tmp_path / "train.parquet"
        parquet_path.write_bytes(b"PAR1")
        api_key_client._request = MagicMock(return_value={"name": "ds1"})

        api_key_client.create_dataset(
            name="ds1",
            category="sft",
            column_mapping={"prompt": "text"},
            file=parquet_path,
        )

        files = api_key_client._request.call_args[1]["files"]
        uploads = _file_uploads(files)
        assert len(uploads) == 1
        assert uploads[0][0] == "train.parquet"
        assert uploads[0][2] == PARQUET_CONTENT_TYPE
        assert all(name != "config" for name, _ in _multipart_items(files))

    def test_create_parquet_folder_omits_config_when_one_config(
        self, api_key_client, tmp_path
    ):
        shard = tmp_path / "main" / "train-00000-of-00001.parquet"
        shard.parent.mkdir()
        shard.write_bytes(b"PAR1")
        (tmp_path / "README.md").write_text("skip")
        (tmp_path / "eval.yaml").write_text("skip: true")
        api_key_client._request = MagicMock(return_value={"name": "ds1"})

        api_key_client.create_dataset(
            name="ds1",
            category="reasoning",
            column_mapping={"question": "q", "answer": "a"},
            file=tmp_path,
        )

        files = api_key_client._request.call_args[1]["files"]
        uploads = _file_uploads(files)
        assert len(uploads) == 1
        assert uploads[0][0] == "train-00000-of-00001.parquet"
        assert uploads[0][2] == PARQUET_CONTENT_TYPE
        assert all(name != "config" for name, _ in _multipart_items(files))

    def test_create_parquet_folder_sends_config_when_required(
        self, api_key_client, tmp_path
    ):
        main = tmp_path / "main" / "train-00000-of-00001.parquet"
        socratic = tmp_path / "socratic" / "train-00000-of-00001.parquet"
        main.parent.mkdir()
        socratic.parent.mkdir()
        main.write_bytes(b"MAIN")
        socratic.write_bytes(b"SOC")
        api_key_client._request = MagicMock(return_value={"name": "gsm8k"})

        api_key_client.create_dataset(
            name="gsm8k",
            category="reasoning",
            column_mapping={"question": "q", "answer": "a"},
            file=tmp_path,
            config="main",
        )

        files = api_key_client._request.call_args[1]["files"]
        uploads = _file_uploads(files)
        assert len(uploads) == 1
        assert uploads[0][0] == "main/train-00000-of-00001.parquet"
        assert _multipart_text(files, "config") == "main"

    def test_create_parquet_folder_requires_config_when_multiple(
        self, api_key_client, tmp_path
    ):
        main = tmp_path / "main" / "train.parquet"
        socratic = tmp_path / "socratic" / "train.parquet"
        main.parent.mkdir()
        socratic.parent.mkdir()
        main.write_bytes(b"MAIN")
        socratic.write_bytes(b"SOC")

        with pytest.raises(ArenaValidationError, match="multiple configs"):
            api_key_client.create_dataset(
                name="gsm8k",
                category="reasoning",
                column_mapping={},
                file=tmp_path,
            )

    def test_create_parquet_folder_root_shards(self, api_key_client, tmp_path):
        (tmp_path / "train.parquet").write_bytes(b"PAR1")
        (tmp_path / "valid.parquet").write_bytes(b"PAR2")
        api_key_client._request = MagicMock(return_value={"name": "ds1"})

        api_key_client.create_dataset(
            name="ds1",
            category="sft",
            column_mapping={"prompt": "text"},
            file=tmp_path,
        )

        uploads = _file_uploads(api_key_client._request.call_args[1]["files"])
        names = [part[0] for part in uploads]
        assert names == ["train.parquet", "valid.parquet"]
        assert all(part[2] == PARQUET_CONTENT_TYPE for part in uploads)
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

    def test_create_tabular_category_raises(self, api_key_client):
        api_key_client._request = MagicMock()
        with pytest.raises(ArenaValidationError, match="Invalid dataset category"):
            api_key_client.create_dataset(
                name="ds1",
                category="tabular",
                column_mapping={"feature": "x", "target": "y"},
            )
        api_key_client._request.assert_not_called()

    def test_nested_hf_folder_uses_inner_config_names(self, api_key_client, tmp_path):
        main = tmp_path / "gsm8k" / "main" / "train.parquet"
        socratic = tmp_path / "gsm8k" / "socratic" / "train.parquet"
        main.parent.mkdir(parents=True)
        socratic.parent.mkdir(parents=True)
        main.write_bytes(b"MAIN")
        socratic.write_bytes(b"SOC")
        api_key_client._request = MagicMock(return_value={"name": "gsm8k"})

        api_key_client.create_dataset(
            name="gsm8k",
            category="reasoning",
            column_mapping={"question": "q", "answer": "a"},
            file=tmp_path,
            config="main",
        )

        files = api_key_client._request.call_args[1]["files"]
        uploads = _file_uploads(files)
        assert len(uploads) == 1
        assert uploads[0][0] == "main/train.parquet"
        assert uploads[0][2] == PARQUET_CONTENT_TYPE
        assert _multipart_text(files, "config") == "main"

    def test_nested_hf_folder_requires_config_when_multiple(
        self, api_key_client, tmp_path
    ):
        main = tmp_path / "gsm8k" / "main" / "train.parquet"
        socratic = tmp_path / "gsm8k" / "socratic" / "train.parquet"
        main.parent.mkdir(parents=True)
        socratic.parent.mkdir(parents=True)
        main.write_bytes(b"MAIN")
        socratic.write_bytes(b"SOC")

        with pytest.raises(ArenaValidationError, match="multiple configs"):
            api_key_client.create_dataset(
                name="gsm8k",
                category="reasoning",
                column_mapping={},
                file=tmp_path,
            )

    def test_single_nested_hf_folder_strips_to_filename(self, api_key_client, tmp_path):
        shard = tmp_path / "gsm8k" / "main" / "train.parquet"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"PAR1")
        api_key_client._request = MagicMock(return_value={"name": "gsm8k"})

        api_key_client.create_dataset(
            name="gsm8k",
            category="reasoning",
            column_mapping={"question": "q", "answer": "a"},
            file=tmp_path,
        )

        files = api_key_client._request.call_args[1]["files"]
        uploads = _file_uploads(files)
        assert uploads[0][0] == "train.parquet"
        assert all(name != "config" for name, _ in _multipart_items(files))

    def test_create_parquet_split_directories_requires_config(
        self, api_key_client, tmp_path
    ):
        train = tmp_path / "main" / "train" / "data.parquet"
        test = tmp_path / "main" / "test" / "data.parquet"
        train.parent.mkdir(parents=True)
        test.parent.mkdir(parents=True)
        train.write_bytes(b"TR")
        test.write_bytes(b"TEST")

        with pytest.raises(ArenaValidationError, match="multiple configs"):
            api_key_client.create_dataset(
                name="ds1",
                category="reasoning",
                column_mapping={"question": "q", "answer": "a"},
                file=tmp_path,
            )

    def test_create_parquet_split_directories_selects_named_config(
        self, api_key_client, tmp_path
    ):
        train = tmp_path / "main" / "train" / "data.parquet"
        test = tmp_path / "main" / "test" / "data.parquet"
        train.parent.mkdir(parents=True)
        test.parent.mkdir(parents=True)
        train.write_bytes(b"TR")
        test.write_bytes(b"TEST")
        api_key_client._request = MagicMock(return_value={"name": "ds1"})

        api_key_client.create_dataset(
            name="ds1",
            category="reasoning",
            column_mapping={"question": "q", "answer": "a"},
            file=tmp_path,
            config="train",
        )

        files = api_key_client._request.call_args[1]["files"]
        uploads = _file_uploads(files)
        names = [part[0] for part in uploads]
        assert names == ["train/data.parquet"]
        assert uploads[0][2] == PARQUET_CONTENT_TYPE
        assert _multipart_text(files, "config") == "train"


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
        assert files == []

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
        assert files == [("file", ("dataset.csv", b"col\n1\n", CSV_CONTENT_TYPE))]

    def test_parquet_file_uses_parquet_content_type(self, tmp_path):
        path = tmp_path / "split.parquet"
        path.write_bytes(b"PAR1")
        _, files = ArenaClient._build_create_dataset_multipart(
            name="n",
            category="sft",
            column_mapping={},
            file=path,
        )
        try:
            assert files[0][0] == "file"
            assert files[0][1][0] == "split.parquet"
            assert files[0][1][2] == PARQUET_CONTENT_TYPE
        finally:
            ArenaClient._close_upload_files(files)

    def test_empty_parquet_folder_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "README.md").write_text("no shards")
        with pytest.raises(ArenaValidationError, match="No parquet files found"):
            ArenaClient._build_create_dataset_multipart(
                name="n",
                category="sft",
                column_mapping={},
                file=empty,
            )

    def test_csv_only_folder_raises(self, tmp_path):
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir()
        (csv_dir / "data.csv").write_text("a,b\n1,2\n")
        with pytest.raises(ArenaValidationError, match="No parquet files found"):
            ArenaClient._build_create_dataset_multipart(
                name="n",
                category="sft",
                column_mapping={},
                file=csv_dir,
            )

    def test_uppercase_parquet_suffix(self, tmp_path):
        path = tmp_path / "train.PARQUET"
        path.write_bytes(b"PAR1")
        _, files = ArenaClient._build_create_dataset_multipart(
            name="n",
            category="sft",
            column_mapping={},
            file=path,
        )
        try:
            assert files[0][1][2] == PARQUET_CONTENT_TYPE
        finally:
            ArenaClient._close_upload_files(files)

    def test_config_names_strip_shared_dataset_prefix(self, tmp_path):
        main = tmp_path / "gsm8k" / "main" / "train.parquet"
        socratic = tmp_path / "gsm8k" / "socratic" / "train.parquet"
        main.parent.mkdir(parents=True)
        socratic.parent.mkdir(parents=True)
        main.write_bytes(b"MAIN")
        socratic.write_bytes(b"SOC")
        shards = ArenaClient._parquet_shard_paths(tmp_path)

        names = ArenaClient._parquet_config_names(tmp_path, shards)

        assert names == ["main", "socratic"]

    def test_config_names_split_directories_are_configs(self, tmp_path):
        train = tmp_path / "main" / "train" / "data.parquet"
        test = tmp_path / "main" / "test" / "data.parquet"
        train.parent.mkdir(parents=True)
        test.parent.mkdir(parents=True)
        train.write_bytes(b"TR")
        test.write_bytes(b"TEST")
        shards = ArenaClient._parquet_shard_paths(tmp_path)

        names = ArenaClient._parquet_config_names(tmp_path, shards)

        assert names == ["test", "train"]

    def test_config_names_wrapping_folder_split_directories_are_configs(self, tmp_path):
        train = tmp_path / "gsm8k" / "main" / "train" / "data.parquet"
        test = tmp_path / "gsm8k" / "main" / "test" / "data.parquet"
        train.parent.mkdir(parents=True)
        test.parent.mkdir(parents=True)
        train.write_bytes(b"TR")
        test.write_bytes(b"TEST")
        shards = ArenaClient._parquet_shard_paths(tmp_path)

        names = ArenaClient._parquet_config_names(tmp_path, shards)

        assert names == ["test", "train"]

    def test_config_names_flat_shards_use_default(self, tmp_path):
        (tmp_path / "train.parquet").write_bytes(b"A")
        (tmp_path / "test.parquet").write_bytes(b"B")
        shards = ArenaClient._parquet_shard_paths(tmp_path)

        names = ArenaClient._parquet_config_names(tmp_path, shards)

        assert names == ["default"]

    def test_flat_shards_accept_default_config(self, tmp_path):
        (tmp_path / "train.parquet").write_bytes(b"A")
        (tmp_path / "test.parquet").write_bytes(b"B")
        _, files = ArenaClient._build_create_dataset_multipart(
            name="n",
            category="sft",
            column_mapping={},
            file=tmp_path,
            config="default",
        )
        try:
            names = [part[1][0] for part in files]
            assert names == ["test.parquet", "train.parquet"]
        finally:
            ArenaClient._close_upload_files(files)

    def test_config_names_keep_distinct_nested_roots(self, tmp_path):
        alpha = tmp_path / "alpha" / "extra" / "train.parquet"
        beta = tmp_path / "beta" / "extra" / "train.parquet"
        alpha.parent.mkdir(parents=True)
        beta.parent.mkdir(parents=True)
        alpha.write_bytes(b"A")
        beta.write_bytes(b"B")
        shards = ArenaClient._parquet_shard_paths(tmp_path)

        names = ArenaClient._parquet_config_names(tmp_path, shards)

        assert names == ["alpha", "beta"]

    def test_unknown_config_raises(self, tmp_path):
        shard = tmp_path / "main" / "train.parquet"
        shard.parent.mkdir()
        shard.write_bytes(b"PAR1")
        with pytest.raises(ArenaValidationError, match="No parquet files for config"):
            ArenaClient._build_create_dataset_multipart(
                name="n",
                category="sft",
                column_mapping={},
                file=tmp_path,
                config="socratic",
            )


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
