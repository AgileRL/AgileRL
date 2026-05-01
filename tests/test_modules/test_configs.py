"""Tests for agilerl.modules.configs."""

import tempfile
from pathlib import Path

import pytest

from agilerl.modules.configs import MlpNetConfig


class TestNetConfigFromDict:
    def test_net_config_from_dict(self):
        """Covers NetConfig.from_dict."""
        config = MlpNetConfig.from_dict({"hidden_size": [64, 64]})
        assert config.hidden_size == [64, 64]


class TestNetConfigFromYaml:
    def test_net_config_from_yaml(self):
        """Covers NetConfig.from_yaml with valid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("NET_CONFIG:\n  hidden_size: [32, 32]\n")
            path = f.name
        try:
            config = MlpNetConfig.from_yaml(path)
            assert config.hidden_size == [32, 32]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_net_config_from_yaml_missing_net_config(self):
        """Covers NetConfig.from_yaml when NET_CONFIG not in file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("other_key: value\n")
            path = f.name
        try:
            with pytest.raises(AssertionError, match="NET_CONFIG not found"):
                MlpNetConfig.from_yaml(path)
        finally:
            Path(path).unlink(missing_ok=True)


class TestNetConfigGetitemSetitem:
    def test_net_config_getitem_setitem(self):
        """Covers __getitem__ and __setitem__."""
        config = MlpNetConfig(hidden_size=[64, 64])
        assert config["hidden_size"] == [64, 64]
        config["hidden_size"] = [32, 32]
        assert config.hidden_size == [32, 32]


class TestNetConfigContains:
    def test_net_config_contains(self):
        """Covers __contains__."""
        config = MlpNetConfig(hidden_size=[64, 64])
        assert "hidden_size" in config
        assert "missing" not in config


class TestNetConfigGet:
    def test_net_config_get(self):
        """Covers get with existing and default."""
        config = MlpNetConfig(hidden_size=[64, 64])
        assert config.get("hidden_size") == [64, 64]
        assert config.get("missing") is None
        assert config.get("missing", 42) == 42


class TestNetConfigPop:
    def test_net_config_pop_missing(self):
        """Covers pop when attr missing, returns default."""
        config = MlpNetConfig(hidden_size=[64, 64])
        val = config.pop("missing", None)
        assert val is None


class TestNetConfigKeysValuesItems:
    def test_net_config_keys_values_items(self):
        """Covers keys, values, items."""
        config = MlpNetConfig(hidden_size=[64, 64])
        keys = config.keys()
        assert "hidden_size" in keys
        values = config.values()
        assert [64, 64] in values
        items = list(config.items())
        assert any(k == "hidden_size" and v == [64, 64] for k, v in items)
