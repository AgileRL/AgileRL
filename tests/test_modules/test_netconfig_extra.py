from agilerl.modules.configs import MlpNetConfig
from agilerl.utils.evolvable_networks import config_from_dict


def test_netconfig_iter_yields_field_names():
    cfg = MlpNetConfig(hidden_size=[64])
    keys = list(cfg)
    assert "hidden_size" in keys
    assert keys == cfg.keys()


def test_netconfig_copy_is_a_deep_copy():
    cfg = MlpNetConfig(hidden_size=[64])
    clone = cfg.copy()
    assert clone == cfg
    assert clone is not cfg
    assert clone.hidden_size is not cfg.hidden_size


def test_config_from_dict_passthrough_for_netconfig():
    cfg = MlpNetConfig(hidden_size=[64])
    assert config_from_dict(cfg) is cfg
