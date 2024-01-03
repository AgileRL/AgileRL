import os

import accelerate

from agilerl.utils.ilql_utils import (
    add_system_configs,
    convert_path,
    strip_from_beginning,
    strip_from_end,
    to_bin,
)


def test_convert_path_none():
    path = None
    converted_path = convert_path(path)
    assert converted_path is None


def test_convert_path():
    path = "example_path"
    converted_path = convert_path(path)
    assert converted_path == os.path.join(
        os.path.dirname(os.path.realpath("agilerl/utils/ilql_utils.py")), "../../", path
    )


def test_add_system_configs():
    config = {}
    accelerator = accelerate.Accelerator()
    config = add_system_configs(config, accelerator)

    assert config["device"] == str(accelerator.device)
    assert config["num_processes"] == accelerator.num_processes
    assert config["use_fp16"] == accelerator.use_fp16


def test_to_bin_none():
    n = 10
    pad_to_size = None

    bins = to_bin(n, pad_to_size)

    assert bins == [1, 0, 1, 0]


def test_to_bin():
    n = 10
    pad_to_size = 5

    bins = to_bin(n, pad_to_size)

    assert bins == [0, 1, 0, 1, 0]


def test_strip_from_end():
    string = "this string will be stripped from the end"
    strip_key = "from the end"

    string = strip_from_end(string, strip_key)

    assert string == "this string will be stripped "


def test_strip_from_beginning():
    string = "this string will be stripped from the beginning"
    strip_key = "this string will be stripped "

    string = strip_from_beginning(string, strip_key)

    assert string == "from the beginning"


def test_no_strip_from_beginning():
    string = "this string will not be stripped"
    strip_key = "zzz"

    string = strip_from_beginning(string, strip_key)

    assert string == "this string will not be stripped"
