import os
from pathlib import Path
from unittest.mock import patch

from agilerl.utils.cache import Cache


def test_cache_init():
    cache = Cache(None)

    assert cache.cache == {}
    assert cache.cache_hit_rate == 1.0


def test_set_get_update():
    cache = Cache(None)

    cache.__setitem__("key", "value")
    value = cache.__getitem__("key")

    assert value == "value"
    assert cache.__contains__("key")
    assert not cache.__contains__("nope")
    assert len(cache) == 1
    assert cache.items()[0] == ("key", "value")
    assert cache.keys()[0] == "key"
    assert cache.values()[0] == "value"

    cache.update({"key2": "value2"})

    assert cache.get_hit_rate() == 0.9901
    assert cache.get_cache() == {"key": "value", "key2": "value2"}


def test_dump_load(tmpdir):
    cache = Cache({"key": "value"})
    cache_path = Path(tmpdir) / "cache.pkl"

    with patch("os.path.exists") as _:
        cache.dump(cache_path)

    new_cache = Cache(None)
    new_cache.load(cache_path)

    assert new_cache.cache == {"key": "value"}
    assert new_cache.cache == cache.cache


def test_dump_makedirs():
    cache = Cache({"key": "value"})
    cache_path = "no"
    with patch("os.makedirs") as mock_md, patch("pickle.dump") as mock_dump:
        cache.dump(cache_path)

        mock_md.assert_called()
        mock_dump.assert_called()
    os.remove(cache_path)
