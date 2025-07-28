from __future__ import annotations

import os
import pickle as pkl
from typing import Any


class Cache:
    def __init__(self, cache_init: dict | None = None) -> None:
        """Initializes the Cache object.

        :param cache_init: Initial cache dictionary.
        :type cache_init: dict | None
        """
        assert cache_init is None or isinstance(cache_init, dict)
        if cache_init is None:
            cache_init = {}
        self.cache = cache_init
        self.cache_hit_rate = 1.0

    def dump(self, file_name: str) -> None:
        """Dumps the cache to a file.

        :param file_name: Name of the file to dump the cache to.
        :type file_name: str
        """
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, "wb") as f:
            pkl.dump(self.cache, f)

    def load(self, file_name: str) -> None:
        """Loads the cache from a file.

        :param file_name: Name of the file to load the cache from.
        :type file_name: str
        """
        with open(file_name, "rb") as f:
            self.cache.update(pkl.load(f))

    def __getitem__(self, key: str) -> Any:
        """Gets an item from the cache.

        :param key: Key of the item to get.
        :type key: str

        :return: Item from the cache.
        :rtype: Any
        """
        self.cache_hit_rate = (self.cache_hit_rate * 0.99) + 0.01
        return self.cache[key]

    def __setitem__(self, key: str, newvalue: Any) -> None:
        """Sets an item in the cache.

        :param key: Key of the item to set.
        :type key: str
        :param newvalue: Value to set.
        :type newvalue: Any
        """
        self.cache_hit_rate = self.cache_hit_rate * 0.99
        self.cache[key] = newvalue

    def __contains__(self, key: str) -> bool:
        """Checks if a key is in the cache.

        :param key: Key to check.
        :type key: str

        :return: True if the key is in the cache, False otherwise.
        :rtype: bool
        """
        return key in self.cache

    def __len__(self) -> int:
        """Gets the length of the cache.

        :return: Length of the cache.
        :rtype: int
        """
        return len(self.cache)

    def items(self) -> list[tuple[str, Any]]:
        """Gets the items of the cache.

        :return: Items of the cache.
        :rtype: list
        """
        return list(self.cache.items())

    def keys(self) -> list[str]:
        """Gets the keys of the cache.

        :return: Keys of the cache.
        :rtype: list
        """
        return list(self.cache.keys())

    def values(self) -> list[Any]:
        """Gets the values of the cache.

        :return: Values of the cache.
        :rtype: list
        """
        return list(self.cache.values())

    def update(self, new_stuff: dict) -> None:
        """Updates the cache with new items.

        :param new_stuff: New items to update the cache with.
        :type new_stuff: dict
        """
        self.cache.update(new_stuff)

    def get_hit_rate(self) -> float:
        """Gets the hit rate of the cache.

        :return: Hit rate of the cache.
        :rtype: float
        """
        return self.cache_hit_rate

    def get_cache(self) -> dict:
        """Gets the cache.

        :return: Cache.
        :rtype: dict
        """
        return self.cache
