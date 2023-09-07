from __future__ import annotations

import os
import pickle as pkl
from typing import Any


class Cache:
    def __init__(self, cache_init: dict | None = None) -> None:
        assert cache_init is None or isinstance(cache_init, dict)
        if cache_init is None:
            cache_init = {}
        self.cache = cache_init
        self.cache_hit_rate = 1.0

    def dump(self, file_name: str):
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, "wb") as f:
            pkl.dump(self.cache, f)

    def load(self, file_name: str):
        with open(file_name, "rb") as f:
            self.cache.update(pkl.load(f))

    def __getitem__(self, key: str) -> dict:
        self.cache_hit_rate = (self.cache_hit_rate * 0.99) + 0.01
        return self.cache[key]

    def __setitem__(self, key: str, newvalue: Any):
        self.cache_hit_rate = self.cache_hit_rate * 0.99
        self.cache[key] = newvalue

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def items(self):
        return list(self.cache.items())

    def keys(self):
        return list(self.cache.keys())

    def values(self):
        return list(self.cache.values())

    def update(self, new_stuff: dict):
        self.cache.update(new_stuff)

    def get_hit_rate(self):
        return self.cache_hit_rate

    def get_cache(self):
        return self.cache
