# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

import agilerl.modules as modules


def test_getattr_unknown_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = modules.ThisSymbolDoesNotExist


def test_getattr_known_symbol_resolves():
    # Lazy re-export path: resolves and caches the symbol.
    assert modules.EvolvableMLP is not None


def test_dir_includes_public_api():
    listing = dir(modules)
    assert "EvolvableMLP" in listing
    assert listing == sorted(listing)
