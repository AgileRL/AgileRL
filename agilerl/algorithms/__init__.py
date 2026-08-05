# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Algorithm implementations.

Re-exports are lazy; the public surface is declared in ``__init__.pyi``.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
