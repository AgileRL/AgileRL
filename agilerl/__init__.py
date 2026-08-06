# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import warnings
from enum import Enum
from importlib.metadata import PackageNotFoundError, metadata, version
from pkgutil import extend_path
from typing import TYPE_CHECKING

import lazy_loader as lazy
from packaging.markers import default_environment
from packaging.requirements import Requirement

# Allow additional distributions (e.g. agilerl-arena) to contribute modules
# under the shared `agilerl.*` namespace.
__path__ = extend_path(__path__, __name__)

if TYPE_CHECKING:
    from agilerl.training.trainer import ArenaTrainer, LocalTrainer

__all__ = ["ArenaTrainer", "LocalTrainer"]

# pygame currently imports deprecated pkg_resources -> suppress warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API\\..*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)


def get_extra_dependencies(package: str, extra: str) -> list[str]:
    requires = metadata(package).get_all("Requires-Dist") or []
    marker_environment: dict[str, str] = {**default_environment(), "extra": extra}
    deps = []
    for req in requires:
        r = Requirement(req)
        if r.marker and r.marker.evaluate(marker_environment):
            deps.append(r.name)
    return deps


LLM_PACKAGES = get_extra_dependencies("agilerl", "llm")
ARENA_PACKAGES = get_extra_dependencies("agilerl", "arena")


def _is_distribution_installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


# Use these flags for lazy import checks
HAS_LLM_DEPENDENCIES = all(_is_distribution_installed(pkg) for pkg in LLM_PACKAGES)
HAS_ARENA_DEPENDENCIES = all(_is_distribution_installed(pkg) for pkg in ARENA_PACKAGES)
HAS_LIGER_KERNEL = _is_distribution_installed("liger-kernel")
HAS_VLLM = _is_distribution_installed("vllm")


class AgentType(Enum):
    """Enumeration of supported agent types."""

    SingleAgent = "single_agent"
    MultiAgent = "multi_agent"
    LLMAgent = "llm_agent"
    OfflineAgent = "offline_agent"
    BanditAgent = "bandit_agent"


# NOTE: Need to lazy-load to avoid circular imports
__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "training.trainer": ["LocalTrainer", "ArenaTrainer"],
    },
)
