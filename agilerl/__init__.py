import warnings
from enum import Enum
from importlib.metadata import PackageNotFoundError, metadata, version

from packaging.markers import default_environment
from packaging.requirements import Requirement

# pygame currently imports deprecated pkg_resources -> suppress warning
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API\\..*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)


def get_extra_dependencies(package: str, extra: str) -> list[str]:
    requires = metadata(package).get_all("Requires-Dist") or []
    marker_environment = default_environment()
    marker_environment["extra"] = extra
    deps = []
    for req in requires:
        r = Requirement(req)
        if r.marker and r.marker.evaluate(marker_environment):
            deps.append(r.name)
    return deps


def _is_dist_installed(dist_name: str) -> bool:
    """Check if a distribution is installed by its PyPI name.

    Uses distribution metadata instead of find_spec because PyPI names
    can differ from import names.
    """
    try:
        metadata(dist_name)
        return True
    except PackageNotFoundError:
        return False


LLM_PACKAGES = get_extra_dependencies("agilerl", "llm")
ARENA_PACKAGES = get_extra_dependencies("agilerl", "arena")


def _is_distribution_installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


# Use these flags for laxzy import checks
HAS_LLM_DEPENDENCIES = all(_is_distribution_installed(pkg) for pkg in LLM_PACKAGES)
HAS_ARENA_DEPENDENCIES = all(_is_distribution_installed(pkg) for pkg in ARENA_PACKAGES)
HAS_LIGER_KERNEL = _is_distribution_installed("liger-kernel")
HAS_VLLM = _is_distribution_installed("vllm")
HAS_DEEPSPEED = _is_distribution_installed("deepspeed")


class AgentType(Enum):
    """Enumeration of supported agent types."""

    SingleAgent = "single_agent"
    MultiAgent = "multi_agent"
    LLMAgent = "llm_agent"
    OfflineAgent = "offline_agent"
    BanditAgent = "bandit_agent"
