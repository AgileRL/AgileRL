from importlib.metadata import PackageNotFoundError, metadata, version

from packaging.markers import default_environment
from packaging.requirements import Requirement


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


LLM_PACKAGES = get_extra_dependencies("agilerl", "llm")


def _is_distribution_installed(distribution: str) -> bool:
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


HAS_LLM_DEPENDENCIES = all(_is_distribution_installed(pkg) for pkg in LLM_PACKAGES)
