from importlib.metadata import PackageNotFoundError, metadata

from packaging.requirements import Requirement


def get_extra_dependencies(package: str, extra: str) -> list[str]:
    requires = metadata(package).get_all("Requires-Dist") or []
    deps = []
    for req in requires:
        r = Requirement(req)
        if r.marker and r.marker.evaluate({"extra": extra}):
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
HAS_LLM_DEPENDENCIES = all(_is_dist_installed(pkg) for pkg in LLM_PACKAGES)
