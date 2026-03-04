from importlib.metadata import metadata, packages_distributions
from importlib.util import find_spec

from packaging.requirements import Requirement

# Reverse map: distribution name -> list of importable top-level packages
_DIST_TO_MODULES: dict[str, list[str]] = {}
for _mod, _dists in packages_distributions().items():
    for _dist in _dists:
        _DIST_TO_MODULES.setdefault(_dist.lower(), []).append(_mod)


def get_extra_dependencies(package: str, extra: str) -> list[str]:
    requires = metadata(package).get_all("Requires-Dist") or []
    deps = []
    for req in requires:
        r = Requirement(req)
        if r.marker and r.marker.evaluate({"extra": extra}):
            deps.append(r.name)
    return deps


def _is_installed(dist_name: str) -> bool:
    """Check whether a distribution is importable.

    Resolves PyPI distribution names (e.g. ``python-keycloak``) to their
    actual importable top-level module (e.g. ``keycloak``) before probing.
    """
    modules = _DIST_TO_MODULES.get(dist_name.lower())
    if modules:
        return any(find_spec(m) is not None for m in modules)
    # Fallback: try the distribution name directly (works when they match)
    return find_spec(dist_name) is not None


LLM_PACKAGES = get_extra_dependencies("agilerl", "llm")
HAS_LLM_DEPENDENCIES = all(_is_installed(pkg) for pkg in LLM_PACKAGES)

ARENA_PACKAGES = get_extra_dependencies("agilerl", "arena")
HAS_ARENA_DEPENDENCIES = all(_is_installed(pkg) for pkg in ARENA_PACKAGES)
