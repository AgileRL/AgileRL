from importlib.metadata import metadata
from importlib.util import find_spec

from packaging.requirements import Requirement


def get_extra_dependencies(package: str, extra: str) -> list[str]:
    requires = metadata(package).get_all("Requires-Dist") or []
    deps = []
    for req in requires:
        r = Requirement(req)
        if r.marker and r.marker.evaluate({"extra": extra}):
            deps.append(r.name)
    return deps


LLM_PACKAGES = get_extra_dependencies("agilerl", "llm")
HAS_LLM_DEPENDENCIES = all(find_spec(pkg) is not None for pkg in LLM_PACKAGES)
