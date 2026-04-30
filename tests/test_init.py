import pytest

import agilerl


@pytest.mark.parametrize("package,extra", [("agilerl", "llm")])
def test_get_extra_dependencies(package, extra):
    result = agilerl.get_extra_dependencies(package, extra)
    assert isinstance(result, list)
    assert all(isinstance(d, str) for d in result)


def test_llm_packages_and_has_llm_dependencies():
    _ = agilerl.LLM_PACKAGES
    _ = agilerl.HAS_LLM_DEPENDENCIES


class TestIsDistributionInstalled:
    def test_is_distribution_installed_found(self):
        assert agilerl._is_distribution_installed("agilerl") is True

    def test_is_distribution_installed_not_found(self):
        assert (
            agilerl._is_distribution_installed("_nonexistent_pkg_xyz_12345_") is False
        )
