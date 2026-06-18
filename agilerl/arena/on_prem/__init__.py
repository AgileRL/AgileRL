"""On-prem worker cluster install/teardown for the Arena CLI.

The capability-gated command groups live in :mod:`group`, the Click commands in
:mod:`commands`, and the provider-specific orchestration in :mod:`installer`
(over :class:`~agilerl.arena.on_prem.api.OnPremApi`).
"""

from agilerl.arena.on_prem.api import OnPremApi
from agilerl.arena.on_prem.commands import (
    build_install_command,
    build_teardown_command,
    register_on_prem_install,
)
from agilerl.arena.on_prem.endpoints import SetupKind
from agilerl.arena.on_prem.group import (
    ArenaRootGroup,
    OnPremDynamicGroup,
    capabilities_show_on_prem_root,
    caps_allow_on_prem_at_root,
    register_on_prem_manifest_group,
)
from agilerl.arena.on_prem.installer import (
    HelmInstaller,
    OnPremInstaller,
    SwarmInstaller,
    build_installer,
    normalize_setup_type,
    run_on_prem_install,
    run_on_prem_teardown,
)

__all__ = [
    "ArenaRootGroup",
    "HelmInstaller",
    "OnPremApi",
    "OnPremDynamicGroup",
    "OnPremInstaller",
    "SetupKind",
    "SwarmInstaller",
    "build_install_command",
    "build_installer",
    "build_teardown_command",
    "capabilities_show_on_prem_root",
    "caps_allow_on_prem_at_root",
    "normalize_setup_type",
    "register_on_prem_install",
    "register_on_prem_manifest_group",
    "run_on_prem_install",
    "run_on_prem_teardown",
]
