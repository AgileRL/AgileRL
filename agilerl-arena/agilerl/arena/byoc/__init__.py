# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""BYOC cluster registration and related helpers for the Arena CLI.

The capability-gated command groups live in :mod:`group`, the Click commands in
:mod:`commands`, and the provider-specific orchestration in :mod:`installer`
(over :class:`~agilerl.arena.byoc.api.ByocApi`).
"""

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_register import run_cluster_register
from agilerl.arena.byoc.commands import (
    build_cluster_destroy_command,
    build_cluster_generate_spec_command,
    build_cluster_plan_command,
    build_cluster_provision_command,
    build_cluster_register_command,
    build_cluster_rotate_token_command,
    build_cluster_status_command,
    build_cluster_unregister_command,
)
from agilerl.arena.byoc.endpoints import SetupKind
from agilerl.arena.byoc.group import (
    ArenaRootGroup,
    ByocRootAccess,
    ClusterDynamicGroup,
    caps_allow_byoc_at_root,
    register_byoc_manifest_group,
    resolve_byoc_root_access,
)
from agilerl.arena.byoc.installer import (
    ByocInstaller,
    HelmInstaller,
    build_installer,
    normalize_setup_type,
    run_byoc_install,
    run_byoc_teardown,
)

__all__ = [
    "ArenaRootGroup",
    "ByocApi",
    "ByocInstaller",
    "ByocRootAccess",
    "ClusterDynamicGroup",
    "HelmInstaller",
    "SetupKind",
    "build_cluster_destroy_command",
    "build_cluster_generate_spec_command",
    "build_cluster_plan_command",
    "build_cluster_provision_command",
    "build_cluster_register_command",
    "build_cluster_rotate_token_command",
    "build_cluster_status_command",
    "build_cluster_unregister_command",
    "build_installer",
    "caps_allow_byoc_at_root",
    "normalize_setup_type",
    "register_byoc_manifest_group",
    "resolve_byoc_root_access",
    "run_byoc_install",
    "run_byoc_teardown",
    "run_cluster_register",
]
