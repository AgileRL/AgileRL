# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Enable Cilium Gateway API on Nebius managed Kubernetes clusters."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import click
import yaml

from agilerl.arena.byoc.cluster_helm import AGENT_NAMESPACE
from agilerl.arena.byoc.provisioning.inference import (
    DEFAULT_INFERENCE_TLS_SECRET_NAME,
    build_inference_wildcard_hostname,
    normalize_inference_domain,
)
from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    apply_manifests,
    ensure_namespace,
    kubeconfig_env,
    require_kubectl,
)

GATEWAY_API_VERSION = "v1.6.0"
GATEWAY_API_STANDARD_CRDS = (
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_gatewayclasses.yaml",
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_gateways.yaml",
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_httproutes.yaml",
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_referencegrants.yaml",
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_grpcroutes.yaml",
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/standard/gateway.networking.k8s.io_backendtlspolicies.yaml",
)
# The Gateway controller does not start without TLSRoute.
GATEWAY_API_EXPERIMENTAL_CRDS = (
    f"https://raw.githubusercontent.com/kubernetes-sigs/gateway-api/{GATEWAY_API_VERSION}/config/crd/experimental/gateway.networking.k8s.io_tlsroutes.yaml",
)
CILIUM_REQUIRED_GATEWAY_CRDS = (
    "gatewayclasses.gateway.networking.k8s.io",
    "gateways.gateway.networking.k8s.io",
    "httproutes.gateway.networking.k8s.io",
    "grpcroutes.gateway.networking.k8s.io",
    "referencegrants.gateway.networking.k8s.io",
    "backendtlspolicies.gateway.networking.k8s.io",
    "tlsroutes.gateway.networking.k8s.io",
)
CILIUM_GATEWAY_CLASS = "cilium"
CILIUM_GATEWAY_CONTROLLER = "io.cilium/gateway-controller"
CILIUM_OPERATOR_NAMESPACE = "kube-system"
CILIUM_GATEWAY_CONFIG = {
    "enable-gateway-api": "true",
    "enable-envoy-config": "true",
}
CILIUM_OPERATOR_SERVICE_ACCOUNT = "cilium-operator"
CILIUM_GATEWAY_API_CLUSTER_ROLE = "arena-cilium-gateway-api"
CILIUM_GATEWAY_API_CLUSTER_ROLE_BINDING = "arena-cilium-gateway-api"
CILIUM_GATEWAY_SERVICES_ROLE = "arena-cilium-gateway-services"
CILIUM_GATEWAY_SERVICES_ROLE_BINDING = "arena-cilium-gateway-services"
HTTP_LISTENER_NAME = "http"
HTTPS_LISTENER_NAME = "https"
HTTPS_REDIRECT_ROUTE_NAME = "arena-https-redirect"
ROLLOUT_TIMEOUT = "10m"


def build_gateway_api_parent_refs(
    gateway_name: str,
    gateway_namespace: str = AGENT_NAMESPACE,
) -> list[dict[str, str]]:
    """Return parent refs for Arena cluster registration."""
    return [
        {
            "group": "gateway.networking.k8s.io",
            "kind": "Gateway",
            "name": gateway_name,
            "namespace": gateway_namespace,
            "sectionName": HTTPS_LISTENER_NAME,
        }
    ]


def configure_gateway_api(
    kubeconfig_path: Path,
    gateway_name: str,
    domain: str,
    tls_secret_name: str | None,
    infrastructure_annotations: dict[str, str] | None = None,
    run: KubectlRun = subprocess.run,
) -> list[dict[str, str]]:
    """Install Gateway API CRDs, enable Cilium, and create the Arena Gateway."""
    require_kubectl()
    normalized_domain = normalize_inference_domain(domain)
    env = kubeconfig_env(kubeconfig_path)
    apply_gateway_api_crds(env=env, run=run)
    ensure_namespace(AGENT_NAMESPACE, env=env, run=run)
    resolved_tls_secret_name = ensure_inference_tls_secret(
        tls_secret_name=tls_secret_name,
        domain=normalized_domain,
        gateway_namespace=AGENT_NAMESPACE,
        env=env,
        run=run,
    )
    ensure_cilium_gateway_api_rbac(
        gateway_namespace=AGENT_NAMESPACE,
        env=env,
        run=run,
    )
    enable_cilium_gateway_api(env=env, run=run)
    ensure_cilium_gateway_class(env=env, run=run)
    ensure_cilium_gateway(
        gateway_name=gateway_name,
        gateway_namespace=AGENT_NAMESPACE,
        domain=normalized_domain,
        tls_secret_name=resolved_tls_secret_name,
        env=env,
        run=run,
        infrastructure_annotations=infrastructure_annotations,
    )
    ensure_https_redirect_route(
        gateway_name=gateway_name,
        gateway_namespace=AGENT_NAMESPACE,
        domain=normalized_domain,
        env=env,
        run=run,
    )
    return build_gateway_api_parent_refs(
        gateway_name=gateway_name,
        gateway_namespace=AGENT_NAMESPACE,
    )


def apply_gateway_api_crds(env: dict[str, str], run: KubectlRun) -> None:
    """Install the Gateway API CRDs Cilium's controller requires."""
    missing = not _gateway_crds_installed(env=env, run=run)
    for url in (*GATEWAY_API_STANDARD_CRDS, *GATEWAY_API_EXPERIMENTAL_CRDS):
        _kubectl(
            ["apply", "--server-side", "--force-conflicts", "-f", url],
            env=env,
            run=run,
        )
    # The operator checks these CRDs once at startup.
    if (
        missing
        and _cilium_config_flag("enable-gateway-api", env=env, run=run) == "true"
    ):
        _restart_cilium_operator(env=env, run=run)


def enable_cilium_gateway_api(env: dict[str, str], run: KubectlRun) -> None:
    """Turn on Cilium Gateway API and Envoy config translation."""
    current = {
        name: _cilium_config_flag(name, env=env, run=run)
        for name in CILIUM_GATEWAY_CONFIG
    }
    if current != CILIUM_GATEWAY_CONFIG:
        _kubectl(
            [
                "patch",
                "configmap",
                "cilium-config",
                "-n",
                "kube-system",
                "--type",
                "merge",
                "-p",
                json.dumps({"data": CILIUM_GATEWAY_CONFIG}),
            ],
            env=env,
            run=run,
        )
        _restart_cilium(env=env, run=run)
        return
    if not _gateway_crds_installed(env=env, run=run):
        _restart_cilium(env=env, run=run)


def build_cilium_gateway_api_rbac_manifests(
    gateway_namespace: str,
) -> list[dict[str, Any]]:
    """Return additive RBAC for Cilium Gateway API on managed clusters."""
    operator_subject = {
        "kind": "ServiceAccount",
        "name": CILIUM_OPERATOR_SERVICE_ACCOUNT,
        "namespace": CILIUM_OPERATOR_NAMESPACE,
    }
    return [
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": CILIUM_GATEWAY_API_CLUSTER_ROLE},
            "rules": [
                {
                    "apiGroups": ["gateway.networking.k8s.io"],
                    "resources": [
                        "gatewayclasses",
                        "gateways",
                        "httproutes",
                        "grpcroutes",
                        "referencegrants",
                        "tlsroutes",
                        "tcproutes",
                        "udproutes",
                        "listenersets",
                        "referencepolicies",
                        "backendtlspolicies",
                    ],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["gateway.networking.k8s.io"],
                    "resources": ["gatewayclasses"],
                    "verbs": ["patch"],
                },
                {
                    "apiGroups": ["gateway.networking.k8s.io"],
                    "resources": [
                        "gatewayclasses/status",
                        "gateways/status",
                        "httproutes/status",
                        "grpcroutes/status",
                        "tlsroutes/status",
                        "backendtlspolicies/status",
                        "tcproutes/status",
                        "udproutes/status",
                        "listenersets/status",
                    ],
                    "verbs": ["update", "patch"],
                },
                {
                    "apiGroups": ["cilium.io"],
                    "resources": ["ciliumgatewayclassconfigs"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["cilium.io"],
                    "resources": ["ciliumgatewayclassconfigs/status"],
                    "verbs": ["update", "patch"],
                },
                {
                    "apiGroups": ["cilium.io"],
                    "resources": ["ciliumenvoyconfigs"],
                    "verbs": [
                        "get",
                        "list",
                        "watch",
                        "create",
                        "update",
                        "patch",
                        "delete",
                    ],
                },
                {
                    "apiGroups": ["cilium.io"],
                    "resources": ["ciliumenvoyconfigs/status"],
                    "verbs": ["update", "patch"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["configmaps"],
                    "verbs": ["get", "list", "watch"],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": CILIUM_GATEWAY_API_CLUSTER_ROLE_BINDING},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": CILIUM_GATEWAY_API_CLUSTER_ROLE,
            },
            "subjects": [operator_subject],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "name": CILIUM_GATEWAY_SERVICES_ROLE,
                "namespace": gateway_namespace,
            },
            "rules": [
                {
                    "apiGroups": [""],
                    # Cilium creates an Endpoints object with the Gateway LoadBalancer Service.
                    "resources": ["services", "endpoints"],
                    "verbs": ["create", "update", "patch", "delete"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["secrets"],
                    "verbs": ["get", "list", "watch"],
                },
                {
                    "apiGroups": ["discovery.k8s.io"],
                    "resources": ["endpointslices"],
                    "verbs": [
                        "get",
                        "list",
                        "watch",
                        "create",
                        "update",
                        "patch",
                        "delete",
                    ],
                },
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": {
                "name": CILIUM_GATEWAY_SERVICES_ROLE_BINDING,
                "namespace": gateway_namespace,
            },
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": CILIUM_GATEWAY_SERVICES_ROLE,
            },
            "subjects": [operator_subject],
        },
    ]


def ensure_cilium_gateway_api_rbac(
    gateway_namespace: str,
    env: dict[str, str],
    run: KubectlRun,
) -> None:
    """Grant Cilium operator the Gateway API permissions Nebius omits."""
    manifests = build_cilium_gateway_api_rbac_manifests(
        gateway_namespace=gateway_namespace,
    )
    apply_manifests(manifests, env=env, run=run)


def ensure_inference_tls_secret(
    tls_secret_name: str | None,
    domain: str,
    gateway_namespace: str,
    env: dict[str, str],
    run: KubectlRun,
) -> str:
    """Return the TLS secret used by the Gateway, creating a self-signed cert if needed."""
    secret_name = tls_secret_name or DEFAULT_INFERENCE_TLS_SECRET_NAME
    if _tls_secret_exists(secret_name, gateway_namespace, env=env, run=run):
        return secret_name
    _create_self_signed_tls_secret(
        secret_name=secret_name,
        domain=domain,
        gateway_namespace=gateway_namespace,
        env=env,
        run=run,
    )
    return secret_name


def _tls_secret_exists(
    secret_name: str,
    gateway_namespace: str,
    env: dict[str, str],
    run: KubectlRun,
) -> bool:
    result = run(
        ["kubectl", "get", "secret", secret_name, "-n", gateway_namespace],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _create_self_signed_tls_secret(
    secret_name: str,
    domain: str,
    gateway_namespace: str,
    env: dict[str, str],
    run: KubectlRun,
) -> None:
    if not shutil.which("openssl"):
        msg = (
            "openssl not found on PATH; install openssl to generate a "
            "self-signed inference TLS secret."
        )
        raise click.ClickException(msg)
    wildcard_hostname = build_inference_wildcard_hostname(domain)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        key_path = tmp_dir / "tls.key"
        cert_path = tmp_dir / "tls.crt"
        config_path = tmp_dir / "openssl.cnf"
        config_path.write_text(
            textwrap.dedent(
                f"""\
                [req]
                distinguished_name = req_distinguished_name
                x509_extensions = v3_req
                prompt = no
                [req_distinguished_name]
                CN = {wildcard_hostname}
                [v3_req]
                subjectAltName = DNS:{wildcard_hostname},DNS:{domain}
                keyUsage = digitalSignature, keyEncipherment
                extendedKeyUsage = serverAuth
                """
            ),
            encoding="utf-8",
        )
        openssl = run(
            [
                "openssl",
                "req",
                "-x509",
                "-nodes",
                "-newkey",
                "rsa:2048",
                "-days",
                "825",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-config",
                str(config_path),
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if openssl.returncode != 0:
            detail = openssl.stderr.strip() or openssl.stdout.strip()
            msg = f"Could not generate a self-signed TLS certificate.{f' {detail}' if detail else ''}"
            raise click.ClickException(msg)
        create = run(
            [
                "kubectl",
                "create",
                "secret",
                "tls",
                secret_name,
                "-n",
                gateway_namespace,
                f"--cert={cert_path}",
                f"--key={key_path}",
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            msg = (
                f"Could not create TLS secret {secret_name!r} in "
                f"{gateway_namespace!r}: {create.stderr.strip()}"
            )
            raise click.ClickException(msg)


def ensure_cilium_gateway_class(env: dict[str, str], run: KubectlRun) -> None:
    """Create the Cilium GatewayClass used by the Arena Gateway."""
    manifest = {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "GatewayClass",
        "metadata": {"name": CILIUM_GATEWAY_CLASS},
        "spec": {"controllerName": CILIUM_GATEWAY_CONTROLLER},
    }
    apply = run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml.safe_dump(manifest, sort_keys=False),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        msg = f"Could not create GatewayClass {CILIUM_GATEWAY_CLASS!r}: {apply.stderr.strip()}"
        raise click.ClickException(msg)


def ensure_cilium_gateway(
    gateway_name: str,
    gateway_namespace: str,
    domain: str,
    tls_secret_name: str,
    env: dict[str, str],
    run: KubectlRun,
    infrastructure_annotations: dict[str, str] | None = None,
) -> None:
    """Create a Gateway with wildcard HTTP and HTTPS listeners for *domain*."""
    wildcard_hostname = build_inference_wildcard_hostname(domain)
    spec: dict[str, Any] = {
        "gatewayClassName": CILIUM_GATEWAY_CLASS,
        "listeners": [
            {
                "name": HTTP_LISTENER_NAME,
                "protocol": "HTTP",
                "port": 80,
                "hostname": wildcard_hostname,
                "allowedRoutes": {
                    "namespaces": {"from": "All"},
                },
            },
            {
                "name": HTTPS_LISTENER_NAME,
                "protocol": "HTTPS",
                "port": 443,
                "hostname": wildcard_hostname,
                "tls": {
                    "mode": "Terminate",
                    "certificateRefs": [
                        {
                            "kind": "Secret",
                            "name": tls_secret_name,
                        }
                    ],
                },
                "allowedRoutes": {
                    "namespaces": {"from": "All"},
                },
            },
        ],
    }
    if infrastructure_annotations:
        spec["infrastructure"] = {"annotations": infrastructure_annotations}
    manifest = {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {
            "name": gateway_name,
            "namespace": gateway_namespace,
        },
        "spec": spec,
    }
    apply = run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml.safe_dump(manifest, sort_keys=False),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        msg = (
            f"Could not create Gateway {gateway_name!r} in {gateway_namespace!r}: "
            f"{apply.stderr.strip()}"
        )
        raise click.ClickException(msg)


def ensure_https_redirect_route(
    gateway_name: str,
    gateway_namespace: str,
    domain: str,
    env: dict[str, str],
    run: KubectlRun,
) -> None:
    """Redirect HTTP listener traffic to HTTPS."""
    wildcard_hostname = build_inference_wildcard_hostname(domain)
    manifest = {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": HTTPS_REDIRECT_ROUTE_NAME,
            "namespace": gateway_namespace,
        },
        "spec": {
            "parentRefs": [
                {
                    "name": gateway_name,
                    "namespace": gateway_namespace,
                    "sectionName": HTTP_LISTENER_NAME,
                }
            ],
            "hostnames": [wildcard_hostname],
            "rules": [
                {
                    "filters": [
                        {
                            "type": "RequestRedirect",
                            "requestRedirect": {
                                "scheme": "https",
                                "statusCode": 301,
                            },
                        }
                    ],
                }
            ],
        },
    }
    apply = run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml.safe_dump(manifest, sort_keys=False),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        msg = (
            f"Could not create HTTPRoute {HTTPS_REDIRECT_ROUTE_NAME!r} "
            f"in {gateway_namespace!r}: {apply.stderr.strip()}"
        )
        raise click.ClickException(msg)


def _restart_cilium_operator(env: dict[str, str], run: KubectlRun) -> None:
    _kubectl(
        ["rollout", "restart", "deployment/cilium-operator", "-n", "kube-system"],
        env=env,
        run=run,
    )
    _kubectl(
        [
            "rollout",
            "status",
            "deployment/cilium-operator",
            "-n",
            "kube-system",
            f"--timeout={ROLLOUT_TIMEOUT}",
        ],
        env=env,
        run=run,
    )


def _restart_cilium(env: dict[str, str], run: KubectlRun) -> None:
    for resource in (
        "deployment/cilium-operator",
        "daemonset/cilium",
        "daemonset/cilium-envoy",
    ):
        _kubectl(
            ["rollout", "restart", resource, "-n", "kube-system"], env=env, run=run
        )
    for resource in (
        "deployment/cilium-operator",
        "daemonset/cilium",
        "daemonset/cilium-envoy",
    ):
        _kubectl(
            [
                "rollout",
                "status",
                resource,
                "-n",
                "kube-system",
                f"--timeout={ROLLOUT_TIMEOUT}",
            ],
            env=env,
            run=run,
        )


def _cilium_config_flag(name: str, env: dict[str, str], run: KubectlRun) -> str:
    result = run(
        [
            "kubectl",
            "get",
            "configmap",
            "cilium-config",
            "-n",
            "kube-system",
            "-o",
            f"jsonpath={{.data.{name}}}",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = textwrap.dedent(
            f"""\
            Could not read Cilium configuration: {result.stderr.strip()}
            Nebius clusters run Cilium in kube-system; check cluster access."""
        ).strip()
        raise click.ClickException(msg)
    return result.stdout.strip().lower()


def _gateway_crds_installed(env: dict[str, str], run: KubectlRun) -> bool:
    for name in CILIUM_REQUIRED_GATEWAY_CRDS:
        result = run(
            ["kubectl", "get", "crd", name],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
    return True


def _kubectl(argv: list[str], env: dict[str, str], run: KubectlRun) -> None:
    result = run(
        ["kubectl", *argv], env=env, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        msg = f"kubectl {' '.join(argv)} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
