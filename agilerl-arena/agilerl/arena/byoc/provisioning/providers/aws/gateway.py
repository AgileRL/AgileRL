# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install the AWS Load Balancer Controller and a Cilium Gateway on an NLB."""

from __future__ import annotations

import datetime
import hashlib
import subprocess
from pathlib import Path

import click
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from agilerl.arena.byoc.cluster_helm import AGENT_NAMESPACE, helm_available
from agilerl.arena.byoc.provisioning.gateway_api import (
    configure_gateway_api,
)
from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    kubeconfig_env,
    require_kubectl,
)

LOAD_BALANCER_CONTROLLER_CHART_VERSION = "3.6.0"
LOAD_BALANCER_CONTROLLER_RELEASE = "aws-load-balancer-controller"
LOAD_BALANCER_CONTROLLER_NAMESPACE = "kube-system"
LOAD_BALANCER_CONTROLLER_SERVICE_ACCOUNT = "aws-load-balancer-controller"
LOAD_BALANCER_CHART_REPO = "https://aws.github.io/eks-charts"
WEBHOOK_CERT_DAYS = 3650
WEBHOOK_CERT_DIRECTORY = "load-balancer-webhook"
WEBHOOK_HELM_VALUES = "helm-values.yaml"
WEBHOOK_CERT_ANNOTATION = "arena.agilerl.dev/webhook-cert-sha"
WEBHOOK_DNS_NAMES = (
    "aws-load-balancer-webhook-service.kube-system",
    "aws-load-balancer-webhook-service.kube-system.svc",
    "aws-load-balancer-webhook-service.kube-system.svc.cluster.local",
)
NLB_SERVICE_ANNOTATIONS = {
    "service.beta.kubernetes.io/aws-load-balancer-backend-protocol": "TCP",
    "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "instance",
    "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
    "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
}
HELM_TIMEOUT = "10m"
GATEWAY_DELETE_TIMEOUT = "10m"


def configure_eks_gateway(
    kubeconfig_path: Path,
    gateway_name: str,
    domain: str,
    cluster_name: str,
    region: str,
    vpc_id: str,
    tls_secret_name: str | None,
    run: KubectlRun = subprocess.run,
) -> list[dict[str, str]]:
    """Install the load balancer controller and a Cilium Gateway on an NLB."""
    env = kubeconfig_env(kubeconfig_path)
    install_aws_load_balancer_controller(
        cluster_name=cluster_name,
        region=region,
        vpc_id=vpc_id,
        cert_dir=kubeconfig_path.parent / WEBHOOK_CERT_DIRECTORY,
        env=env,
        run=run,
    )
    return configure_gateway_api(
        kubeconfig_path=kubeconfig_path,
        gateway_name=gateway_name,
        domain=domain,
        tls_secret_name=tls_secret_name,
        infrastructure_annotations=NLB_SERVICE_ANNOTATIONS,
        run=run,
    )


def delete_eks_gateway(
    kubeconfig_path: Path,
    gateway_name: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Delete the Gateway and wait until its NLB is gone."""
    require_kubectl()
    env = kubeconfig_env(kubeconfig_path)
    click.echo(f"Deleting Gateway {gateway_name!r} in namespace {AGENT_NAMESPACE!r}.")
    _run(
        [
            "kubectl",
            "delete",
            "gateway",
            gateway_name,
            "--namespace",
            AGENT_NAMESPACE,
            "--ignore-not-found=true",
            "--wait=true",
            "--cascade=foreground",
            "--timeout",
            GATEWAY_DELETE_TIMEOUT,
        ],
        env=env,
        run=run,
    )
    # The Service finalizer blocks until the NLB and its ENIs are gone.
    _run(
        [
            "kubectl",
            "delete",
            "service",
            f"cilium-gateway-{gateway_name}",
            "--namespace",
            AGENT_NAMESPACE,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout",
            GATEWAY_DELETE_TIMEOUT,
        ],
        env=env,
        run=run,
    )


def write_load_balancer_webhook_certs(directory: Path) -> tuple[Path, Path, Path]:
    """Write a webhook CA and serving cert the API server can verify.

    Helm's ``genCA`` signatures fail API server verification.

    :return: Paths of the CA cert, serving cert, and serving key.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "aws-load-balancer-controller-ca")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=WEBHOOK_CERT_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name(
                [
                    x509.NameAttribute(
                        NameOID.COMMON_NAME, LOAD_BALANCER_CONTROLLER_RELEASE
                    )
                ]
            )
        )
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=WEBHOOK_CERT_DAYS))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(name) for name in WEBHOOK_DNS_NAMES]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = directory / "ca.crt"
    cert_path = directory / "tls.crt"
    key_path = directory / "tls.key"
    ca_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return ca_path, cert_path, key_path


def ensure_load_balancer_webhook_certs(directory: Path) -> tuple[Path, Path, Path]:
    """Return the webhook pair for *directory*, creating it when it is missing.

    A new pair changes the webhook CA while running pods still serve the previous cert.

    :return: Paths of the CA cert, serving cert, and serving key.
    """
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    ca_path = directory / "ca.crt"
    cert_path = directory / "tls.crt"
    key_path = directory / "tls.key"
    paths = (ca_path, cert_path, key_path)
    if all(path.is_file() for path in paths) and _webhook_certs_match(*paths):
        key_path.chmod(0o600)
        return paths
    return write_load_balancer_webhook_certs(directory)


def install_aws_load_balancer_controller(
    cluster_name: str,
    region: str,
    vpc_id: str,
    cert_dir: Path,
    env: dict[str, str],
    run: KubectlRun,
) -> None:
    """Install or upgrade the AWS Load Balancer Controller for NLB Services."""
    if not helm_available():
        msg = "helm not found on PATH; install Helm 3 before provisioning the AWS load balancer."
        raise click.ClickException(msg)
    ca_path, cert_path, key_path = ensure_load_balancer_webhook_certs(cert_dir)
    values_path = _write_webhook_helm_values(cert_dir, ca_path, cert_path, key_path)
    _run(
        [
            "helm",
            "upgrade",
            "--install",
            LOAD_BALANCER_CONTROLLER_RELEASE,
            LOAD_BALANCER_CONTROLLER_RELEASE,
            "--repo",
            LOAD_BALANCER_CHART_REPO,
            "--namespace",
            LOAD_BALANCER_CONTROLLER_NAMESPACE,
            "--version",
            LOAD_BALANCER_CONTROLLER_CHART_VERSION,
            "--set",
            f"clusterName={cluster_name}",
            "--set",
            f"region={region}",
            "--set",
            f"vpcId={vpc_id}",
            "--set",
            "serviceAccount.create=true",
            "--set",
            f"serviceAccount.name={LOAD_BALANCER_CONTROLLER_SERVICE_ACCOUNT}",
            "-f",
            str(values_path),
            "--wait",
            "--timeout",
            HELM_TIMEOUT,
        ],
        env=env,
        run=run,
    )


def _webhook_certs_match(ca_path: Path, cert_path: Path, key_path: Path) -> bool:
    try:
        ca_cert = x509.load_pem_x509_certificate(ca_path.read_bytes())
        serving = x509.load_pem_x509_certificate(cert_path.read_bytes())
        private_key = serialization.load_pem_private_key(
            key_path.read_bytes(), password=None
        )
        ca_cert.public_key().verify(
            serving.signature,
            serving.tbs_certificate_bytes,
            ec.ECDSA(hashes.SHA256()),
        )
    except (ValueError, InvalidSignature):
        return False
    serving_key = serving.public_key()
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        return False
    if not isinstance(serving_key, ec.EllipticCurvePublicKey):
        return False
    return private_key.public_key().public_numbers() == serving_key.public_numbers()


def _write_webhook_helm_values(
    directory: Path, ca_path: Path, cert_path: Path, key_path: Path
) -> Path:
    digest = hashlib.sha256(
        ca_path.read_bytes() + cert_path.read_bytes() + key_path.read_bytes()
    ).hexdigest()
    # The checksum is on the pod template, so this release rolls pods onto the secret.
    rendered = (
        "keepTLSSecret: true\n"
        "podAnnotations:\n"
        f"  {WEBHOOK_CERT_ANNOTATION}: {digest}\n"
        "webhookTLS:\n"
        "  caCert: |\n"
        f"{_pem_literal(ca_path.read_text())}"
        "  cert: |\n"
        f"{_pem_literal(cert_path.read_text())}"
        "  key: |\n"
        f"{_pem_literal(key_path.read_text())}"
    )
    values_path = directory / WEBHOOK_HELM_VALUES
    values_path.write_text(rendered)
    values_path.chmod(0o600)
    return values_path


def _pem_literal(pem: str) -> str:
    text = pem if pem.endswith("\n") else f"{pem}\n"
    return "".join(f"    {line}\n" for line in text.splitlines())


def _run(argv: list[str], env: dict[str, str], run: KubectlRun) -> None:
    result = run(
        argv,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        rendered = " ".join(argv)
        msg = f"{rendered} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
