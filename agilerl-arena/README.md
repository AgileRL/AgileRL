# agilerl-arena

`agilerl-arena` is the standalone Arena SDK + CLI package for AgileRL.

It provides:

- Python client for Arena workflows (auth, environment validation, experiment submission, deployment, inference)
- `arena` CLI for scripting and CI usage
- The training manifest schema (`agilerl.arena.models`): the Pydantic models every runtime trains from, plus `arena manifest validate` and `arena manifest schema`

This package is distributed independently from core `agilerl`, but exposes modules through the shared namespace:

```python
from agilerl.arena import ArenaClient, Agent
from agilerl.arena.models import TrainingManifest
```

Core `agilerl` re-exports those models and adds only what a local run needs (environment construction, a buildable replay buffer).

## Installation

Install the SDK and CLI on their own (no torch):

```bash
pip install agilerl-arena
```

Core `agilerl` depends on this package, so `pip install agilerl` includes it.

## Quickstart

### 1) Authenticate

Preferred for CI/automation:

```bash
export ARENA_API_KEY="arena_pat_..."
```

Or interactive login:

```bash
arena login
```

### 2) Validate a training manifest

```bash
arena manifest validate path/to/manifest.yaml
arena manifest schema
```

### 3) Validate an environment

```bash
arena env validate my-env --source path/to/my_env.py
```

### 4) Check GPU memory (LLM manifests)

```bash
arena memory estimate path/to/manifest.yaml --gpu "NVIDIA L4"

# Longest context one dedicated L4 serving GPU can hold:
arena memory solve max_model_len --inference --gpu "NVIDIA L4" \
    --model Qwen/Qwen2.5-7B-Instruct
```

`estimate` is a pre-submission gate (exit 0 fits, 3 over budget, 2 usage error).
`solve` holds every other input fixed and returns the largest value of one
field. Install `agilerl-arena[hub]` unless you pass `--config path/to/config.json`.
See `agilerl/arena/memory/README.md`.

### 5) Submit a training manifest

```bash
arena experiments submit path/to/manifest.yaml --project my-project
```

## Python SDK example

```python
from agilerl.arena import ArenaClient
from agilerl.arena.models import TrainingManifest

client = ArenaClient()  # uses ARENA_API_KEY if set

TrainingManifest.get_validated("dqn.yaml")

client.validate_environment(
    source="acrobot.py",
    name="acrobot-env",
)

result = client.submit_experiment(
    manifest="dqn.yaml",
    resource_id="arena-medium",
    project="my-project",
)

print(result)
```

## Inference example

```python
from agilerl.arena import Agent

agent = Agent("https://<deployment-id>.inference.agilerl.com", api_key="arena_pat_...")
action, _ = agent.get_action(observation)
```

Use the deployment URL from `arena agent list`.

## Notes on packaging and imports

- Distribution name: `agilerl-arena`
- Python import namespace: `agilerl.arena`
- CLI command: `arena`

`agilerl-arena` and `agilerl` intentionally share the `agilerl.*` namespace as separate packages.

## BYOC cluster registration

Register a customer Kubernetes cluster against Arena (enterprise BYOC v1) and
write Helm values files locally. Re-running with the same cluster name updates
the existing registration (upsert) instead of creating a duplicate.

```bash
arena cluster register --name my-cluster \
  --storage-endpoint http://s3.corp.example.com:9000 \
  --storage-bucket arena-prod \
  --storage-secret-name corp-s3
```

`--agent-namespace` (default `arena`) is the namespace for the agent Helm
release. With `--install`, the release is named after the cluster (`--name`),
unless `arena-byoc-agent` is already installed in that namespace. The chart
names its ClusterRoles after the release, so per-cluster release names let two
Arena agents share one Kubernetes cluster without colliding.

Without `--no-write`, the CLI writes into `--output-dir` (default
`./arena-cluster-<name>`):

- `agent-helm-values.yaml`
- `storage-helm-values.yaml`, when Arena returns bundled MinIO values
- `cluster-token.txt`, mode 0600

Bundled MinIO is optional. `--lab` is shorthand for `--install-storage
--narrow-allowed-ips`:

```bash
arena cluster register --name lab-cluster --install --lab --no-write
```

The first `--install-storage` installs MinIO in the `storage` namespace. Later
upserts reuse that Secret and do not reinstall the storage chart. Before
installing the agent chart, the CLI copies the storage Secret into the agent
namespace and creates that namespace if it is missing.

Register enables the BYOC provider unless you pass `--skip-enable`. Requires an
Arena server with the enterprise BYOC cluster register API.

### Lifecycle

`arena cluster unregister --name my-cluster` removes the Arena registration.
`--uninstall-helm` also uninstalls the agent chart. Bundled MinIO
(`arena-byoc-storage`) stays unless you pass `--uninstall-storage`.

`arena cluster rotate-token --name my-cluster` rotates the Arena cluster token
and writes it into the in-cluster Secret, then restarts the agent Deployment
Helm found for that cluster. The Secret name comes from
`existingClusterTokenSecret` in the agent Helm values, and is `cluster-token`
otherwise. Pass `--agent-namespace` when Helm lists more than one agent install.

Both commands resolve a kubeconfig in this order: `--kubeconfig`,
`./arena-cluster-<name>/kubeconfig`, `KUBECONFIG`, `~/.kube/config`. They find
Helm releases with `helm list --all-namespaces`, preferring a release named
after the cluster and falling back to `arena-byoc-agent`.

## Cloud cluster provisioning

`arena cluster` can provision a Kubernetes cluster with a packaged
Terraform module, then register and install the Arena Helm agent. Nebius is the
first supported provider.

Install Terraform and the [Nebius CLI](https://docs.nebius.com/cli/quickstart), then generate
a spec and fill in the cloud IDs:

```bash
arena cluster generate-spec --output nebius-cluster.yaml
```
After `provision`, the CLI runs `nebius mk8s cluster get-credentials` using the
Terraform `cluster_id` output and writes `./arena-cluster-<name>/kubeconfig`.
By default it also installs Gateway API `v1.6.0` CRDs, enables Cilium Gateway
API (`enable-gateway-api: "true"`), and creates an `arena` Gateway in the
`arena` namespace with wildcard HTTP and HTTPS listeners for
`*.<inference.domain>`. Set `arena.inference.domain` in the cluster spec and
optionally `arena.inference.hostname_template` for the host part (default
`inference-{deploymentId}`). Registration passes domain and hostname template separately to
Arena and the agent Helm values.
Provisioning also creates a Nebius access key for the `<name>-storage` service
account and writes it to the `arena.storage.secret_name` Secret in the
`arena` namespace (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, endpoint),
which the agent, Ray, and inference charts read. `--agent-namespace` (default
`arena`) sets where `--register-and-install` installs the agent chart. With
`arena.storage.install: true` the MinIO chart creates that Secret
instead. `register` and `--register-and-install` also create a GPU resource
class per `nebius.workers` entry, named `<cluster>-<worker.name>`, with node selector
`nebius.com/node-group-id` set to that provisioned worker node group. On AWS,
one class is created per `aws.workers` entry with `num_gpus` greater than 0,
selected by `eks.amazonaws.com/nodegroup`. CPU, GPU
count, and memory come from each worker's `preset`. Registration holds back 1
vCPU and 2 GiB for the kubelet and system DaemonSets. VRAM per GPU
(`gramPerGpu`, required by Arena's GPU experiment cascade) comes from
each worker's `platform`.
`arena` holds settings that are the same on every cloud: storage, inference,
gateway, and workload scheduling. `nebius` or `aws` holds that cloud's
resources. `cluster register --spec` and `provision --register-and-install`
both read `arena`. Set `arena.storage.install: true` to install the bundled
MinIO Helm chart (lab profile). `arena.storage.secret_name` defaults to
`arena-storage` in that case, and `storage` otherwise. Values in explicit
command-line flags override the YAML spec.
Before `plan` / `provision` / `destroy`, the Nebius Terraform provider needs
credentials. Either export a service-account authorized key:

```bash
export SA_ID="serviceaccount-..."
export AUTHKEY_PUBLIC_ID="publickey-..."
export AUTHKEY_PRIVATE_PATH="$HOME/.nebius/authorized_key.pem"
export NEBIUS_CLI="$HOME/.nebius/bin/nebius"
```

or omit `SA_ID` and authenticate as your Nebius CLI user. In that case set
`nebius.service_account_id` to an existing account, or leave it unset to
create a service account named `arena`, add it to the tenant `admins` group,
and attach it to the cluster node groups.

```bash
arena cluster plan --spec nebius-cluster.yaml
arena cluster provision nebius --spec nebius-cluster.yaml
KUBECONFIG=./arena-cluster-arena-nebius/kubeconfig \
  arena cluster register --spec nebius-cluster.yaml --install
```

Use `--register-and-install` on `provision` to run the last command
automatically. `terraform_state.bucket` is required. It must be a dedicated
Object Storage bucket, not the experiment-data bucket. `plan` / `provision`
look it up with the Nebius CLI. If it is missing, set
`terraform_state.create_bucket: true` or pass `--create-state-bucket`
(or confirm the prompt) to create it, a service account, and an access key.
Keys are written to `./arena-cluster-<name>/tfstate-credentials.json` and used
as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for the Terraform S3 backend.
On AWS, provisioning creates an IAM user named `<name>-tfstate`, allows it only
on that bucket, and writes the same file. Only the S3 backend uses these keys
(through a profile file in the Terraform directory); the AWS provider keeps
your own credentials. If the file is lost, a new key is
created on that user.
If that file is gone, the keys are fetched again from the Nebius access key
named `<name>-tfstate`. When the spec omits `nebius.project_id` or
`nebius.storage_project_id`, `register`, `plan`, `provision`, `status`, and
`destroy` load them from the cluster stored in Arena, then fetch those keys.
`arena cluster destroy --name <cluster>` does this without a spec file.
Terraform module files default to `./arena-cluster-<name>/terraform`; pass
`--state-dir` to keep them elsewhere.

```yaml
terraform_state:
  bucket: arena-nebius-tfstate
  # key: clusters/arena-nebius/terraform.tfstate
```

Provisioning uses two Nebius projects. `nebius.project_id` holds the cluster,
VPC, subnet, and shared filesystem. `nebius.storage_project_id` holds the
experiment bucket, its IAM, and the Terraform state bucket. Omit either one and
Terraform creates it, which makes `nebius.region` required.

`destroy` deletes the Kubernetes cluster and asks for confirmation. It deletes
the compute project when provisioning created it, and always keeps the storage
project. The experiment-data bucket is never deleted unless you pass
`--delete-storage`. `--delete-storage-project` also deletes the Terraform state
bucket and the storage project, and only works when provisioning created that
project. Otherwise the CLI asks whether to delete this cluster's Terraform state
object and keeps the state bucket. `--yes` skips prompts and keeps the state
object. The next `provision` imports the existing experiment bucket instead of
creating a new one.

```bash
arena cluster destroy --spec nebius-cluster.yaml
arena cluster destroy --spec nebius-cluster.yaml --delete-storage
arena cluster destroy --spec nebius-cluster.yaml --delete-storage-project
```

`destroy` does not uninstall the BYOC Helm releases. Use `arena cluster
unregister --name <name> --uninstall-helm` for that.

More CLI commands are in the published Arena docs and `arena --help`.
