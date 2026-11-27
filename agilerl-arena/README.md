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

### 3) Validate a GEM environment

```bash
arena env validate my-gem-env --source path/to/my_gem_env/
```

### 4) Check GPU memory (LLM manifests)

```bash
arena memory estimate path/to/manifest.yaml --gpu "NVIDIA L4"

# An unset engine width or max_num_seqs is resolved from the node's GPUs:
arena memory estimate path/to/manifest.yaml --gpu "NVIDIA L4" --num-gpus 4

# Longest context the training run fits on an L4:
arena memory solve max_model_len path/to/manifest.yaml --gpu "NVIDIA L4"
```

`estimate` is a pre-submission gate (exit 0 fits, 3 over budget, 2 usage error).
`solve` holds the rest of the manifest fixed and returns the largest
`max_model_len` that fits. For a model outside the Arena catalog, install
`agilerl-arena[hub]` or pass `--config path/to/config.json`.
See `agilerl/arena/memory/README.md`.

### 5) Submit a training manifest

```bash
arena experiments submit path/to/manifest.yaml --project my-project

# List the GPU types a GRPO run trains on and the --num-gpus values each takes:
arena resources list --algorithm GRPO

# Pick a GPU type and count; Arena uses the fewest nodes that fit:
arena experiments submit path/to/manifest.yaml --project my-project \
    --num-gpus 8 --gpu-type h100-80
```

Without `--num-gpus` or `--gpu-type`, Arena picks the cheapest resource that
fits. With them, Arena picks only a layout the run's memory estimate fits and
rejects the request when none does.

### 6) Watch a run

```bash
arena experiments status my-experiment            # status, and why it failed if it did
arena experiments logs my-experiment --tail 100   # newest training log lines
arena experiments logs my-experiment --follow     # stream lines until the run ends
```

`logs` prints the same lines as the Log view in the web app, then the failure
reason when the run failed, including failures before training started (for
example a Python package that would not install). In Python:
`client.get_experiment_status(...)`, `client.get_experiment_logs(...)` and
`client.iter_experiment_logs(..., follow=True)`.

## Python SDK example

```python
from agilerl.arena import ArenaClient
from agilerl.arena.models import TrainingManifest

client = ArenaClient()  # uses ARENA_API_KEY if set

TrainingManifest.get_validated("grpo.yaml")

client.validate_environment(
    source="my_gem_env/",
    name="my-gem-env",
)

result = client.submit_experiment(
    manifest="grpo.yaml",
    project="my-project",
)

print(result)
```

## Inference example

```python
from agilerl.arena import Agent

agent = Agent("https://<deployment-id>.inference.agilerl.com", api_key="arena_pat_...")
result = agent.generate("What is 2 + 2?")
print(result.results[0].completion)
```

Use the deployment URL from `arena agent list`.

## Notes on packaging and imports

- Distribution name: `agilerl-arena`
- Python import namespace: `agilerl.arena`
- CLI command: `arena`

`agilerl-arena` and `agilerl` intentionally share the `agilerl.*` namespace as separate packages.

## BYOC cluster registration

### Connect a cluster in one step

`arena cluster connect` checks the cluster, registers it, installs the agent
and waits until the agent connects and sends its first health report:

```bash
arena cluster connect --name my-cluster --context my-context \
  --storage-endpoint https://s3.corp.example.com \
  --storage-bucket arena-prod \
  --storage-secret-name corp-s3
```

It stops before changing anything in Arena or the cluster if a preflight check
fails. Re-running it on a connected cluster keeps the agent token and upgrades
the agent in place; `--rotate-token` issues a new token. It waits up to
`--wait-timeout` seconds (default 900) for a heartbeat and a health report
from after the new agent ReplicaSet is ready, prints that report, and exits
non-zero if a health check is red. When Arena suggests resource classes for
the cluster's GPU nodes, it lists them and creates them with `--yes`, or after
a prompt in a terminal; without either, create them from the cluster's page in
Arena. It also names classes whose node selector matches no node, and says
when the agent needs an upgrade to report its nodes.
Connect needs `kubectl` and `helm` on `PATH` and a kubeconfig that can
create cluster-scoped RBAC, which in practice is cluster-admin. For options
connect does not have (inference domain, ingress, bundled MinIO, a cluster
spec), use `arena cluster register --install`.

`arena cluster preflight --context my-context` runs the same checks on their
own and changes nothing. `--json` prints the report as one JSON document, for
the Arena web wizard. The exit code is non-zero when a check fails.

| Check | Fails when | Warns when |
|---|---|---|
| `kubernetes_version` | the API server is older than 1.30 | |
| `can_create_*` | you may not create namespaces, ClusterRoles, ClusterRoleBindings, ValidatingAdmissionPolicies or their bindings | kubectl could not answer |
| `can_escalate_clusterroles`, `can_bind_clusterroles` | | you lack them (Helm then needs every right the agent's ClusterRoles grant) |
| `node_inventory` | | no node has allocatable `nvidia.com/gpu`, or a GPU node has no `nvidia.com/gpu.product` label |
| `storage_class` | `--ray-data-storage-class-name` names a missing StorageClass | there is no single default StorageClass |
| `privileged_pods` | admission refuses a privileged pod (a server-side dry run) in the agent namespace, or in `default` before that namespace exists | the namespace cannot be read, you may not create pods there, or the dry run fails for another reason (the probe name is taken, a webhook cannot dry-run, a timeout) |
| `one_agent` | another Arena agent is installed in the cluster | its Deployments cannot be listed |

Arena runs one agent per Kubernetes cluster. With `--name` of a cluster Arena
already has in this Kubernetes cluster, its own agent does not count as
another one.

### Register

Register a Kubernetes cluster against Arena (enterprise BYOC) and
write Helm values files locally. Re-running with the same cluster name updates
the existing registration (upsert) instead of creating a duplicate, and keeps
its agent token. `--install` then leaves the agent's token Secret alone, and
stops if the agent namespace has no token Secret yet. Pass `--rotate-token` to
issue a new token; with `--install` the CLI writes it to that Secret. The old
token keeps working for 24 hours. `arena cluster provision --register-and-install`
always issues a new token, because the cluster it just provisioned holds none.
`--byoc-scope` is `customer` (the default when omitted) or `platform`.
A platform cluster shares the Arena catalog and bucket and needs a platform
admin. Choosing platform uses ephemeral placement unless `--placement`
is set, or the spec already sets both scope and placement.

```bash
arena cluster register --name my-cluster \
  --storage-endpoint http://s3.corp.example.com:9000 \
  --storage-bucket arena-prod \
  --storage-secret-name corp-s3
```

`--agent-namespace` (default `arena`) is the namespace for the agent Helm
release. With `--install`, the release is named after the cluster (`--name`),
unless `arena-byoc-agent` is already installed in that namespace. The chart
names its ClusterRoles after the release, so two releases would not collide,
but Arena runs one agent per Kubernetes cluster and `connect` refuses a second.

Without `--no-write`, the CLI writes into `--output-dir` (default
`./arena-cluster-<name>`):

- `agent-helm-values.yaml`
- `storage-helm-values.yaml`, when Arena returns bundled MinIO values
- `cluster-token.txt`, mode 0600, when Arena returns a token

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
A cluster with runs is refused unless `--stop-runs` stops them. While Arena
reports the cluster as Deregistering, the CLI waits for the agent to delete
its job namespaces, then offers to uninstall the agent chart (default yes;
`--uninstall-helm` skips the prompt, `--keep-agent` keeps it). `--force`
removes a cluster whose agent is offline and prints a command that deletes
what the agent left. Bundled MinIO (`arena-byoc-storage`) stays unless you pass
`--uninstall-storage`.

`arena cluster rotate-token --name my-cluster` asks the cluster's agent to
renew its credential. The agent picks the new one up on its next poll and the
current one keeps working until then, so nothing in the cluster changes and no
kubeconfig is needed.

`arena cluster revoke-token --name my-cluster` revokes every credential of the
cluster now, for one you think leaked. The agent is refused from its next call
until you run `arena cluster connect` again, which issues a new enrollment
token and installs it.

`arena cluster smoke --name my-cluster` checks that the cluster can run jobs.
The agent installs a small training release with its own ServiceAccount, waits
until its Ray head is Ready, removes it, and the command prints whether that
passed or the reason it failed. The release asks for no GPU and mounts no
storage. `--timeout` sets how long to wait (default 3600 seconds).

`arena cluster manifest --name my-cluster --format argocd|flux` writes the
GitOps manifest that installs the agent at Arena's current release: an Argo CD
`Application`, or a Flux `HelmRepository` and `HelmRelease`, pinned to the
chart version and carrying the cluster's values, which hold no secrets. `-o`
writes it to a file instead of standard output. On a new cluster, first run the
bootstrap command from the install package's `INSTALL.md`, which creates the
Secrets the agent reads. To upgrade a GitOps install, commit the manifest again
after an Arena release; the command prints any ClusterRole changes, which your
GitOps tool needs the rights to apply. Arena serves manifests only when it
publishes the agent chart to a private registry.

Every command that talks to Kubernetes (`preflight`, `connect`,
`register --install`, `register --install-storage`, `unregister`,
`upgrade`, `smoke`) takes
`--kubeconfig` and `--context`. Without `--context` it uses the current context
of `--kubeconfig`, of `./arena-cluster-<name>/kubeconfig`, or of kubectl's own
default (`KUBECONFIG`, `~/.kube/config`). That last one is used only after you
confirm the context and API server it names; in a non-interactive shell the
command stops and asks for `--context`. The commands find Helm releases with
`helm list --all-namespaces`, preferring a release named after the cluster and
falling back to `arena-byoc-agent`.

Arena records each cluster's identity, the UID of its `kube-system` namespace.
Registering with `--install` sends it, and the agent reports it on every
heartbeat. A cluster registered before Arena kept identities takes the first one
reported. After that, `connect`, `upgrade` and `unregister`
stop before they change anything when the context points at a different
Kubernetes cluster, and `smoke` starts nothing.

`unregister`, `rotate-token`, `revoke-token`, `upgrade`, `manifest` and `smoke` look the cluster up in your active
org. `--platform` targets AgileRL's own platform clusters instead, which needs
AgileRL staff access. Register, `register-classes` and `destroy` take the
scope from `--byoc-scope` or the spec's `byoc_scope`.

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
selected by `eks.amazonaws.com/nodegroup`. CPU, GPU count, and memory are the
machine size from the worker preset or instance. VRAM per GPU
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
arena cluster provision --spec nebius-cluster.yaml
arena cluster register --spec nebius-cluster.yaml --install \
  --kubeconfig ./arena-cluster-arena-nebius/kubeconfig
```

Use `--register-and-install` on `provision` to run the last command
automatically. After the cluster is registered, add GPU sizes from the
Nebius project:

```bash
arena cluster register-classes --spec nebius-cluster.yaml --all
arena cluster register-classes --spec nebius-cluster.yaml \
  --preset gpu-h100-sxm:8gpu-128vcpu-1600gb \
  --preset gpu-h200-sxm:1gpu-16vcpu-200gb
```

`--all` registers every GPU preset. Repeat `--preset platform:preset` to
register some of them. With neither flag, the CLI prints the list and asks
which to register. The class CPU and memory are that preset's machine size.
A worker's `register_node_group` defaults to false, so the class is registered
without creating a Nebius node group. That class selects
`nebius.com/platform` and `nebius.com/preset`. `gpu.type` is the platform and
`gpu.count` matches the preset. Set `register_node_group: true` on that
worker, or pass `--register-node-group`, to append a missing preset to
`nebius.workers` (min size 0) and apply Terraform before the class is created.
The class then selects `nebius.com/node-group-id`.
If a class with that name already exists and differs, the CLI prints the diff
and asks before updating it. `--yes` applies the update.
`terraform_state.bucket` is required. It must be a dedicated
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

### Module versions

The `.tf` files are a versioned module, `arena-cluster-aws` or
`arena-cluster-nebius`. This CLI implements orchestration contracts `aws.v1`
and `nebius.v1`: the order of applies, the variables it writes, and the
outputs it reads. A module release can change resources inside that contract.
A new phase or a new required variable is a new contract and needs a new CLI.

Each release is `MAJOR.MINOR.PATCH`. The wheel bundles `1.0.0`. A cluster
pins the release in `./arena-cluster-<name>/module.json` and, after register,
under `byocProvider.config.module` (`name`, `version`, `digest`,
`orchestration`). Plan, apply, and destroy keep using that pin. Set
`module_version` in the spec, or pass `--module-version`, to choose a release
for a new cluster. `--module-version` on an existing pin is the upgrade.
`plan` first. A spec version that disagrees with the pin stops the command.

`--module-catalog-url` points at a catalog. `--module-catalog-public-key` is
the Ed25519 PEM that signed it. The CLI checks `index.json.sig` and
`manifest.json.sig`, then the `sha256` digest of the module files (not the
tarball). A yanked release is skipped for new clusters and still downloads
for a pin. If the catalog cannot be reached and the cluster has no pin, the
CLI uses the bundled module. A requested version that is not bundled needs
the catalog.

The catalog layout is:

```text
modules/aws/index.json
modules/aws/index.json.sig
modules/aws/1.0.0/manifest.json
modules/aws/1.0.0/manifest.json.sig
modules/aws/1.0.0/module.tar.gz
```

`manifest.json` carries `schema` 1, `name`, `provider`, `version`,
`orchestration`, `terraform_version`, `archive`, `digest`, and `files`.
The tarball contains those files and nothing else. It does not contain
`backend.tf`. The Nebius module does not contain `provider.tf`; the CLI
writes both at run time. Credentials and state stay on the machine that
runs Terraform. The catalog never sees them.

`aws.v1` applies `aws_eks_cluster.arena`, installs Cilium, then applies the
rest. It keeps the experiment bucket and its IAM user unless
`--delete-storage` is set. `nebius.v1` has no Cilium phase. Destroy leaves
the Nebius projects to `finish_destroy`, and keeps object storage unless
`--delete-storage` is set.

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

`destroy` does not uninstall the BYOC Helm releases. `arena cluster
unregister --name <name> --uninstall-helm` uninstalls the agent once Arena has
archived the cluster.

More CLI commands are in the published Arena docs and `arena --help`.
