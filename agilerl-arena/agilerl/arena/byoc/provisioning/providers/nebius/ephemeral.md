# Ephemeral Nebius projects

`arena cluster provision nebius` creates the managed Kubernetes cluster. Ephemeral GPU jobs do not run in that cluster's project. Each region they can be placed in needs its own project and tunnel-agent IAM, created here, before the first job.

The in-cluster provisioner service account cannot create these. Do not grant it tenant admin. Use an account that can create projects, groups, memberships, and access permits.

Create one set per hunt region. The cluster spec's `nebius.region` is one. If resource classes hunt more regions, each region needs the same set. Names match the agent chart's `provisioner.yaml` (`tunnel_agent`).

| Resource | Name | Parent |
| --- | --- | --- |
| IAM v2 project | `arena-{region}` | tenant, in that region |
| IAM group | `arena-tunnel-agents` | the `arena-{region}` project |
| Service account | `arena-tunnel-agent` | the `arena-{region}` project |
| Group membership | `arena-tunnel-agent` is a member | the group |
| Access permit | role `applicationtunnel.agent`, resource id is the project id | the group |

One access permit per project. It covers every tunnel created in that project later. Do not create a permit per tunnel, and do not delete the permit when a job ends.

The provisioner service account also needs `admin` on each `arena-{region}` project, so it can create the head VM with `arena-tunnel-agent` attached. Create this once per tenant, plus one permit per region:

| Resource | Name | Parent |
| --- | --- | --- |
| IAM group | `arena-byoc-provisioners` | tenant |
| Group membership | the provisioner service account is a member | that group |
| Access permit | role `admin`, resource id is the `arena-{region}` project id | that group |

Scope `admin` to the region projects. Do not add the provisioner to the tenant `admins` group. It stays in the tenant `editors` group for everything else.

This project is not the managed-Kubernetes project (`nebius.project_id`, Terraform `var.cluster_name`) and not the storage project. Leave both of those, and the experiment bucket, as they are.

## What the provisioner still creates

Inside `arena-{region}`, once the table above exists:

- VPC network `arena` and subnet `arena`, CIDR `10.0.0.0/13` (`subnet_cidr` in `provisioner.yaml`)
- One unnamed tunnel per job, label `byoc-job=byoc-job-{job_id}`
- The k3s VMs and the shared filesystem

Boot disks use public image families in `project-e00public-images`: `ubuntu24.04-driverless` on the head, `ubuntu24.04-cuda13.0` on GPU and preprocessing nodes. Nothing to create for those.

## What that service account cannot do

Listing works. Creating these does not, and IAM returns `PermissionDenied`:

- IAM projects. `iam project create` on the tenant is denied. Create the project with IAM v2 (`iam v2 project`). The provisioner still looks it up with `iam project get-by-name --parent-id <tenant> --name arena-{region}`.
- IAM groups. `iam group create` in the project is denied.
- Group memberships. `iam group-membership create` on the group is denied. The list command is `iam group-membership list-members`, and the JSON key is `memberships` (not `items`). The member id is `spec.member_id`.
- Access permits. `iam access-permit create` on the group is denied. `iam access-permit list --parent-id <group>` works. The JSON key is `items`. An empty list is `{}` with that key omitted. The provisioner only checks that the project permit exists. If it is missing, the job fails with `tunnel agent group … has no applicationtunnel.agent access permit on project …`.
- VMs with a service account attached. Without project `admin`, creating the head VM with `arena-tunnel-agent` returns Compute `PermissionDenied` with an empty resource id. Tenant `editor` is not enough. The `arena-byoc-provisioners` permit above fixes this.

The same account can list those resources, create the network and subnet, create tunnels, and create VMs that have no service account. Subnet CIDR comes back on `status.ipv4_private_cidrs`. Tunnels reject `--name` (`name is not supported: tunnel is an unnamed resource`); identify them by the `byoc-job` label.

A preemptible GPU VM needs `follows_spot_price` and `recovery_policy = FAIL`. `RECOVER` is rejected. The head VM is not preemptible. That stays in the provisioner, not in this command.
