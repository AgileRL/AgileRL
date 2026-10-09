terraform {
  required_version = ">= 1.10.0"

  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.8"
    }
  }
}

locals {
  project_id = (
    var.project_id != ""
    ? var.project_id
    : one(nebius_iam_v2_project.arena[*].id)
  )
  storage_project_id = (
    var.storage_project_id != ""
    ? var.storage_project_id
    : one(nebius_iam_v2_project.storage[*].id)
  )
  subnet_id = (
    var.subnet_id != ""
    ? var.subnet_id
    : one(nebius_vpc_v1_subnet.arena[*].id)
  )
  cluster_service_account_id = (
    var.service_account_id != ""
    ? var.service_account_id
    : one(nebius_iam_v1_service_account.arena[*].id)
  )
}

# Compute project: VPC, subnet, Kubernetes, and the shared filesystem.
resource "nebius_iam_v2_project" "arena" {
  count     = var.project_id == "" ? 1 : 0
  parent_id = var.tenant_id
  name      = var.cluster_name
  region    = var.region
}

# Storage project: experiment bucket, storage IAM, and the CLI Terraform state
# bucket. Cluster destroy keeps it so experiment data outlives the compute project.
resource "nebius_iam_v2_project" "storage" {
  count     = var.storage_project_id == "" ? 1 : 0
  parent_id = var.tenant_id
  name      = "${var.cluster_name}-storage"
  region    = var.region
}

resource "nebius_vpc_v1_network" "arena" {
  count     = var.subnet_id == "" ? 1 : 0
  parent_id = local.project_id
  name      = "${var.cluster_name}-network"
}

resource "nebius_vpc_v1_subnet" "arena" {
  count      = var.subnet_id == "" ? 1 : 0
  parent_id  = local.project_id
  name       = "${var.cluster_name}-subnet"
  network_id = one(nebius_vpc_v1_network.arena[*].id)

  ipv4_private_pools = {
    use_network_pools = true
  }
}

data "nebius_iam_v1_group" "admins" {
  count     = var.service_account_id == "" ? 1 : 0
  name      = "admins"
  parent_id = var.tenant_id
}

resource "nebius_iam_v1_service_account" "arena" {
  count       = var.service_account_id == "" ? 1 : 0
  parent_id   = local.project_id
  name        = "arena"
  description = "Arena Kubernetes cluster service account"
}

resource "nebius_iam_v1_group_membership" "arena_admin" {
  count     = var.service_account_id == "" ? 1 : 0
  parent_id = data.nebius_iam_v1_group.admins[0].id
  member_id = nebius_iam_v1_service_account.arena[0].id
}

resource "nebius_iam_v1_service_account" "storage" {
  parent_id   = local.storage_project_id
  name        = "${var.cluster_name}-storage"
  description = "Arena object storage service account"
}

resource "nebius_iam_v2_access_key" "storage" {
  parent_id   = local.storage_project_id
  name        = "${var.cluster_name}-storage"
  description = "Arena object storage access key"

  account = {
    service_account = {
      id = nebius_iam_v1_service_account.storage.id
    }
  }

  # INLINE returns the secret in the API response so Terraform can output it.
  secret_delivery_mode = "INLINE"
}

resource "nebius_iam_v1_group" "storage_editors" {
  parent_id = var.tenant_id
  name      = "${var.cluster_name}-storage-editors"
}

resource "nebius_iam_v1_group_membership" "storage_editor" {
  parent_id = nebius_iam_v1_group.storage_editors.id
  member_id = nebius_iam_v1_service_account.storage.id
}

# Experiment data, metrics, and checkpoints. Cluster destroy does not delete
# this bucket unless the CLI is passed --delete-storage.
resource "nebius_storage_v1_bucket" "arena_data" {
  parent_id             = local.storage_project_id
  name                  = var.storage_bucket_name
  default_storage_class = "STANDARD"
  versioning_policy     = "DISABLED"
  max_size_bytes        = var.object_storage_size_gib * 1024 * 1024 * 1024

  bucket_policy = {
    rules = [{
      paths    = ["*"]
      roles    = ["storage.editor"]
      group_id = nebius_iam_v1_group.storage_editors.id
    }]
  }
}

resource "nebius_compute_v1_filesystem" "shared" {
  count            = var.provision_shared_filesystem ? 1 : 0
  name             = "${var.cluster_name}-shared"
  parent_id        = local.project_id
  type             = var.filesystem_type
  size_gibibytes   = var.filesystem_size_gib
  block_size_bytes = 4096
}

resource "nebius_mk8s_v1_cluster" "arena" {
  name      = var.cluster_name
  parent_id = local.project_id

  control_plane = {
    endpoints = {
      public_endpoint = {}
    }
    version           = var.kubernetes_version
    subnet_id         = local.subnet_id
    etcd_cluster_size = var.etcd_cluster_size
  }
}

resource "nebius_mk8s_v1_node_group" "arena" {
  name      = "arena"
  parent_id = nebius_mk8s_v1_cluster.arena.id
  autoscaling = {
    min_node_count = var.arena_min_node_count
    max_node_count = var.arena_max_node_count
  }
  depends_on = [nebius_iam_v1_group_membership.arena_admin]

  template = {
    service_account_id = local.cluster_service_account_id
    resources = {
      platform = var.arena_platform
      preset   = var.arena_preset
    }
    filesystems = var.provision_shared_filesystem ? [{
      attach_mode = "READ_WRITE"
      mount_tag   = var.filesystem_mount_tag
      existing_filesystem = {
        id = nebius_compute_v1_filesystem.shared[0].id
      }
    }] : []
  }
}

resource "nebius_mk8s_v1_node_group" "worker" {
  for_each = { for worker in var.workers : worker.name => worker }

  name      = each.value.name
  parent_id = nebius_mk8s_v1_cluster.arena.id
  autoscaling = {
    min_node_count = each.value.min_node_count
    max_node_count = each.value.max_node_count
  }
  depends_on = [nebius_iam_v1_group_membership.arena_admin]

  template = {
    service_account_id = local.cluster_service_account_id
    resources = {
      platform = each.value.platform
      preset   = each.value.preset
    }
    gpu_settings = {
      drivers_preset = each.value.gpu_drivers_preset
    }
    taints = [{
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }]
    filesystems = var.provision_shared_filesystem ? [{
      attach_mode = "READ_WRITE"
      mount_tag   = var.filesystem_mount_tag
      existing_filesystem = {
        id = nebius_compute_v1_filesystem.shared[0].id
      }
    }] : []
  }
}
