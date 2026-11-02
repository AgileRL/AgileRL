output "cluster_name" {
  value = nebius_mk8s_v1_cluster.arena.name
}

output "cluster_id" {
  value = nebius_mk8s_v1_cluster.arena.id
}

output "project_id" {
  value = local.project_id
}

output "storage_project_id" {
  value = local.storage_project_id
}

output "subnet_id" {
  value = local.subnet_id
}

output "context" {
  value = var.cluster_name
}

output "storage_class_name" {
  value = var.provision_shared_filesystem ? "arena-shared" : null
}

output "storage_endpoint" {
  value = var.storage_endpoint
}

output "storage_bucket" {
  value = nebius_storage_v1_bucket.arena_data.name
}

output "storage_access_key_id" {
  value = nebius_iam_v2_access_key.storage.status.aws_access_key_id
}

output "storage_secret_access_key" {
  value     = nebius_iam_v2_access_key.storage.status.secret
  sensitive = true
}

output "worker_node_group_ids" {
  value = {
    for name, group in nebius_mk8s_v1_node_group.worker : name => group.id
  }
}

output "service_account_id" {
  value = one(nebius_iam_v1_service_account.arena[*].id)
}

output "service_account_public_key_id" {
  value = one(nebius_iam_v1_auth_public_key.arena[*].id)
}

output "service_account_private_key" {
  value     = one(tls_private_key.arena[*].private_key_pem)
  sensitive = true
}
