output "cluster_name" {
  value = aws_eks_cluster.arena.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.arena.endpoint
}

output "cluster_id" {
  value = aws_eks_cluster.arena.name
}

output "context" {
  value = aws_eks_cluster.arena.name
}

output "region" {
  value = var.region
}

output "vpc_id" {
  value = aws_vpc.arena.id
}

output "storage_class_name" {
  value = var.provision_shared_filesystem ? "arena-shared" : null
}

output "efs_file_system_id" {
  value = one(aws_efs_file_system.arena[*].id)
}

output "storage_endpoint" {
  value = var.storage_endpoint
}

output "storage_bucket" {
  value = aws_s3_bucket.arena_data.bucket
}

output "storage_access_key_id" {
  value = aws_iam_access_key.storage.id
}

output "storage_secret_access_key" {
  value     = aws_iam_access_key.storage.secret
  sensitive = true
}

output "worker_node_group_ids" {
  value = {
    for name, group in aws_eks_node_group.worker : name => group.node_group_name
  }
}
