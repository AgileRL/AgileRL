variable "cluster_name" {
  type = string
}

variable "tenant_id" {
  type = string
}

variable "region" {
  type    = string
  default = ""
}

variable "project_id" {
  type    = string
  default = ""
}

variable "storage_project_id" {
  type    = string
  default = ""
}

variable "subnet_id" {
  type    = string
  default = ""
}

variable "service_account_id" {
  type    = string
  default = ""
}

variable "storage_bucket_name" {
  type = string
}

variable "storage_endpoint" {
  type = string
}

variable "kubernetes_version" {
  type    = string
  default = "1.35"
}

variable "etcd_cluster_size" {
  type    = number
  default = 3
}

variable "arena_min_node_count" {
  type    = number
  default = 2
}

variable "arena_max_node_count" {
  type    = number
  default = 10
}

variable "arena_platform" {
  type    = string
  default = "cpu-e2"
}

variable "arena_preset" {
  type    = string
  default = "2vcpu-8gb"
}

variable "workers" {
  type = list(object({
    name               = string
    min_node_count     = number
    max_node_count     = number
    platform           = string
    preset             = string
    gpu_drivers_preset = string
  }))
}

variable "object_storage_size_gib" {
  type = number
}

variable "provision_shared_filesystem" {
  type    = bool
  default = true
}

variable "filesystem_type" {
  type    = string
  default = "NETWORK_SSD"
}

variable "filesystem_size_gib" {
  type    = number
  default = 256
}

variable "filesystem_mount_tag" {
  type    = string
  default = "csi-storage"
}
