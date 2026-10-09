variable "cluster_name" {
  type = string
}

variable "region" {
  type = string
}

variable "storage_bucket_name" {
  type = string
}

variable "storage_endpoint" {
  type = string
}

variable "kubernetes_version" {
  type    = string
  default = "1.36"
}

variable "arena_min_node_count" {
  type    = number
  default = 2
}

variable "arena_max_node_count" {
  type    = number
  default = 10
}

variable "arena_instance_type" {
  type    = string
  default = "t3.small"
}

variable "provision_shared_filesystem" {
  type    = bool
  default = false
}

variable "enable_gateway_api" {
  type    = bool
  default = false
}

variable "workers" {
  type = list(object({
    name           = string
    min_node_count = number
    max_node_count = number
    instance_type  = string
    num_gpus       = number
  }))
  default = []
}
