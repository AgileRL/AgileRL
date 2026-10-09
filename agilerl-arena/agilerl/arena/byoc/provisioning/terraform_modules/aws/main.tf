terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.58.0"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
  enable_pod_identity = (
    var.provision_shared_filesystem || var.enable_gateway_api || length(var.workers) > 0
  )
  # Unpacked CUDA train image plus content-store blobs, with kubelet's 10% eviction reserve.
  node_disk_size_gib = 100
  # Kubelet registers this taint. The autoscaler treats the prefix as still starting.
  gpu_worker_user_data = <<-EOT
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="BOUNDARY"

    --BOUNDARY
    Content-Type: application/node.eks.aws

    ---
    apiVersion: node.eks.aws/v1alpha1
    kind: NodeConfig
    spec:
      kubelet:
        config:
          registerWithTaints:
            - key: startup-taint.cluster-autoscaler.kubernetes.io/nvidia-gpu
              value: "true"
              effect: NoSchedule
    --BOUNDARY--
    EOT
}

resource "aws_vpc" "arena" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = var.cluster_name
  }
}

resource "aws_internet_gateway" "arena" {
  vpc_id = aws_vpc.arena.id

  tags = {
    Name = var.cluster_name
  }
}

resource "aws_subnet" "arena" {
  count                   = 2
  vpc_id                  = aws_vpc.arena.id
  cidr_block              = cidrsubnet(aws_vpc.arena.cidr_block, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                        = "${var.cluster_name}-${local.azs[count.index]}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }
}

resource "aws_route_table" "arena" {
  vpc_id = aws_vpc.arena.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.arena.id
  }
}

resource "aws_route_table_association" "arena" {
  count          = 2
  subnet_id      = aws_subnet.arena[count.index].id
  route_table_id = aws_route_table.arena.id
}

resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-eks-cluster"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cluster" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_eks_cluster" "arena" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = var.kubernetes_version
  # Cilium is the CNI and replaces kube-proxy. CoreDNS is installed after Cilium.
  bootstrap_self_managed_addons = false

  vpc_config {
    subnet_ids              = aws_subnet.arena[*].id
    endpoint_public_access  = true
    endpoint_private_access = false
  }

  depends_on = [aws_iam_role_policy_attachment.cluster]
}

resource "aws_iam_role" "node" {
  name = "${var.cluster_name}-eks-node"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "node" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
  ])

  role       = aws_iam_role.node.name
  policy_arn = each.value
}

# Cilium ENI IPAM reads route tables. The VPC CNI policy does not grant that.
resource "aws_iam_role_policy" "cilium_eni" {
  name = "${var.cluster_name}-cilium-eni"
  role = aws_iam_role.node.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ec2:DescribeRouteTables"]
      Resource = "*"
    }]
  })
}

resource "aws_eks_node_group" "arena" {
  cluster_name    = aws_eks_cluster.arena.name
  node_group_name = "arena"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.arena[*].id
  instance_types  = [var.arena_instance_type]
  disk_size       = local.node_disk_size_gib

  scaling_config {
    min_size     = var.arena_min_node_count
    max_size     = var.arena_max_node_count
    desired_size = var.arena_min_node_count
  }

  # The node group changes desired size after apply.
  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  depends_on = [aws_iam_role_policy_attachment.node]
}

resource "aws_launch_template" "gpu_worker" {
  count = length([for worker in var.workers : worker if worker.num_gpus > 0]) > 0 ? 1 : 0
  name  = "${var.cluster_name}-gpu-worker"

  # Pod identity reaches IMDS from the pod network namespace.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = local.node_disk_size_gib
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  user_data = base64encode(local.gpu_worker_user_data)
}

resource "aws_eks_node_group" "worker" {
  for_each = { for worker in var.workers : worker.name => worker }

  cluster_name    = aws_eks_cluster.arena.name
  node_group_name = each.value.name
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = aws_subnet.arena[*].id
  instance_types  = [each.value.instance_type]
  disk_size       = each.value.num_gpus > 0 ? null : local.node_disk_size_gib
  ami_type        = each.value.num_gpus > 0 ? "AL2023_x86_64_NVIDIA" : "AL2023_x86_64_STANDARD"

  dynamic "launch_template" {
    for_each = each.value.num_gpus > 0 ? aws_launch_template.gpu_worker : []
    content {
      id      = launch_template.value.id
      version = launch_template.value.latest_version
    }
  }
  # The device plugin chart only schedules onto nodes carrying this label.
  labels = each.value.num_gpus > 0 ? {
    "nvidia.com/gpu.present" = "true"
  } : {}

  scaling_config {
    min_size     = each.value.min_node_count
    max_size     = each.value.max_node_count
    desired_size = each.value.min_node_count
  }

  dynamic "taint" {
    for_each = each.value.num_gpus > 0 ? [1] : []
    content {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  depends_on = [aws_iam_role_policy_attachment.node]
}

# The autoscaler reads the ASG. Node-group tags also land on the system group.
locals {
  worker_autoscaler_tags = merge([
    for worker in var.workers : merge(
      {
        "${worker.name}/enabled" = {
          group = worker.name
          key   = "k8s.io/cluster-autoscaler/enabled"
          value = "true"
        }
        "${worker.name}/owned" = {
          group = worker.name
          key   = "k8s.io/cluster-autoscaler/${var.cluster_name}"
          value = "owned"
        }
      },
      worker.num_gpus > 0 ? {
        "${worker.name}/label" = {
          group = worker.name
          key   = "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu.present"
          value = "true"
        }
        "${worker.name}/taint" = {
          group = worker.name
          key   = "k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu"
          value = "true:NoSchedule"
        }
        "${worker.name}/gpu" = {
          group = worker.name
          key   = "k8s.io/cluster-autoscaler/node-template/resources/nvidia.com/gpu"
          value = tostring(worker.num_gpus)
        }
      } : {}
    )
  ]...)
}

resource "aws_autoscaling_group_tag" "worker_autoscaler" {
  for_each = local.worker_autoscaler_tags

  autoscaling_group_name = aws_eks_node_group.worker[each.value.group].resources[0].autoscaling_groups[0].name

  tag {
    key                 = each.value.key
    value               = each.value.value
    propagate_at_launch = true
  }
}

resource "aws_efs_file_system" "arena" {
  count          = var.provision_shared_filesystem ? 1 : 0
  creation_token = var.cluster_name
  encrypted      = true

  tags = {
    Name = "${var.cluster_name}-shared"
  }
}

resource "aws_security_group" "efs" {
  count  = var.provision_shared_filesystem ? 1 : 0
  name   = "${var.cluster_name}-efs"
  vpc_id = aws_vpc.arena.id

  ingress {
    description     = "NFS from the EKS cluster"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_eks_cluster.arena.vpc_config[0].cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_efs_mount_target" "arena" {
  count           = var.provision_shared_filesystem ? 2 : 0
  file_system_id  = aws_efs_file_system.arena[0].id
  subnet_id       = aws_subnet.arena[count.index].id
  security_groups = [aws_security_group.efs[0].id]
}

resource "aws_iam_role" "efs_csi" {
  count = var.provision_shared_filesystem ? 1 : 0
  name  = "${var.cluster_name}-efs-csi"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "efs_csi" {
  count      = var.provision_shared_filesystem ? 1 : 0
  role       = aws_iam_role.efs_csi[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy"
}

resource "aws_eks_addon" "pod_identity" {
  count        = local.enable_pod_identity ? 1 : 0
  cluster_name = aws_eks_cluster.arena.name
  addon_name   = "eks-pod-identity-agent"

  depends_on = [aws_eks_node_group.arena]
}

resource "aws_iam_role" "load_balancer_controller" {
  count = var.enable_gateway_api ? 1 : 0
  name  = "${var.cluster_name}-lbc"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_policy" "load_balancer_controller" {
  count  = var.enable_gateway_api ? 1 : 0
  name   = "${var.cluster_name}-lbc"
  policy = file("${path.module}/aws_load_balancer_controller_iam.json")
}

resource "aws_iam_role_policy_attachment" "load_balancer_controller" {
  count      = var.enable_gateway_api ? 1 : 0
  role       = aws_iam_role.load_balancer_controller[0].name
  policy_arn = aws_iam_policy.load_balancer_controller[0].arn
}

resource "aws_iam_role" "cluster_autoscaler" {
  count = length(var.workers) > 0 ? 1 : 0
  name  = "${var.cluster_name}-cluster-autoscaler"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_role_policy" "cluster_autoscaler" {
  count = length(var.workers) > 0 ? 1 : 0
  name  = "${var.cluster_name}-cluster-autoscaler"
  role  = aws_iam_role.cluster_autoscaler[0].name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeScalingActivities",
          "ec2:DescribeImages",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplateVersions",
          "ec2:GetInstanceTypesFromInstanceRequirements",
          "eks:DescribeNodegroup",
        ]
        Resource = ["*"]
      },
      {
        Effect = "Allow"
        Action = [
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup",
        ]
        Resource = ["*"]
        Condition = {
          StringEquals = {
            "autoscaling:ResourceTag/k8s.io/cluster-autoscaler/enabled" = "true"
          }
        }
      },
    ]
  })
}

resource "aws_eks_pod_identity_association" "cluster_autoscaler" {
  count           = length(var.workers) > 0 ? 1 : 0
  cluster_name    = aws_eks_cluster.arena.name
  namespace       = "kube-system"
  service_account = "cluster-autoscaler"
  role_arn        = aws_iam_role.cluster_autoscaler[0].arn

  depends_on = [aws_eks_addon.pod_identity]
}

resource "aws_eks_pod_identity_association" "load_balancer_controller" {
  count           = var.enable_gateway_api ? 1 : 0
  cluster_name    = aws_eks_cluster.arena.name
  namespace       = "kube-system"
  service_account = "aws-load-balancer-controller"
  role_arn        = aws_iam_role.load_balancer_controller[0].arn

  depends_on = [aws_eks_addon.pod_identity]
}

resource "aws_eks_pod_identity_association" "efs_csi_controller" {
  count           = var.provision_shared_filesystem ? 1 : 0
  cluster_name    = aws_eks_cluster.arena.name
  namespace       = "kube-system"
  service_account = "efs-csi-controller-sa"
  role_arn        = aws_iam_role.efs_csi[0].arn

  depends_on = [aws_eks_addon.pod_identity]
}

resource "aws_eks_pod_identity_association" "efs_csi_node" {
  count           = var.provision_shared_filesystem ? 1 : 0
  cluster_name    = aws_eks_cluster.arena.name
  namespace       = "kube-system"
  service_account = "efs-csi-node-sa"
  role_arn        = aws_iam_role.efs_csi[0].arn

  depends_on = [aws_eks_addon.pod_identity]
}

resource "aws_eks_addon" "efs_csi" {
  count                       = var.provision_shared_filesystem ? 1 : 0
  cluster_name                = aws_eks_cluster.arena.name
  addon_name                  = "aws-efs-csi-driver"
  resolve_conflicts_on_create = "OVERWRITE"

  depends_on = [
    aws_eks_pod_identity_association.efs_csi_controller,
    aws_eks_pod_identity_association.efs_csi_node,
    aws_efs_mount_target.arena,
  ]
}

resource "aws_s3_bucket" "arena_data" {
  bucket        = var.storage_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "arena_data" {
  bucket = aws_s3_bucket.arena_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_user" "storage" {
  name = "${var.cluster_name}-storage"
}

resource "aws_iam_user_policy" "storage" {
  name = "${var.cluster_name}-storage"
  user = aws_iam_user.storage.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:*"]
      Resource = [
        aws_s3_bucket.arena_data.arn,
        "${aws_s3_bucket.arena_data.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_access_key" "storage" {
  user = aws_iam_user.storage.name
}
