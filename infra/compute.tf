# Compute clusters.
#
# Two of them, because the campaign has two kinds of work with incompatible needs.
#
# The sweep is 165 independent units, each writing its own artifact the moment it
# finishes and each treating a truncated file as incomplete. A preemption therefore
# costs at most the unit in flight, which is what makes spot close to free here. That
# property was built for interruption on a laptop, before Azure was on the table, and it
# happens to be exactly what spot compute asks for.
#
# The canary and any rate measurement, by contrast, exist to *measure* per-unit
# wall-clock. Running those on spot would let preemption contaminate the very numbers
# they establish, which then feed the cost projection for everything after.
#
# Both scale to zero. An idle cluster with a non-zero minimum is the single largest
# avoidable cost on a project like this, and it is the mistake that does not announce
# itself.

locals {
  # Counted against two allowances, not one. Azure ML grants dedicated and low-priority
  # vCPU separately for the same family, and validates each cluster's maximum when the
  # cluster is created rather than against running cores — so a configuration that fits
  # one allowance can still be refused against the other, with the counter reading zero
  # used.
  sweep_vcpu       = var.sweep_max_nodes * var.units_per_node
  measurement_vcpu = var.measurement_max_nodes * var.measurement_units_per_node
}

resource "null_resource" "quota_arithmetic" {
  lifecycle {
    precondition {
      condition     = local.sweep_vcpu <= var.lowpriority_quota_vcpu
      error_message = <<-EOT
        The spot cluster exceeds the granted low-priority allowance.

          ${var.sweep_max_nodes} nodes x ${var.units_per_node} vCPU = ${local.sweep_vcpu}
          granted                                                   = ${var.lowpriority_quota_vcpu}
      EOT
    }
    precondition {
      condition     = local.measurement_vcpu <= var.dedicated_quota_vcpu
      error_message = <<-EOT
        The dedicated cluster exceeds the granted dedicated allowance, which is much
        smaller than the low-priority one and is the constraint that actually binds here.

          ${var.measurement_max_nodes} nodes x ${var.measurement_units_per_node} vCPU = ${local.measurement_vcpu}
          granted                                                                      = ${var.dedicated_quota_vcpu}

        Read both with scripts/azure_check_quota.sh, from the Azure ML counter rather than
        from `az vm list-usage`, which reports a separately granted Microsoft.Compute
        allowance for the same family.
      EOT
    }
  }
}

resource "azurerm_machine_learning_compute_cluster" "sweep" {
  depends_on = [null_resource.quota_arithmetic]

  name                          = "sweep-spot"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  location                      = azurerm_resource_group.main.location
  vm_size                       = var.sweep_sku
  vm_priority                   = "LowPriority"

  scale_settings {
    # Zero is not a default here, it is the guarantee: no node exists unless a unit
    # needs one.
    min_node_count = 0

    # Set to zero to park the platform between waves: allocation becomes impossible
    # while every resource stays intact.
    max_node_count = var.sweep_max_nodes

    # While a wave is queued the cluster is never idle, so this window only governs the
    # tail after the last unit. Short, because that tail is pure waste.
    scale_down_nodes_after_idle_duration = var.sweep_idle_timeout
  }

  identity {
    type = "SystemAssigned"
  }

  tags = merge(local.tags, {
    role     = "sweep"
    priority = "spot"
  })
}

resource "azurerm_machine_learning_compute_cluster" "measurement" {
  # Ordering is load-bearing, not cosmetic. Both clusters draw on the same family
  # allowance, and Azure ML checks each maximum at creation. Without the dependency
  # Terraform is free to create this one first, against a sweep cluster still reserving
  # the whole allowance.
  depends_on = [azurerm_machine_learning_compute_cluster.sweep]

  name                          = "measure-dedicated"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  location                      = azurerm_resource_group.main.location
  vm_size                       = var.measurement_sku
  vm_priority                   = "Dedicated"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = var.measurement_max_nodes
    scale_down_nodes_after_idle_duration = var.measurement_idle_timeout
  }

  identity {
    type = "SystemAssigned"
  }

  tags = merge(local.tags, {
    role     = "measurement"
    priority = "dedicated"
  })
}

output "sweep_cluster" {
  value       = azurerm_machine_learning_compute_cluster.sweep.name
  description = "Cluster the fan-out submits to."
}

output "measurement_cluster" {
  value       = azurerm_machine_learning_compute_cluster.measurement.name
  description = "Cluster for anything whose purpose is to measure duration."
}

output "concurrency" {
  value       = var.sweep_max_nodes * var.units_per_node
  description = "Units that can run at once when the sweep cluster is fully scaled out."
}
