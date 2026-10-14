# Data-plane access, declared rather than clicked.
#
# Creating a storage account does not grant access to what is inside it. Control-plane
# ownership and data-plane roles are separate in Azure, and the gap is quiet: the first
# upload failed with an authentication error that reads like a login problem and is
# actually a missing role assignment.
#
# The workaround is an account key, which works and is the wrong answer to keep: a key is
# a long-lived shared secret that cannot be scoped or attributed, and it has to be passed
# to anything that needs the data. These assignments remove the need for one.

data "azurerm_role_definition" "blob_contributor" {
  name = "Storage Blob Data Contributor"
}

data "azurerm_role_definition" "blob_reader" {
  name = "Storage Blob Data Reader"
}

# The human running the campaign, so uploads and downloads use their own identity and
# appear in the audit log as theirs.
resource "azurerm_role_assignment" "operator_blob" {
  scope              = azurerm_storage_account.main.id
  role_definition_id = data.azurerm_role_definition.blob_contributor.id
  principal_id       = data.azurerm_client_config.current.object_id
}

# The workspace's own identity, which is what a job runs as. Read-only on purpose: a job
# consumes the cached corpora and writes its results through the workspace's outputs, so
# write access here would let a training run modify the data of record it is reading.
resource "azurerm_role_assignment" "workspace_cache_read" {
  scope              = azurerm_storage_container.cache.resource_manager_id
  role_definition_id = data.azurerm_role_definition.blob_reader.id
  principal_id       = azurerm_machine_learning_workspace.main.identity[0].principal_id
}

# Results, which jobs do write.
resource "azurerm_role_assignment" "workspace_artifacts_write" {
  scope              = azurerm_storage_container.artifacts.resource_manager_id
  role_definition_id = data.azurerm_role_definition.blob_contributor.id
  principal_id       = azurerm_machine_learning_workspace.main.identity[0].principal_id
}

# Pulling the job image. Without it a cluster schedules the job and then fails to start
# it, which looks like a capacity problem and is a permissions one.
resource "azurerm_role_assignment" "workspace_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_machine_learning_workspace.main.identity[0].principal_id
}

output "cache_container" {
  value       = azurerm_storage_container.cache.name
  description = "Where the corpora and feature tensors live; jobs read only from here."
}

# The compute clusters' own identities, which is what actually mounts a datastore.
#
# Granting the workspace identity is not enough, and the failure is misleading: the job
# schedules, the node starts, and the mount fails with "this request is not authorized to
# perform this operation using this permission" — which reads like a storage firewall and
# is a missing role on a different principal. A job runs as its compute, not as its
# workspace.
resource "azurerm_role_assignment" "sweep_cache_read" {
  scope              = azurerm_storage_container.cache.resource_manager_id
  role_definition_id = data.azurerm_role_definition.blob_reader.id
  principal_id       = azurerm_machine_learning_compute_cluster.sweep.identity[0].principal_id
}

resource "azurerm_role_assignment" "measurement_cache_read" {
  scope              = azurerm_storage_container.cache.resource_manager_id
  role_definition_id = data.azurerm_role_definition.blob_reader.id
  principal_id       = azurerm_machine_learning_compute_cluster.measurement.identity[0].principal_id
}

# Results are written through the workspace's default store rather than this container, so
# the clusters need contributor on the artifacts container only if a job targets it
# directly. Granted, because the canary does.
resource "azurerm_role_assignment" "sweep_artifacts_write" {
  scope              = azurerm_storage_container.artifacts.resource_manager_id
  role_definition_id = data.azurerm_role_definition.blob_contributor.id
  principal_id       = azurerm_machine_learning_compute_cluster.sweep.identity[0].principal_id
}

# The workspace's default store, which is where a job's outputs and snapshot land. Without
# it the job cannot write what it produced, which fails at the end rather than the start
# and after the compute has been paid for.
resource "azurerm_role_assignment" "sweep_account_read" {
  scope              = azurerm_storage_account.main.id
  role_definition_id = data.azurerm_role_definition.blob_contributor.id
  principal_id       = azurerm_machine_learning_compute_cluster.sweep.identity[0].principal_id
}
