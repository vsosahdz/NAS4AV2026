# The durable platform: resource group, storage, workspace, registry, secret store.
#
# Individual runs are never Terraform resources. A campaign submits 165 jobs through the
# workspace SDK and none of them changes state here, so an interrupted sweep leaves the
# platform untouched and a destroyed platform leaves no job half-described.

terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.9"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.3"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      # Refuse to destroy a group that still holds anything Terraform does not know
      # about. The alternative silently deletes results somebody copied in by hand.
      prevent_deletion_if_contains_resources = true
    }
    key_vault {
      # Soft delete is not optional on Key Vault, and a purge on destroy is what lets
      # the platform be rebuilt under the same name. Without it a teardown leaves a
      # tombstone that blocks the next apply.
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

data "azurerm_client_config" "current" {}

data "azurerm_subscription" "current" {}

# The guard that makes the pinned target mean something. Terraform resolves the active
# subscription from the CLI context, so without this an apply would create resources
# wherever `az account show` happens to point — billing a budget that did not authorise
# them, which is quiet and expensive.
resource "null_resource" "subscription_guard" {
  lifecycle {
    precondition {
      condition     = data.azurerm_subscription.current.display_name == var.subscription_name
      error_message = <<-EOT
        The active subscription is not the one this campaign is pinned to.

          active:   ${data.azurerm_subscription.current.display_name}
          expected: ${var.subscription_name}

        Fix with:  az account set --subscription "${var.subscription_name}"
      EOT
    }
  }
}

locals {
  prefix = "nas4av"

  tags = {
    project    = "nas4av"
    owner      = var.owner_tag
    managed_by = "terraform"
    # Neither the code nor the metadata credits a tool as an author; this records what
    # provisioned the resource, which is a different claim.
    purpose = "authorship-verification-nas"
  }
}

# Storage account names are globally unique and allow no punctuation, so a suffix is
# generated once and kept in state rather than derived from something that might change.
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

resource "azurerm_resource_group" "main" {
  depends_on = [null_resource.subscription_guard]

  name     = "${local.prefix}-rg"
  location = var.location
  tags     = local.tags
}

resource "azurerm_storage_account" "main" {
  name                     = "${local.prefix}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  # Results are the only irreplaceable thing here — the compute is fungible and the code
  # is in git. LRS rather than GRS because the campaign is short and the artifacts are
  # downloaded and checksummed before teardown.
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  https_traffic_only_enabled      = true

  blob_properties {
    delete_retention_policy {
      # A week to notice an accidental delete. Long enough to be useful, short enough
      # that a destroyed platform does not keep billing.
      days = 7
    }
  }

  tags = local.tags
}

# Feature tensors and corpora, uploaded once and read from here by every job. The spec
# this follows requires that the pipeline run from cache with no public network, so a
# job that reaches the internet for data is a job that has escaped its own provenance.
resource "azurerm_storage_container" "cache" {
  name                  = "cache"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "artifacts" {
  name                  = "artifacts"
  storage_account_id    = azurerm_storage_account.main.id
  container_access_type = "private"
}

resource "azurerm_key_vault" "main" {
  name                       = "${local.prefix}-kv-${random_string.suffix.result}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false

  tags = local.tags
}

resource "azurerm_container_registry" "main" {
  name                = "${local.prefix}acr${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false

  tags = local.tags
}

resource "azurerm_application_insights" "main" {
  name                = "${local.prefix}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"

  tags = local.tags
}

resource "azurerm_machine_learning_workspace" "main" {
  name                    = "${local.prefix}-ws"
  location                = azurerm_resource_group.main.location
  resource_group_name     = azurerm_resource_group.main.name
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id
  container_registry_id   = azurerm_container_registry.main.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.tags
}

output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "workspace" {
  value = azurerm_machine_learning_workspace.main.name
}

output "storage_account" {
  value = azurerm_storage_account.main.name
}

output "container_registry" {
  value = azurerm_container_registry.main.login_server
}
