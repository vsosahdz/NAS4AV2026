# Inputs. Defaults come from infra/target.tfvars, which is committed on purpose: it
# carries no secret and it is what the subscription guard checks against.

variable "subscription_name" {
  description = <<-EOT
    Display name of the subscription this campaign is pinned to. Asserted before any
    resource is created. Pinned by name rather than id so that no subscription
    identifier is committed — the provider resolves the id from the CLI context.
  EOT
  type        = string
}

variable "location" {
  description = <<-EOT
    Azure region. Storage, workspace and both clusters share it; cross-region traffic
    costs egress and adds latency to every job.
  EOT
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]+$", var.location))
    error_message = "Use the lowercase region short name, for example southcentralus."
  }
}

variable "cpu_sku" {
  description = <<-EOT
    VM size for both clusters. CPU rather than GPU, by measurement rather than by
    preference: these architectures peak in the low millions of parameters on batches of
    four and run about 2.5x faster on CPU than on the GPU the prior campaign used,
    because at this size kernel launch overhead dominates the arithmetic.

    A GPU SKU here would cost more, contend for scarcer quota, and run slower.
  EOT
  type        = string
  default     = "Standard_F16s_v2"
}

variable "units_per_node" {
  description = <<-EOT
    Concurrent work units per node. One per core: every evaluation is single-threaded by
    construction, because OMP_NUM_THREADS is pinned to 1 so that float64 reduction order
    does not vary with the thread count and results stay reproducible.
  EOT
  type        = number
  default     = 16

  validation {
    condition     = var.units_per_node >= 1
    error_message = "At least one unit per node."
  }
}

variable "family_quota_vcpu" {
  description = <<-EOT
    Granted vCPU for the SKU's family under **Microsoft.MachineLearningServices** in this
    region — not the Microsoft.Compute quota that `az vm list-usage` reports. Azure ML
    enforces its own counter for the same family and the two are granted independently;
    read the enforced one with scripts/azure_check_quota.sh.

    Cluster maximums are checked against this at plan time, so an oversized request fails
    with arithmetic rather than as ClusterMinNodesExceedCoreQuota at creation. That
    failure mode is worth guarding against specifically: it reports the family allowance
    as unused while refusing the request, because the validation happens at creation
    rather than against running cores.
  EOT
  type        = number
}

variable "sweep_max_nodes" {
  description = <<-EOT
    Maximum spot nodes for the sweep. The whole three-strategy comparison is 99 measured
    single-core hours, so four 16-vCPU nodes clear it in under two hours of wall clock.
  EOT
  type        = number
  default     = 4

  validation {
    condition     = var.sweep_max_nodes >= 1
    error_message = "At least one node."
  }
}

variable "measurement_max_nodes" {
  description = <<-EOT
    Maximum dedicated nodes for measurement work. One, deliberately: this cluster exists
    to measure per-unit wall-clock, so serial execution on a single node removes any
    doubt about contention in the numbers that set every projection downstream.
  EOT
  type        = number
  default     = 1

  validation {
    condition     = var.measurement_max_nodes >= 1
    error_message = "At least one node."
  }
}

variable "sweep_idle_timeout" {
  description = <<-EOT
    How long a sweep node waits for work before scaling down, as an ISO-8601 duration.
    While a wave is queued the cluster is never idle, so this governs only the tail after
    the last unit — which is pure waste.
  EOT
  type        = string
  default     = "PT5M"
}

variable "measurement_idle_timeout" {
  type        = string
  description = "Idle window for the measurement cluster, as an ISO-8601 duration."
  default     = "PT2M"
}

variable "job_timeout_minutes" {
  description = <<-EOT
    Wall-clock limit on a single job. An idle window cannot catch a hung job — it holds
    its node busy forever and bills until somebody notices, which at three in the morning
    is nobody.

    Set from the measured per-unit cost rather than guessed: the slowest setting is about
    4.3 s per evaluation at a budget of 1,200, so a unit is roughly 90 minutes at one per
    core, and this bound leaves room for a slower node without protecting a hang.
  EOT
  type        = number
  default     = 90

  validation {
    condition     = var.job_timeout_minutes > 0
    error_message = "A job must have a positive time limit; an unbounded job bills until noticed."
  }
}

variable "budget_amount_usd" {
  description = <<-EOT
    Monthly budget for the resource group. Set well above the projected cost, so that an
    alert means something went wrong rather than that the campaign is proceeding.
  EOT
  type        = number

  validation {
    condition     = var.budget_amount_usd > 0
    error_message = "A budget of zero disables the guard rather than tightening it."
  }
}

variable "budget_start_date" {
  description = <<-EOT
    First day of the budget period, RFC 3339. Must be the first of a month at or before
    the current one — a future start date creates a budget that covers nothing while
    appearing to be in place.
  EOT
  type        = string

  validation {
    condition     = can(regex("^[0-9]{4}-[0-9]{2}-01T00:00:00Z$", var.budget_start_date))
    error_message = "Use the first of a month, for example 2026-09-01T00:00:00Z."
  }
}

variable "budget_contact_emails" {
  type        = list(string)
  description = "Who is told when a budget threshold is crossed."

  validation {
    condition     = length(var.budget_contact_emails) > 0
    error_message = "A budget with no contact is a budget nobody hears about."
  }
}

variable "owner_tag" {
  type        = string
  description = "Owner tag applied to every resource, so an orphan can be traced back."
}
