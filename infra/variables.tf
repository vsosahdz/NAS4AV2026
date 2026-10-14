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

variable "sweep_sku" {
  description = <<-EOT
    VM size for the spot sweep cluster.

    The work is CPU-bound — these architectures peak in the low millions of parameters on
    batches of four and run about 2.5x faster on CPU than on the GPU the prior campaign
    used — so a CPU SKU is what it wants. It is not what this subscription offers: every
    CPU size Azure ML supports is NotAvailableForSubscription here and in the fallback
    region, because the subscription was provisioned for GPU work. T4 nodes are available.

    So the sweep runs on a T4 node and uses its vCPUs, leaving the GPU idle. Wasteful in
    principle, nearly free on spot, and the same node then serves the one task that does
    want a GPU: fine-tuning the strong-input transformer baseline.
  EOT
  type        = string
  default     = "Standard_NC16as_T4_v3"
}

variable "measurement_sku" {
  description = <<-EOT
    VM size for the dedicated measurement cluster, which is a different size on purpose.

    The dedicated allowance for this family is **4 vCPU** — one small node, total. Spot is
    governed only by the regional low-priority cap. Sizing both clusters alike would make
    the dedicated one impossible, so the measurement cluster takes the smallest T4 and the
    sweep takes a large one.
  EOT
  type        = string
  default     = "Standard_NC4as_T4_v3"
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

variable "measurement_units_per_node" {
  description = "vCPU on one measurement node, which is a smaller SKU than the sweep's."
  type        = number
  default     = 4
}

variable "dedicated_quota_vcpu" {
  description = <<-EOT
    Granted **dedicated** vCPU for the SKU family under Microsoft.MachineLearningServices
    in this region. Read with scripts/azure_check_quota.sh.

    Separate from the low-priority allowance and far smaller: this subscription grants 4
    dedicated against an uncapped low-priority. A single allowance covering both, which an
    earlier version of this configuration assumed, makes the dedicated cluster impossible
    to size.
  EOT
  type        = number
}

variable "lowpriority_quota_vcpu" {
  description = <<-EOT
    Granted **low-priority** vCPU, which for this family is governed by the regional cap
    rather than by a family limit of its own.
  EOT
  type        = number
}

variable "unused_family_quota_vcpu" {
  description = <<-EOT
    Retained so an existing target.tfvars does not break. Superseded by the split
    dedicated and low-priority allowances above.

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
  default     = 0
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
