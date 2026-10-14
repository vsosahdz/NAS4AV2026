# The single source of truth for where this campaign runs.
#
# Pinned by display name, not by id: the name is not sensitive, it is readable, and
# Terraform resolves it to an id from the CLI context at apply time. No subscription
# guid is ever committed.
#
# Every script and the Terraform wrapper assert the active subscription matches this
# name before doing anything. Creating resources against the wrong subscription would
# bill the wrong budget, so it is guarded rather than trusted.

subscription_name = "Z_T_de_cancer_de_mama_con_imagenes_de_alta_calidad_VASG"

# Storage, workspace and both clusters live here — cross-region traffic costs egress and
# adds latency to every job.
#
# VERIFIED 2026-10-12 with scripts/azure_check_quota.sh.
location = "southcentralus"

# The SKU the sweep runs on.
#
# CPU, not GPU, and that is a measurement rather than a preference: these architectures
# peak in the low millions of parameters on batches of four, and run about 2.5x faster on
# a laptop CPU than on the GTX 1650 the prior campaign used, because at this size kernel
# launch overhead dominates. See design.md D22.
#
# The work is CPU-bound, and this subscription has no CPU to offer Azure ML. Every
# supported CPU size is NotAvailableForSubscription here and in eastus2: the subscription
# was provisioned for GPU work. T4 nodes are available, so the sweep runs on one and uses
# its vCPUs with the GPU idle — wasteful in principle, nearly free on spot, and the same
# node then serves the transformer baseline, which is the one task that wants a GPU.
#
# Two sizes, because the two allowances are nothing alike. Dedicated is 4 vCPU for this
# family — one small node, total — while low-priority is governed only by the regional
# cap of 300. Sizing both clusters alike would make the dedicated one impossible.
sweep_sku       = "Standard_NC16as_T4_v3"
measurement_sku = "Standard_NC4as_T4_v3"

# Cluster sizing. The maximums must sum to within the granted Azure ML allowance for the
# family in this region; the plan checks the arithmetic so an oversized request fails
# with a readable message instead of as ClusterMinNodesExceedCoreQuota at creation.
#
#   spot:      4 nodes x 16 vCPU =  64, against 300 regional low-priority
#   dedicated: 1 node  x  4 vCPU =   4, against 4 granted — exactly the allowance
lowpriority_quota_vcpu = 300
dedicated_quota_vcpu   = 4
sweep_max_nodes        = 4
measurement_max_nodes  = 1

# How many units one node runs at once. One per core: each evaluation is single-threaded
# by construction, since OMP_NUM_THREADS is pinned to 1 so that float64 reduction order
# does not vary with the thread count.
units_per_node             = 16
measurement_units_per_node = 4

# Cost guard. The whole comparison is 99 measured single-core hours, which is a few
# dollars on spot CPU; the budget is set well above it so an alert means something went
# wrong rather than that the campaign is proceeding.
budget_amount_usd     = 50
budget_start_date     = "2026-10-01T00:00:00Z" # the current month — a future date leaves the campaign uncovered
budget_contact_emails = ["vsosa@tec.mx"]
owner_tag             = "vsosa@tec.mx"

# Lifecycle. No node exists without work: both clusters sit at zero, and these windows
# govern only the tail after the last unit of a wave. A per-job timeout catches what an
# idle window cannot — a hung job holds its node busy forever.
sweep_idle_timeout       = "PT5M"
measurement_idle_timeout = "PT2M"
job_timeout_minutes      = 90
