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
# VERIFY BEFORE APPLY: that the chosen CPU SKU is unrestricted in this region and that
# the Azure ML vCPU allowance covers the cluster maximums below. Read the enforced
# counter with scripts/azure_check_quota.sh, not with `az vm list-usage` — Azure ML keeps
# its own allowance for the same family and the two are granted independently.
location = "southcentralus"

# The SKU the sweep runs on.
#
# CPU, not GPU, and that is a measurement rather than a preference: these architectures
# peak in the low millions of parameters on batches of four, and run about 2.5x faster on
# a laptop CPU than on the GTX 1650 the prior campaign used, because at this size kernel
# launch overhead dominates. See design.md D22.
#
# One unit per core, so a 16-vCPU node runs sixteen units concurrently.
cpu_sku = "Standard_F16s_v2"

# Cluster sizing. The maximums must sum to within the granted Azure ML allowance for the
# family in this region; the plan checks the arithmetic so an oversized request fails
# with a readable message instead of as ClusterMinNodesExceedCoreQuota at creation.
#
#   (4 + 1) nodes x 16 vCPU = 80
family_quota_vcpu     = 80
sweep_max_nodes       = 4
measurement_max_nodes = 1

# How many units one node runs at once. One per core: each evaluation is single-threaded
# by construction, since OMP_NUM_THREADS is pinned to 1 so that float64 reduction order
# does not vary with the thread count.
units_per_node = 16

# Cost guard. The whole comparison is 99 measured single-core hours, which is a few
# dollars on spot CPU; the budget is set well above it so an alert means something went
# wrong rather than that the campaign is proceeding.
budget_amount_usd     = 50
budget_start_date     = "2026-09-01T00:00:00Z" # the current month — a future date leaves the campaign uncovered
budget_contact_emails = ["vsosa@tec.mx"]
owner_tag             = "vsosa@tec.mx"

# Lifecycle. No node exists without work: both clusters sit at zero, and these windows
# govern only the tail after the last unit of a wave. A per-job timeout catches what an
# idle window cannot — a hung job holds its node busy forever.
sweep_idle_timeout       = "PT5M"
measurement_idle_timeout = "PT2M"
job_timeout_minutes      = 90
