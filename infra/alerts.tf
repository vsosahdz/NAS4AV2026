# Cost guard.
#
# The projected campaign is a few dollars of spot CPU, so an alert here does not mean the
# campaign is proceeding — it means something is wrong. That is the point of setting the
# budget well above the projection rather than close to it: a threshold that fires during
# normal operation is a threshold people learn to ignore.

resource "azurerm_consumption_budget_resource_group" "main" {
  name              = "${local.prefix}-budget"
  resource_group_id = azurerm_resource_group.main.id

  amount     = var.budget_amount_usd
  time_grain = "Monthly"

  time_period {
    # Must be the first of a month at or before the current one. A future start date
    # creates a budget that covers nothing while appearing to be in place, which is
    # worse than having none.
    start_date = var.budget_start_date
  }

  # Actual spend, at half the budget. Early enough to investigate while the campaign is
  # still running.
  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = var.budget_contact_emails
  }

  # Forecast rather than actual, so the warning arrives before the money is spent rather
  # than after. This is the one that catches a cluster left scaled up overnight.
  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    contact_emails = var.budget_contact_emails
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = var.budget_contact_emails
  }
}

output "budget_usd" {
  value       = var.budget_amount_usd
  description = "Monthly ceiling; alerts at 50% actual, 80% forecast, 100% actual."
}
