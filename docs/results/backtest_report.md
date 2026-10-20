# LeadTime backtest report

```json
{
  "backend": "lightgbm",
  "quantiles": [
    0.1,
    0.5,
    0.8,
    0.9
  ],
  "n_orders": 39105,
  "folds": 6,
  "horizon_days": 30,
  "label_gap_days": 21,
  "train_seconds": 441.0,
  "features": 46
}
```

**Delivery-date MAE (median forecast): 1.99 days vs carrier estimate 2.86 days (-31%).**

## Rolling-origin backtest (mean over folds)

| stage | model | mae | bias | pinball_mean | coverage_q10 | coverage_q50 | coverage_q80 | coverage_q90 |
|---|---|---|---|---|---|---|---|---|
| approval | lightgbm_segmented | 0.084 | -0.048 | 0.035 | 0.114 | 0.470 | 0.793 | 0.922 |
| approval | route_quantile_baseline | 0.326 | -0.272 | 0.131 | 0.101 | 0.496 | 0.798 | 0.900 |
| handling | lightgbm_segmented | 1.229 | -0.309 | 0.441 | 0.162 | 0.523 | 0.767 | 0.860 |
| handling | route_quantile_baseline | 1.634 | -0.701 | 0.614 | 0.095 | 0.492 | 0.785 | 0.889 |
| total | carrier_estimate | 2.865 | 0.570 | nan | nan | nan | nan | nan |
| total | lightgbm_segmented | 1.987 | -0.572 | 0.732 | 0.160 | 0.503 | 0.761 | 0.861 |
| total | route_quantile_baseline | 2.527 | -0.969 | 0.927 | 0.098 | 0.489 | 0.791 | 0.888 |
| total | sum_of_stage_medians | 1.979 | nan | nan | nan | nan | nan | nan |
| transit | lightgbm_segmented | 1.243 | -0.439 | 0.475 | 0.149 | 0.500 | 0.756 | 0.867 |
| transit | route_quantile_baseline | 1.489 | -0.585 | 0.560 | 0.104 | 0.512 | 0.817 | 0.912 |

## Calibration of the end-to-end forecast (pooled out-of-sample)

| quantile | empirical_coverage | gap |
|---|---|---|
| 0.100 | 0.159 | 0.059 |
| 0.500 | 0.502 | 0.002 |
| 0.800 | 0.762 | -0.038 |
| 0.900 | 0.861 | -0.039 |

## Promise policies (same out-of-sample orders)

| policy | quantile | buffer_days | n | on_time_rate | late_rate | mean_promised_days | median_promised_days | mean_slack_days | promise_mae | mean_days_late_when_late |
|---|---|---|---|---|---|---|---|---|---|---|
| carrier_estimate | nan | 0.000 | 14965 | 0.660 | 0.340 | 8.697 | 8.213 | 0.570 | 2.863 | 3.369 |
| model_p50 | 0.500 | 0.000 | 14965 | 0.602 | 0.398 | 8.054 | 7.000 | -0.073 | 2.056 | 2.673 |
| model_p80 | 0.800 | 0.000 | 14965 | 0.816 | 0.184 | 9.785 | 9.000 | 1.658 | 2.776 | 3.032 |
| composed_p80 | 0.800 | 0.000 | 14965 | 0.841 | 0.159 | 10.107 | 9.000 | 1.980 | 2.967 | 3.099 |
| model_p80_calibrated | 0.800 | -0.154 | 14965 | 0.800 | 0.200 | 9.633 | 9.000 | 1.507 | 2.687 | 2.954 |

## A/B simulation (lead-time matched)

model_p80 vs carrier_estimate: late rate 31.167% vs 33.338% (-6.5%, 95% CI [-10.8%, -2.3%], p=4.52e-03) at mean promised lead time 8.70 vs 8.67 days (treatment: P61 -0.00 d)

Required sample size per arm for this effect at alpha=0.05, power=0.8: **7,281** orders.

## Feature importance (top 15)

| feature | importance |
|---|---|
| product_category | 1053.7 |
| seller_hist_handling_mean | 600.3 |
| customer_state | 557.7 |
| shipping_limit_days | 547.0 |
| distance_km | 542.8 |
| seller_hist_handling_mean_30d | 448.2 |
| seller_hist_transit_mean_90d | 445.7 |
| seller_load_ratio | 443.8 |
| seller_hist_handling_mean_90d | 420.2 |
| days_to_black_friday | 403.2 |
| carrier_estimate_days | 349.7 |
| total_freight | 349.3 |
| freight_ratio | 342.7 |
| total_volume_cm3 | 327.8 |
| total_price | 325.5 |

## Figures

![mae_by_fold.png](mae_by_fold.png)
![calibration.png](calibration.png)
![promise_frontier.png](promise_frontier.png)
![feature_importance.png](feature_importance.png)
