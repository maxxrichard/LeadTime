# Modelling notes

## Targets

| stage | from | to | typical share of lead time |
|---|---|---|---|
| approval | purchase | payment approval | small (boleto payments dominate the tail) |
| handling | approval | carrier hand-over | seller picking/packing; most controllable |
| transit | carrier hand-over | customer delivery | line-haul + last mile; distance & region |
| total | purchase | customer delivery | what the promise is made on |

Targets are modelled on `log1p(days)`: quantiles are equivariant under monotone transforms, so the
back-transformed prediction is still the exact quantile, and the skew of the raw durations is
removed for the trees.

## Features (see `leadtime.features.build.FEATURE_COLUMNS`)

* **seller** – point-in-time mean handling time (all-time, 90 d, 30 d), 90 d SLA-miss rate against
  the shipping limit, 90 d transit mean, tenure, order volume in 1/7/30 d, open backlog (purchased but
  not yet handed over), load ratio (7 d volume vs 30 d run-rate)
* **product** – category, items, weight, volume, price, freight, freight ratio, instalments
* **route** – origin/destination state and region, same-state/same-region flags, haversine distance,
  point-in-time route transit mean (all-time, 90 d), destination-state fallback, the carrier
  estimate horizon itself and the seller's shipping-limit horizon (both known at purchase)
* **calendar** – day of week, hour, month, ISO week, weekend/holiday flags, days to next holiday,
  days to Black Friday, peak-season flag
* **load** – network-wide orders in the last 1 and 7 days

## Model

One LightGBM booster per (segment, stage, quantile) with `objective="quantile"`. Segments are
`intra_state`, `inter_state`, `cross_region`; a segment with fewer than `min_segment_rows` training
orders falls back to the global model for that stage. Quantile crossing is repaired by sorting the
predicted quantiles per row. XGBoost (`reg:quantileerror`) is a drop-in alternative via
`model.backend: xgboost`.

## Evaluation

* **Rolling-origin backtest** – cutoffs every 30 days; train on orders delivered before the cutoff
  *and purchased at least `label_gap_days` (21) earlier*, test on orders purchased in the following
  30 days. The gap matters: without it only ~70% of the orders purchased in the last month before a
  cutoff have a label, and they are the fast ones, which biased the end-to-end forecast low by about
  a day and dragged P80 coverage to 68% in an early run. Reports MAE and bias of the median, mean pinball
  loss, and coverage of every quantile, for the model, a route-level empirical-quantile baseline and
  the carrier estimate.
* **Promise policies** – every policy is scored on the same out-of-sample orders: on-time rate, mean
  promised lead time, promise MAE, slack, mean days late when late. A calibrated variant tunes a
  constant buffer on the earlier folds and is scored on all folds. The *frontier* plot sweeps a
  constant buffer for the carrier estimate and the model, giving on-time rate vs promised lead time.
* **A/B simulation** – random 50/50 split of the out-of-sample orders; treatment buffer tuned so its
  mean promised lead time equals control's (lead-time matched), then late rate, two-proportion
  z-test, bootstrap CI of the relative lift and the per-arm sample size an online test would need.

## Monitoring

PSI per feature between a reference window and the current window (`warn` 0.10, `alert` 0.25) and
weekly MAE / upper-quantile coverage of the out-of-sample predictions with a coverage tolerance.
