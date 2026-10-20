# Data

`data/raw/` holds the nine Olist CSVs (not committed). Two ways to fill it:

| | command | orders |
|---|---|---|
| **Real dataset** (Kaggle, needs API token) | `make olist` or `python scripts/download_olist.py` | ~100k |
| **Synthetic stand-in** (offline, reproducible) | `leadtime synth` | 40k by default |

Both produce identical file names and columns, so every later step (`prepare`, `features`,
`backtest`, `train`) is the same. `data/processed/` receives `orders.parquet` (one canonical row per
delivered order with eight timestamps and stage durations) and `features.parquet`.

The eight timestamps per order and the stages derived from them:

```
purchase_ts ─► approved_ts ─► carrier_ts ─► delivered_ts        estimated_ts (carrier estimate)
   └ approval ─┘└ handling ──┘└── transit ──┘                    shipping_limit_ts (seller SLA)
   └────────────────── total ───────────────┘                    review_creation_ts, review_answer_ts
```

The synthetic generator (`leadtime.data.synthetic`) encodes seller-specific handling speed, weekend
and holiday effects, load-dependent handling, distance/region-dependent transit with a heavy tail,
a Black-Friday/December peak and a conservative route-level carrier estimate. It is a stand-in for
running the code, not a statistical replica of Olist; always report numbers from the real data when
you have it.
