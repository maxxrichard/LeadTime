"""Typed configuration loaded from YAML (see configs/default.yaml)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    n_synthetic_orders: int = 40_000
    seed: int = 42
    max_total_days: float = 90.0  # drop pathological orders (Olist has a handful > 150 days)


@dataclass
class ModelConfig:
    backend: str = "lightgbm"  # or "xgboost"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.8, 0.9])
    min_segment_rows: int = 2_000  # fall back to the global model below this
    log_target: bool = True
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
    )


@dataclass
class BacktestConfig:
    n_folds: int = 6
    horizon_days: int = 30
    step_days: int = 30
    min_train_days: int = 180
    # Orders purchased within this many days before the cutoff are excluded from training: their labels
    # are right-censored (only the fast ones are delivered by the cutoff), which biases the model low.
    label_gap_days: int = 21


@dataclass
class PromiseConfig:
    quantile: float = 0.8
    buffer_days: float = 0.0
    round_up: bool = True


@dataclass
class ABConfig:
    seed: int = 7
    n_bootstrap: int = 1_000
    alpha: float = 0.05
    power: float = 0.8
    match_lead_time: bool = True
    match_mode: str = "quantile"  # quantile | buffer


@dataclass
class MonitoringConfig:
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    coverage_tolerance: float = 0.05


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    promise: PromiseConfig = field(default_factory=PromiseConfig)
    ab: ABConfig = field(default_factory=ABConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        cfg = cls()
        for f in fields(cls):
            if f.name in d and d[f.name]:
                section = getattr(cfg, f.name)
                for k, v in d[f.name].items():
                    if not hasattr(section, k):
                        raise KeyError(f"Unknown config key {f.name}.{k}")
                    setattr(section, k, v)
        return cfg

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            return cls()
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
