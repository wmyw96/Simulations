"""Structured records shared across DGPs, estimators, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    np = Any


@dataclass
class SampledData:
    """A single sampled dataset produced by a data-generating process."""

    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    pi_x: np.ndarray | None = None
    mu_x: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        """Compatibility alias for x."""
        return self.x

    @property
    def T(self) -> np.ndarray:
        """Compatibility alias for t."""
        return self.t

    @property
    def Y(self) -> np.ndarray:
        """Compatibility alias for y."""
        return self.y


@dataclass
class EstimateResult:
    """Structured output returned by an estimator fit."""

    target: str
    estimate: float
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRecord:
    """One evaluator row for a single estimator on a single trial."""

    study_name: str
    dgp_name: str
    estimator_name: str
    trial_id: int
    data_seed: int
    estimator_seed: int | None
    theta_true: float | None
    estimate_result: EstimateResult
    runtime_sec: float
    fit_status: str = "success"
    fit_message: str = ""
    dgp_config: dict[str, Any] = field(default_factory=dict)
    estimator_config: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
