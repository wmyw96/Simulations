"""Abstract interface for simulation estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from simlab.core.records import EstimateResult, SampledData
from simlab.core.types import ConfigDict


class Estimator(ABC):
    """Base class for all estimators evaluated in simulation studies."""

    def __init__(self, name: str, hyper_parameters: ConfigDict):
        self.name = name
        self.hyper_parameters = deepcopy(hyper_parameters)
        self.est_params: EstimateResult | None = None

    @abstractmethod
    def fit(self, data: SampledData) -> EstimateResult:
        """Fit the estimator on sampled data and return the estimate."""

    @abstractmethod
    def predict(self, data: SampledData):
        """Return model-specific predictions on new data."""

    def get_hyper_parameters(self) -> ConfigDict:
        """Return a copy of the estimator hyper-parameters."""
        return deepcopy(self.hyper_parameters)

    def summary(self) -> EstimateResult | None:
        """Return the current fitted estimate if available."""
        return self.est_params
