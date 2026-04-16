"""Abstract interface for data-generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from simlab.core.records import SampledData
from simlab.core.types import ConfigDict


class DataGeneratingProcess(ABC):
    """Base class for all simulation data-generating processes."""

    def __init__(self, name: str, params: ConfigDict):
        self.name = name
        self.params = deepcopy(params)

    @abstractmethod
    def sample(self, n: int, seed: int | None = None) -> SampledData:
        """Sample one dataset of size n."""

    @abstractmethod
    def true_parameter(self) -> float | dict[str, float] | None:
        """Return the estimand value implied by the current DGP."""

    def get_params(self) -> ConfigDict:
        """Return a copy of the DGP configuration."""
        return deepcopy(self.params)
