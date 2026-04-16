"""Abstract interface for simulation evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from simlab.core.types import ConfigDict


class Evaluator(ABC):
    """Base class for experiment runners comparing estimators on DGPs."""

    def __init__(
        self,
        name: str,
        dgp_name: str,
        dgp_param_grid: dict[str, list],
        estimators: list,
        n_trials: int,
        seed: int | None = None,
        evaluation_config: ConfigDict | None = None,
    ):
        self.name = name
        self.dgp_name = dgp_name
        self.dgp_param_grid = deepcopy(dgp_param_grid)
        self.estimators = list(estimators)
        self.n_trials = n_trials
        self.seed = seed
        self.evaluation_config = deepcopy(evaluation_config or {})
        self.results = None

    @abstractmethod
    def run(self):
        """Execute the experiment and populate results."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist evaluator results to disk."""
