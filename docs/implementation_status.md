# Implementation Status

## Purpose

This document tracks what is already implemented in the repository versus what is still planned. It is meant to be a quick operational reference, not a full design document.

## Current Project Decisions

- The package lives under `src/simlab`.
- Runtime records use dataclasses.
- DGP parameters and estimator hyper-parameters use plain dictionaries.
- The framework should be built on top of NumPy.
- The framework should not depend on pandas.

## Implemented Now

### Package scaffold

The following package structure exists:

```text
src/simlab/
  __init__.py
  core/
  dgp/
  estimators/
  evaluation/
  utils/
```

This means the project now has stable locations for core records, DGPs, estimators, experiment runners, and utilities.

### Exported top-level API

The package currently exports:

- `DataGeneratingProcess`
- `Estimator`
- `Evaluator`
- `SampledData`
- `EstimateResult`
- `TrialRecord`

These are re-exported from `src/simlab/__init__.py`.

### Core dataclasses

The following runtime records are implemented:

#### `SampledData`

Purpose:

- store one sampled dataset from a DGP in a model-agnostic way,
- currently includes an `observed` dictionary and an optional `oracle` dictionary.

#### `EstimateResult`

Purpose:

- store the result of estimator fitting,
- currently includes the target name, point estimate, optional inference fields, and diagnostics.

#### `TrialRecord`

Purpose:

- store one evaluator-level record for one estimator on one trial,
- includes seeds, truth, the estimate result, runtime, configs, and diagnostics.

### Abstract base classes

The following abstract interfaces are implemented:

#### `DataGeneratingProcess`

Implemented methods:

- `__init__(name, params)`
- `sample(n, seed=None)` as an abstract method
- `true_parameter()` as an abstract method
- `get_params()` returning a defensive copy

Current role:

- define the contract for every future DGP.

#### `Estimator`

Implemented methods:

- `__init__(name, hyper_parameters)`
- `fit(data)` as an abstract method
- `predict(data)` as an abstract method
- `get_hyper_parameters()` returning a defensive copy
- `summary()` returning the fitted estimate object

Current role:

- define the common interface for baselines, DML, and the proposed method.

#### `Evaluator`

Implemented methods:

- `__init__(...)`
- `run()` as an abstract method
- `save(path)` as an abstract method

Current role:

- define the contract for experiment runners.

### Utility placeholders

The following modules exist as placeholders for upcoming work:

- `evaluation/experiment.py`
- `evaluation/metrics.py`
- `evaluation/results.py`
- `utils/io.py`
- `utils/grids.py`
- `utils/validation.py`

These files currently establish package structure, but do not yet contain substantive implementations.

### Implemented concrete DGP

The following concrete DGP is now implemented:

#### `PartialLinearModelUniformNoiseDGP`

Location:

- `src/simlab/dgp/partial_linear.py`

Current behavior:

- samples covariates `x` from `Unif[-1, 1]^d`,
- applies callable `func_pi(x)` to build the treatment regression,
- applies uniform treatment noise with scale `sigma_u`,
- applies callable `func_mu(x)` to build the outcome regression,
- applies uniform outcome noise with scale `sigma_eps`,
- returns `SampledData` with observed arrays `x`, `t`, `y` and optional oracle arrays `pi_x`, `mu_x`.

### Implemented concrete estimators

The following PLM estimators are now implemented:

#### `PLMDMLEstimator`

Location:

- `src/simlab/estimators/plm_est.py`

Current behavior:

- builds fully connected ReLU networks for `mu` and `pi`,
- uses batch normalization and residual connections,
- trains `beta`, `mu`, and `pi` on `D2` with one Adam optimizer,
- computes the final estimator on `D1` using Eq. (1.2) from the PLM paper.

#### `PLMOracleAIPWEstimator`

Location:

- `src/simlab/estimators/plm_est.py`

Current behavior:

- deep-copies the ground-truth nuisance functions,
- evaluates them directly on `D1`,
- computes the oracle AIPW estimator using the same Eq. (1.2).

### Randomness helpers

The following helpers are implemented in `core/randomness.py`:

- `derive_seed(base_seed, *parts)`
- `python_rng(seed)`

Current role:

- provide reproducible seed handling for future evaluators and estimators.

### Type aliases

The following aliases are implemented in `core/types.py`:

- `ConfigDict`
- `Metadata`
- `ReadonlyConfig`

Current role:

- provide a shared vocabulary for dictionary-based configuration and metadata.

## Not Implemented Yet

The following pieces are still planned but not yet implemented:

- a simple baseline estimator
- a concrete evaluator or experiment runner
- metric computation functions
- result aggregation and persistence utilities
- unit tests and smoke tests
- runnable study scripts
- additional DGP variants beyond `PartialLinearModelUniformNoiseDGP`

## Tests Added So Far

The following unit test now exists:

- `tests/unit/test_partial_linear_dgp.py`
- `tests/unit/test_plm_estimators.py`

It is designed to check:

- output shapes,
- oracle versus non-oracle sampling behavior,
- seed reproducibility,
- exact zero-noise identities,
- recovery of the true parameter through `true_parameter()`.

The estimator unit test is designed to check:

- exact oracle AIPW computation against a manual formula,
- that the neural DML estimator can fit and predict on PLM data,
- that the odd-sample split rule places the extra observation in `D2`.

## Recommended Next Build Order

The next steps I recommend are:

1. Implement one very simple baseline estimator for the current partial linear DGP.
2. Implement a concrete experiment runner that can execute one small study.
3. Add a first smoke test for the DGP-estimator-evaluator path.
4. Add DML and then the paper-specific estimator.
5. Add additional DGP variants once the baseline experiment path is stable.

## Validation Completed So Far

The scaffold and first concrete DGP have been checked with:

- `python3 -m compileall src`
- `PYTHONPATH=src python3 -c "import simlab; print(simlab.__all__)"`
- `PYTHONPATH=src /Users/yihongg/Code/Simulation/.conda/simlab/bin/python -m unittest tests.unit.test_partial_linear_dgp`
- `PYTHONPATH=src /Users/yihongg/Code/Simulation/.conda/simlab/bin/python -m unittest tests.unit.test_plm_estimators`

Environment note:

- a project-local conda environment now exists at `/Users/yihongg/Code/Simulation/.conda/simlab`,
- NumPy is installed there,
- PyTorch is installed there,
- the current unit tests pass in that environment.

This confirms the current package structure imports correctly and that the first implemented DGP behaves as expected under its unit tests.
