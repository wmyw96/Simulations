"""Shared type aliases used across the simulation framework."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ConfigDict = dict[str, Any]
Metadata = dict[str, Any]
ReadonlyConfig = Mapping[str, Any]
