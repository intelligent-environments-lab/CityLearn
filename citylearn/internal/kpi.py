from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnKPIService:
    """Internal KPI/evaluation service for `CityLearnEnv`."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env
