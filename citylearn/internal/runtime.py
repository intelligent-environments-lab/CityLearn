from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citylearn.citylearn import CityLearnEnv


class CityLearnRuntimeService:
    """Internal runtime orchestration for `CityLearnEnv`."""

    def __init__(self, env: "CityLearnEnv"):
        self.env = env
