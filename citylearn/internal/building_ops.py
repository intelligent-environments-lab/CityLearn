from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from citylearn.building import Building


class BuildingOpsService:
    """Internal observation/action operations for `Building`."""

    def __init__(self, building: "Building"):
        self.building = building
