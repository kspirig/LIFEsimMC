from dataclasses import dataclass

from lifesimmc.core.resources.base_resource import BaseResource


@dataclass
class PlotResource(BaseResource):
    """Resource containing metadata for a saved plot."""

    path: str = None
