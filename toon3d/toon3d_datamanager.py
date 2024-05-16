"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.data.datasets.depth_dataset import DepthDataset


@dataclass
class Toon3DDataManagerConfig(FullImageDatamanagerConfig):
    """Toon3D DataManager Config

    Add your custom datamanager config parameters here.
    """
    _target: Type = field(default_factory=lambda: Toon3DDataManager)


class Toon3DDataManager(FullImageDatamanager[DepthDataset]):
    """Toon3D DataManager

    Args:
        config: the Toon3DDataManager used to instantiate class
    """

    config: Toon3DDataManagerConfig

    def __init__(self, config: Toon3DDataManagerConfig, **kwargs):
        super().__init__(config)

    def setup_train(self):
        """Setup the train dataloader."""
        self.device = "cuda"

