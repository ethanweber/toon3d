"""
Define the toon3d method config.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from toon3d.toon3d_model import Toon3DModelConfig
from toon3d.toon3d_pipeline import Toon3DPipelineConfig
from toon3d.toon3d_datamanager import Toon3DDataManagerConfig
from toon3d.toon3d_dataparser import Toon3DDataParserConfig

from nerfstudio.plugins.types import MethodSpecification

toon3d_config = MethodSpecification(
    TrainerConfig(
        method_name="toon3d",
        steps_per_eval_image=100000,
        steps_per_eval_batch=100000,
        steps_per_eval_all_images=100000,
        steps_per_save=500,
        max_num_iterations=2000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=Toon3DPipelineConfig(
            datamanager=Toon3DDataManagerConfig(
                dataparser=Toon3DDataParserConfig(eval_mode="all", load_3D_points=True, depth_unit_scale_factor=1.0)
            ),
            model=Toon3DModelConfig(
                # sh_degree=0,
                # warmup_length=0,
                # cull_scale_thresh=0.1,
                sh_degree_interval=1,
                output_depth_during_training=True,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, camera_frustum_scale=0.5, default_composite_depth=False),
        vis="viewer",
    ),
    description="Method for reconstructing toon3d.",
)
