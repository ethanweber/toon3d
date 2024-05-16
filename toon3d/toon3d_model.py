"""
Toon3D model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type
import torch
from torch.nn import Parameter
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

import torch
from torch.nn import Parameter

from toon3d.utils.losses import depth_ranking_loss

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components.losses import tv_loss


from toon3d.utils.novel_view_samplers import sample_interpolated_camera


@dataclass
class Toon3DModelConfig(SplatfactoModelConfig):
    """Config for the toon3d model."""

    _target: Type = field(default_factory=lambda: Toon3DModel)

    # depth regularization parameters
    use_tv_loss: bool = True
    """Whether to use the total variation loss."""
    tv_loss_mult: float = 100.0
    """Multiplier for the total variation loss."""
    tv_loss_strides: List[int] = field(default_factory=lambda: [1])
    """Downscale strides to use for the total variation loss."""
    use_tv_in_novel_views: bool = True
    """Whether to use the total variation loss in novel views."""
    tv_in_novel_views_loss_mult: float = 100.0
    """Multiplier for the total variation loss in novel views."""
    tv_in_novel_views_loss_strides: List[int] = field(default_factory=lambda: [1])
    """Downscale strides to use for the total variation loss in novel views."""
    use_depth_ranking_loss: bool = True
    """Whether to use the depth ranking loss."""
    depth_ranking_loss_mult: float = 1.0
    """Multiplier for the depth ranking loss."""
    depth_ranking_loss_patch_size: int = 128
    """Patch size for the depth ranking loss."""
    depth_ranking_loss_num_patches: int = 8
    """Number of patches to sample for the depth ranking loss."""
    use_depth_loss: bool = False
    """Whether to use the depth loss."""
    depth_loss_mult: float = 1.0
    """Multiplier for the depth loss."""

    steps_per_perturb: int = 10
    """Number of steps per perturb for novel view."""

    use_isotropic_loss: bool = False
    """Whether to use the isotropic loss."""
    isotropic_ratio_mult: float = 1e6
    """Multiplier for the isotropic ratio loss."""

    use_ratio_loss: bool = True
    """Whether to use the ratio loss."""
    ratio_loss_mult: float = 1e6
    """Multiplier for the ratio loss."""
    max_ratio_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """

class Toon3DModel(SplatfactoModel):
    """Model for toon3d."""

    config: Toon3DModelConfig

    def __init__(self, *args, **kwargs):
        self.cameras = kwargs["cameras"]
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        """Required for our custom models."""

        super().populate_modules()

        # we toggle this on and off
        # when True, we perturb the camera
        self.perturb_camera = False

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)

        if self.config.use_tv_in_novel_views:

            def toggle_perturb(step, **kwargs):
                if step != 0 and step % self.config.steps_per_perturb == 0:
                    self.perturb_camera = not self.perturb_camera

            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION, TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    toggle_perturb,
                )
            )
        return cbs

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = super().get_param_groups()

        # custom params
        # TODO:
        return gps

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        if self.perturb_camera:
            camera_perturb = sample_interpolated_camera(self.cameras)
            camera_perturb = camera_perturb.reshape(camera.shape).to(camera.device)
            outputs = super().get_outputs(camera_perturb)
            # mediapy.write_image("novel_view.png", outputs["rgb"].detach().cpu())
        else:
            outputs = super().get_outputs(camera)
        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.get_gt_img(batch["image"])
        metrics_dict = {}

        if self.perturb_camera:
            return metrics_dict

        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        # import pdb; pdb.set_trace();
        assert "mask" in batch, "mask must be in batch"

        if self.perturb_camera:
            if self.config.use_tv_in_novel_views:
                pred_depth = outputs["depth"]
                for stride in self.config.tv_in_novel_views_loss_strides:
                    l = self.config.tv_in_novel_views_loss_mult * tv_loss(pred_depth.permute(2, 0, 1)[None, :, ::stride, ::stride])
                    if l > 0:
                        loss_dict[f"depth_tv_in_novel_views_loss_stride-{stride}"] = l
            # hack to avoid having no gradients for backprop
            loss_dict["hack"] = (self.xys * 0).sum()
            return loss_dict

        image = batch["image"] # (height, width, 3)
        mask = batch["mask"].to(image.device)[..., 0] # (height, width)
        depth = batch["depth_image"] # (height, width, 1)

        assert self._get_downscale_factor() == 1, "downscale factor must be 1 for toon3d model"

        gt_img = image

        # rgb loss
        rgb = outputs["rgb"] * mask[...,None].to(outputs["rgb"].device)

        Ll1 = torch.abs(gt_img[mask] - rgb[mask]).mean()
        loss_dict["rgb"] = Ll1

        if self.config.use_tv_loss:
            pred_depth = outputs["depth"]
            for stride in self.config.tv_loss_strides:
                loss_dict[f"depth_tv_loss_stride-{stride}"] = self.config.tv_loss_mult * tv_loss(
                    pred_depth.permute(2, 0, 1)[None, :, ::stride, ::stride]
                )

        if self.config.use_depth_ranking_loss:
            pred_depth = outputs["depth"]
            gt_depth = depth.to(pred_depth.device)
            mask = mask.to(pred_depth.device)
            loss_dict["depth_ranking_loss"] = (
                self.config.depth_ranking_loss_mult
                * depth_ranking_loss(
                    rendered_depth=pred_depth.permute(2, 0, 1)[None, 0],
                    gt_depth=gt_depth.permute(2, 0, 1)[None, 0],
                    mask=mask[...,None].permute(2, 0, 1)[None, 0],
                    patch_size=self.config.depth_ranking_loss_patch_size,
                    num_patches=self.config.depth_ranking_loss_num_patches,
                ).mean()
            )

        if self.config.use_depth_loss:
            pred_depth = outputs["depth"]
            gt_depth = depth.to(pred_depth.device)
            loss_dict["depth_loss"] = self.config.depth_loss_mult * torch.abs(pred_depth[mask] - gt_depth[mask]).mean()

        if self.config.use_isotropic_loss:
            scale_var = torch.mean(torch.var(self.scales, dim=1))
            loss_dict["isotropic_loss"] = self.config.isotropic_ratio_mult * scale_var

        if self.config.use_ratio_loss:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_ratio_ratio),
                )
                - self.config.max_ratio_ratio
            )
            scale_reg = self.config.ratio_loss_mult * scale_reg.mean()

        return loss_dict
