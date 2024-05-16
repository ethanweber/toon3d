"""
render.py
"""
from __future__ import annotations

import json
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from typing_extensions import Annotated

import viser.transforms as tf

from nerfstudio.cameras.camera_paths import (
    get_path_from_json,
)

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
from nerfstudio.scripts.render import (
    get_crop_from_json,
    insert_spherical_metadata_into_file,
    BaseRender,
    RenderCameraPath,
    CropData,
)


from typing import Any, Dict, Optional, Tuple

import torch

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_utils import get_interpolated_poses_many
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
import trimesh
from tqdm import tqdm
from toon3d.toon3d_datamanager import Toon3DDataManagerConfig, Toon3DDataManager
from toon3d.toon3d_dataparser import Toon3DDataParserConfig

def get_interpolated_camera_path(cameras: Cameras, steps: int, order_poses: bool) -> Cameras:
    """Generate a camera path between two cameras. Uses the camera type of the first camera

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices()
    poses = cameras.camera_to_worlds
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps, order_poses=order_poses)

    cameras = Cameras(
        fx=Ks[:, 0, 0],
        fy=Ks[:, 1, 1],
        cx=Ks[:, 0, 2],
        cy=Ks[:, 1, 2],
        camera_type=cameras.camera_type[0],
        camera_to_worlds=poses,
    )
    return cameras


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    render_nearest_camera=False,
    render_nearest_camera_location: Literal["corner", "right"] = "corner",
    return_writer_frames: bool = False,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
        render_nearest_camera: Whether to render the nearest training camera to the rendered camera.
        render_nearest_camera_location: Where to paste the nearest training camera in the rendered image.
        return_writer_frames: Whether to return the frames for the video.
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)
    
    writer_frames = []
    with ExitStack() as stack:
        writer = None

        if render_nearest_camera:
            assert pipeline.datamanager.train_dataset is not None
            train_dataset = pipeline.datamanager.train_dataset
            train_cameras = train_dataset.cameras.to(pipeline.device)
        else:
            train_dataset = None
            train_cameras = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                max_dist, max_idx = -1, -1
                true_max_dist, true_max_idx = -1, -1

                if render_nearest_camera:
                    assert pipeline.datamanager.train_dataset is not None
                    assert train_dataset is not None
                    assert train_cameras is not None
                    cam_pos = cameras[camera_idx].camera_to_worlds[:, 3].cpu()
                    cam_quat = tf.SO3.from_matrix(cameras[camera_idx].camera_to_worlds[:3, :3].numpy(force=True)).wxyz

                    for i in range(len(train_cameras)):
                        train_cam_pos = train_cameras[i].camera_to_worlds[:, 3].cpu()

                        q = tf.SO3.from_matrix(train_cameras[i].camera_to_worlds[:3, :3].numpy(force=True)).wxyz
                        # calculate distance between two quaternions
                        rot_dist = 1 - np.dot(q, cam_quat) ** 2
                        pos_dist = torch.norm(train_cam_pos - cam_pos)
                        dist = 0.3 * rot_dist + 0.7 * pos_dist

                        if true_max_dist == -1 or dist < true_max_dist:
                            true_max_dist = dist
                            true_max_idx = i

                    if max_idx == -1:
                        max_idx = true_max_idx

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    is_depth = rendered_output_name.find("depth") != -1
                    if is_depth:
                        output_image = (
                            colormaps.apply_depth_colormap(
                                output_image,
                                # accumulation=outputs["accumulation"],
                                near_plane=depth_near_plane,
                                far_plane=depth_far_plane,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                if render_nearest_camera:
                    assert train_dataset is not None
                    assert train_cameras is not None
                    img = train_dataset.get_image_float32(max_idx)

                    # Add closest training image to the bottom right of the rendered image
                    if render_nearest_camera_location == "corner":
                        height = cameras.image_height[0] // 4
                        # maintain the resolution of the img to calculate the width from the height
                        width = int(img.shape[1] * (height / img.shape[0]))
                        resized_image = torch.nn.functional.interpolate(
                            img.permute(2, 0, 1)[None], size=(int(height), int(width)), mode="bilinear"
                        )[0].permute(1, 2, 0)
                        resized_image = (
                            colormaps.apply_colormap(
                                image=resized_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                        for i in range(len(render_image)):
                            render_image[i][-height:, -width:, :] = resized_image
                    # Add closest training image to the right of the rendered image
                    elif render_nearest_camera_location == "right":
                        height = cameras.image_height[0]
                        # maintain the resolution of the img to calculate the width from the height
                        width = int(img.shape[1] * (height / img.shape[0]))
                        resized_image = torch.nn.functional.interpolate(
                            img.permute(2, 0, 1)[None], size=(int(height), int(width))
                        )[0].permute(1, 2, 0)
                        resized_image = (
                            colormaps.apply_colormap(
                                image=resized_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                        render_image.append(resized_image)
                    else:
                        raise NotImplementedError

                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    if image_format == "png":
                        media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
                    if image_format == "jpeg":
                        media.write_image(
                            output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                        )
                if output_format == "video":
                    writer_frames.append(render_image)
        if output_format == "video" and not return_writer_frames:
            media.write_video(output_filename, writer_frames, fps=fps)
        elif return_writer_frames:
            return writer_frames

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))

def resize_numpy(image, size):
    """Resize a numpy image."""
    image = torch.nn.functional.interpolate(
        torch.from_numpy(image).permute(2, 0, 1)[None], size=size, mode="bilinear"
    )[0].permute(1, 2, 0).numpy()
    return image

def get_frame_in_background(frame, max_height, max_width):

    # Create a new white background tensor
    background = torch.ones((max_height, max_width, 3))
    
    # zoom the frame in until it hits one of the edges
    frame_height, frame_width = frame.shape[:2]
    scale = min(max_height / frame_height, max_width / frame_width)
    frame = resize_numpy(frame, (int(frame_height * scale), int(frame_width * scale)))

    # Calculate the position to paste the frame onto the background
    top = (max_height - frame.shape[0]) // 2
    left = (max_width - frame.shape[1]) // 2

    # Paste the frame onto the background
    background[top:top+frame.shape[0], left:left+frame.shape[1]] = torch.from_numpy(frame)
    return background

@dataclass
class RenderToon3DCameraPath(RenderCameraPath):
    """Render toon3d camera path."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    render_nearest_camera: bool = True
    """Whether to render the nearest training camera to the rendered camera."""
    render_nearest_camera_location: Literal["corner", "right"] = "corner"
    """Where to paste the nearest training camera in the rendered image."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            render_nearest_camera_location=self.render_nearest_camera_location,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
                render_nearest_camera=self.render_nearest_camera,
                render_nearest_camera_location=self.render_nearest_camera_location,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")

@dataclass
class RenderToon3DInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 96
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    render_nearest_camera: bool = False
    """Whether to render the nearest training camera to the rendered camera."""
    render_nearest_camera_location: Literal["corner", "right"] = "corner"
    """Where to paste the nearest training camera in the rendered image."""
    inference_near_plane: float = 0.1
    """Ignore any gaussians at distances less than this near plane."""
    representation: Literal["splat"] = "splat"
    """Representation to render."""
    nerfstudio_data_path: Path = Path("nerfstudio_data_path")
    """Path to the nerfstudio data."""
    point_radius: float = 0.02
    """Radius of the points to render."""

    def main(self) -> None:
        """Main function."""

        if str(self.output_path).endswith(".mp4"):
            raise ValueError("Output path should be a directory for interpolated rendering.")
        self.output_path.mkdir(parents=True, exist_ok=True)

        install_checks.check_ffmpeg_installed()

        device = "cuda:0"

        if self.representation == "splat":
            _, pipeline, _, _ = eval_setup(
                self.load_config,
                eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                test_mode="test",
            )
            pipeline.model.config.inference_near_plane = self.inference_near_plane

            if self.pose_source == "eval":
                assert pipeline.datamanager.eval_dataset is not None
                cameras = pipeline.datamanager.eval_dataset.cameras
            else:
                assert pipeline.datamanager.train_dataset is not None
                cameras = pipeline.datamanager.train_dataset.cameras

            datamanager = pipeline.datamanager
        else:
            # create cameras from the transforms.json instead of from the pipeline
            datamanager_config=Toon3DDataManagerConfig(
                dataparser=Toon3DDataParserConfig(data=self.nerfstudio_data_path, eval_mode="all", load_3D_points=True, depth_unit_scale_factor=1.0)
            )
            datamanager = Toon3DDataManager(config=datamanager_config)
            cameras = datamanager.train_dataparser_outputs.cameras.to(device)

        all_pair_indices = []
        for i in range(len(cameras)-1):
            for j in range(i+1, len(cameras)):
                all_pair_indices.append([i, j])

        for indices in all_pair_indices:
            CONSOLE.print(f"Rendering trajectory from {indices[0]} to {indices[1]}")
            Ks = cameras.get_intrinsics_matrices()[indices]
            poses = cameras.camera_to_worlds[indices]
            cameras_pair = Cameras(
                fx=Ks[:, 0, 0][..., None],
                fy=Ks[:, 1, 1][..., None],
                cx=Ks[:, 0, 2][..., None],
                cy=Ks[:, 1, 2][..., None],
                camera_type=cameras.camera_type[0],
                camera_to_worlds=poses,
            )

            seconds = self.interpolation_steps * len(cameras_pair) / self.frame_rate
            camera_path = get_interpolated_camera_path(
                cameras=cameras_pair,
                steps=self.interpolation_steps,
                order_poses=self.order_poses,
            )

            if self.representation == "splat":
                writer_frames = _render_trajectory_video(
                    pipeline,
                    camera_path,
                    output_filename=self.output_path,
                    rendered_output_names=self.rendered_output_names,
                    rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                    seconds=seconds,
                    output_format=self.output_format,
                    image_format=self.image_format,
                    depth_near_plane=self.depth_near_plane,
                    depth_far_plane=self.depth_far_plane,
                    colormap_options=self.colormap_options,
                    render_nearest_camera=self.render_nearest_camera,
                    render_nearest_camera_location=self.render_nearest_camera_location,
                    return_writer_frames=True
                )
            else:
                raise ValueError(f"Unknown representation {self.representation}")
            max_height = max([frame.shape[0] for frame in writer_frames])
            max_width = max([frame.shape[1] for frame in writer_frames])

            frames = writer_frames

            new_frames = []
            # paste the frames onto a white background
            for frame in frames:
                new_frame = get_frame_in_background(frame, max_height, max_width)
                new_frames.append(new_frame.numpy())

            # add the fade in and fade out effect
            image_s = np.array(pipeline.datamanager.train_dataset.get_image_float32(indices[0]))
            image_e = np.array(pipeline.datamanager.train_dataset.get_image_float32(indices[1]))
            frame_s = frames[0]
            frame_e = frames[-1]
            image_s = resize_numpy(image_s, frame_s.shape[:2])
            image_e = resize_numpy(image_e, frame_e.shape[:2])

            image_s = get_frame_in_background(image_s, max_height, max_width).numpy()
            image_e = get_frame_in_background(image_e, max_height, max_width).numpy()
            frame_s = get_frame_in_background(frame_s, max_height, max_width).numpy()
            frame_e = get_frame_in_background(frame_e, max_height, max_width).numpy()

            transition_num_frames = 20
            extra_frames_i = np.linspace(0, 1, transition_num_frames)
            extra_frames_s = [image_s * (1 - i) + frame_s * i for i in extra_frames_i]
            extra_frames_e = [frame_e * (1 - i) + image_e * i for i in extra_frames_i]

            new_frames = extra_frames_s + new_frames + extra_frames_e

            output_filename = self.output_path / f"interpolated-{indices[0]}-{indices[1]}.mp4"
            media.write_video(output_filename, new_frames, fps=self.frame_rate)

@dataclass
class RenderToon3DDataset(RenderCameraPath):
    """Render toon3d dataset."""

    def main(self) -> None:
        """Main function."""
        raise NotImplementedError


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderToon3DCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderToon3DInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[RenderToon3DDataset, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa