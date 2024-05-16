from pathlib import Path
import json
import sys
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import mediapy
from skimage import transform
from dataclasses import dataclass
from typing_extensions import Annotated

import tyro
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.colormaps import apply_depth_colormap
from toon3d.utils.image_processing_utils import generate_depths, generate_segments, generate_lines
from nerfstudio.configs.base_config import PrintableConfig

@dataclass
class ProcessData(PrintableConfig):
    """Process data
    
    Sets the default configuration for processing data.
    """

    input_path: Path = Path("data/images/rick-house")
    dataset: str = "rick-house"
    data_prefix: Path = Path("data/processed")
    resize: bool = True
    max_height: int = 720
    max_width: int = 960
    depth_method: str = "marigold"
    compute_segment_anything: bool = True
    compute_lines: bool = False
    update_segment_anything: bool = False
    update_lines: bool = False
    skip_image_resizing: bool = False
    add_to_existing: bool = False
    sam_checkpoint_prefix: str = "data/sam-checkpoints"
    sam_points_per_side: int = 32
    """Set to higher, e.g., 128 for higher quality but slower."""


@dataclass
class InitializeProcessData(ProcessData):
    """Initialize process data
    
    This will take a folder of images and process them.
    """

    def run(self):
        """check here"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.data_prefix.mkdir(exist_ok=True, parents=True)

        dataset_path = self.data_prefix / self.dataset
        images_path = dataset_path / "images"
        depths_path = dataset_path / "depths"
        depth_images_path = dataset_path / "depth-images"

        if self.add_to_existing:
            assert images_path.exists(), f"Dataset images path {images_path} does not exist. Cannot use `add-to-existing` flag."

        dataset_path.mkdir(exist_ok=True)
        images_path.mkdir(exist_ok=True)
        depths_path.mkdir(exist_ok=True)
        depth_images_path.mkdir(exist_ok=True)
        
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        image_filenames = []
        for extension in image_extensions:
            image_filenames.extend(self.input_path.glob(extension))
        image_filenames = sorted(image_filenames)

        if len(image_filenames) == 0:
            CONSOLE.log("[bold red]No images found in input path.")
            sys.exit(1)
        images = [mediapy.read_image(imf)[..., :3] for imf in image_filenames]

        if not self.skip_image_resizing:
            # resize and calculate padding
            for i, image in enumerate(images):
                if self.resize:
                    height, width = image.shape[:2]
                    height_scale_factor = 1
                    width_scale_factor = 1
                    if self.max_height and height > self.max_height:
                        height_scale_factor = self.max_height / height
                    if self.max_width and width > self.max_width:
                        width_scale_factor = self.max_width / width
                    scale_factor = min(height_scale_factor, width_scale_factor)
                    image = transform.resize(image, (scale_factor * height, scale_factor * width), anti_aliasing=True)
                    images[i] = (image * 255).astype(np.uint8)

        CONSOLE.log("[bold yellow]Running Depth Estimation...")
        depths = generate_depths(images, method=self.depth_method, device=device)
        CONSOLE.log("[bold green]:tada: Done Finding Depths.")

        torch.cuda.empty_cache()

        si = len(list(images_path.glob("*.png"))) if self.add_to_existing else 0

        for i, (image, depth) in enumerate(zip(images, depths)):
            # image
            mediapy.write_image(images_path / f"{si+i:05d}.png", image)
            # depth
            torch.save(depth, depths_path / f"{si+i:05d}.pt")
            # depth image
            mediapy.write_image(depth_images_path / f"{si+i:05d}.png", apply_depth_colormap(depth))

        # check if metadata exists
        metadata_path = dataset_path / "metadata.json"
        compute_metadata = not metadata_path.exists()

        if compute_metadata:
            metadata = {}
            frames = []
            for i in range(len(images)):
                frame = {
                    "file_path": f"images/{i:05d}.png",
                    "depth_file_path": f"depths/{i:05d}.pt",
                    "depth_image_file_path": f"depth-images/{i:05d}.png",
                }
                frames.append(frame)
            metadata["frames"] = frames
        elif self.add_to_existing:
            metadata = json.loads(metadata_path.read_text())
            for i in range(len(images)):
                frame = {
                    "file_path": f"images/{si+i:05d}.png",
                    "depth_file_path": f"depths/{si+i:05d}.pt",
                    "depth_image_file_path": f"depth-images/{si+i:05d}.png",
                }
                metadata["frames"].append(frame)
        else:
            metadata = json.loads(metadata_path.read_text())

        do_segment_anything = self.compute_segment_anything and (self.add_to_existing or self.update_segment_anything or compute_metadata or "masks" not in metadata["frames"][0])
        do_lines = self.compute_lines and (self.add_to_existing or self.update_lines or compute_metadata or "lines" not in metadata["frames"][0])
        
        if do_segment_anything:
            CONSOLE.log("[bold yellow]Running Segment Anything... (This may take a while)")
            segments = generate_segments(images, device, sam_checkpoint_prefix=self.sam_checkpoint_prefix, sam_points_per_side=self.sam_points_per_side)
            CONSOLE.log("[bold green]:tada: Done Segmenting Masks.")
            for i, segment in enumerate(segments):
                metadata["frames"][si+i]["masks"] = segment
        elif self.compute_segment_anything:
            CONSOLE.log("[bold yellow]Segment Anything Masks already exist. Use --update-segment-anything to recompute.")
        
        if do_lines:
            CONSOLE.log("[bold yellow]Running Line Detection...")
            lines = generate_lines(images, device)
            CONSOLE.log("[bold green]:tada: Done Finding Lines.")
            for i, line in enumerate(lines):
                metadata["frames"][si+i]["lines"] = line
        elif self.compute_lines:
            CONSOLE.log("[bold yellow]Line Detection already exist. Use --update-lines to recompute.")

        with open(dataset_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if self.add_to_existing:
            # need to update points.json if it exists
            points_path = dataset_path / "points.json"
            if points_path.exists():
                points = json.loads(points_path.read_text())
                for i in range(len(images)):
                    points["points"].append([])
                    points["validPoints"].append([])
                    points["validImages"].append(True)
                for i, segment in enumerate(segments):
                    polygons_as_bool = [[False] * len(s["polygons"]) for s in segment]
                    points["polygons"].append(polygons_as_bool)
                with open(points_path, "w", encoding="utf-8") as f:
                    json.dump(points, f, indent=4)

@dataclass
class AddProcessData(InitializeProcessData):
    """Add process data
    
    This will take a folder of images and add them to an existing dataset which has already been processed.
    It also might already have labels.
    """

    add_to_existing: bool = True

@dataclass
class RemoveProcessData(ProcessData):
    """Remove process data
    
    This will remove the images that are ignored according to the labels.
    """

    def run(self):
        raise NotImplementedError("RemoveProcessData is not implemented yet.")
        # We have to be careful here. The images need to keep the numbering 0, 1, ..., n-1 when saved.

def main(
    process_data: ProcessData,
):
    """Script to process data.

    Args:
        process_data: The process data configuration.
    """
    process_data.run()

Commands = Union[
    Annotated[InitializeProcessData, tyro.conf.subcommand(name="initialize")],
    Annotated[AddProcessData, tyro.conf.subcommand(name="add")],
    Annotated[RemoveProcessData, tyro.conf.subcommand(name="remove")],
]

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()
