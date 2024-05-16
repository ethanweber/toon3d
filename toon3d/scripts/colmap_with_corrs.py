"""
This script will run COLMAP with provided correspondences.
"""

import os
import numpy as np
from pathlib import Path
import torch
import tyro
from tqdm import tqdm
from toon3d.utils.colmap_database import COLMAPDatabase
import shutil
import mediapy
from toon3d.scripts.run import load_dataset
from datetime import datetime
from nerfstudio.utils.rich_utils import CONSOLE


def main(
        data_prefix: Path = Path("data/processed"),
        dataset: str = "rick-house",
        output_prefix: Path = Path("outputs"),
        device: str = "cuda"):
    """Creates a COLMAP database file from a folder of images and a folder of correspondences."""

    images, heights, widths, points, points_normed, points_mask, depths, masks = load_dataset(data_prefix, dataset)
    n, max_num_points = points.shape[:2]

    output_folder = output_prefix / dataset / "colmap_with_corrs" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # create the output folder if needed
    output_folder.mkdir(parents=True, exist_ok=True)
    # also create the sparse subdirectory to write COLMAP info to
    Path(output_folder / "colmap/sparse").mkdir(parents=True, exist_ok=True)

    # copy the images to the output folder
    images_folder = data_prefix / dataset / "images"
    shutil.copytree(images_folder, output_folder / "images")

    # create a database file
    database_path = Path(output_folder / "database.db")

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    image_ids = []
    for i in tqdm(range(n)):

        height, width = float(heights[i]), float(widths[i])

        # add cameras
        model = 1
        FOCAL_PRIOR = 1.2
        focalx = FOCAL_PRIOR * width
        focaly = FOCAL_PRIOR * width
        param_arr = np.array([focalx, focaly, width / 2, height / 2])
        camera_id = db.add_camera(model, width, height, param_arr)

        # add images
        # TODO: get the filename from the dataset
        image_filename = f"{i:05d}.png"
        image_id = db.add_image(image_filename, camera_id)
        image_ids.append(image_id)

        # add keypoints
        kpts = points[i].long().numpy()
        db.add_keypoints(image_id, kpts)  # (num_keypoints, X Y in pixel space)

    indices = np.arange(0, max_num_points)
    matches = np.concatenate([indices[:, None], indices[:, None]], axis=-1)

    matches_files = []

    for i in range(n):
        for j in range(i + 1, n):
            m = points_mask[i] & points_mask[j]
            mas =  matches[m.cpu()]
            matches_files.append(f"{i:05d}.png {j:05d}.png\n")
            for ma in mas:
                matches_files.append(f"{ma[0]} {ma[1]}\n")
            matches_files.append("\n")

    # save the matches to a file
    matches_file = output_folder / "matches.txt"
    with open(matches_file, "w") as f:
        f.writelines(matches_files)

    db.commit()
    db.close()

    cmd = f"colmap matches_importer --database_path {database_path} --match_list_path {matches_file} --match_type raw --SiftMatching.use_gpu 0 --TwoViewGeometry.min_num_inliers 0"
    CONSOLE.print("[bold yellow]Running COLMAP exhaustive matcher...")
    print(cmd)
    print("\n\n")
    os.system(cmd)

    # TODO: make these parameters configurable
    min_num_matches = 8
    min_num_inliers = 4
    init_num_trials = 1000

    cmd = f"colmap mapper --database_path {database_path}"
    cmd += f" --image_path {output_folder / 'images'} --output_path {output_folder / 'colmap/sparse'} --Mapper.tri_ignore_two_view_tracks 0 --Mapper.ba_local_function_tolerance=1e-6 --Mapper.ba_global_function_tolerance=1e-6"
    cmd += f" --Mapper.min_num_matches {min_num_matches}"
    cmd += f" --Mapper.init_num_trials {init_num_trials}"
    cmd += f" --Mapper.init_min_num_inliers {min_num_inliers}"
    cmd += f" --Mapper.abs_pose_min_num_inliers {min_num_inliers}"

    CONSOLE.print("[bold yellow]Running COLMAP mapper...")
    print(cmd)
    print("\n\n")
    os.system(cmd)

    input_path = output_folder / 'colmap/sparse/0'
    if not input_path.exists():
        CONSOLE.print("[bold red]COLMAP failed to create a model.")
        return

    cmd = f"colmap model_converter --input_path {input_path} --output_path {output_folder / 'colmap/sparse/0'} --output_type TXT"
    CONSOLE.print("[bold yellow]Converting COLMAP model to TXT...")
    print(cmd)
    print("\n\n")
    os.system(cmd)

    nerfstudio_cmd = f"ns-train splatfacto --data {output_folder} --viewer.camera_frustum_scale 1.0 --viewer.default_composite_depth False colmap --eval-mode all"
    CONSOLE.print("[bold yellow]Here is the splatfacto command to run:")
    print(nerfstudio_cmd)


tyro.cli(main)