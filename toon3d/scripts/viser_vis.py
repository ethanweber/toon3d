from pathlib import Path

import numpy as np
import torch
import os
import re
import mediapy

import tyro
import viser

from toon3d.utils.viser_utils import view_pcs_cameras
from toon3d.utils.camera_utils import Cameras
from toon3d.scripts.run import load_dataset

def main(
        data_prefix: Path = Path("data/processed"),
        dataset: str = "spritied-away-train",
        output_prefix: Path = Path("outputs"),
        output_method: Path = Path("run"),
        server:viser.ViserServer = None,
        visible:bool=True,
        return_early:bool=False
    ):
    device = "cpu"

    # load general data
    data_path = data_prefix / dataset
    images, heights, widths, points, points_normed, points_mask, depths, masks = load_dataset(data_path, device)
    n, m = points.shape[:2]

    # get latest output folder
    output_dir = output_prefix / dataset / output_method
    output_folder = Path(output_dir / sorted(os.listdir(output_dir))[-1]) # takes latest in folder

    # load cameras
    camera_params_dir = output_folder / "camera_params"
    cameras_dir = Path(camera_params_dir / "cameras")

    coarse_camera_params = torch.load(cameras_dir / "coarse_cameras.pt")
    refined_camera_params = torch.load(cameras_dir / "refined_cameras.pt")
    
    coarse_cameras = Cameras(n).to(device)
    coarse_cameras.load_state_dict(coarse_camera_params)

    refined_cameras = Cameras(n).to(device)
    refined_cameras.load_state_dict(refined_camera_params)

    # load warped images
    warp_visuals_dir = output_folder / "visuals_warp"
    warped_images_dir = Path(warp_visuals_dir / "warped_images")
    regex = re.compile(r'^\d+\.png$')
    warped_image_filenames = sorted([f for f in warped_images_dir.glob("*.png") if regex.match(f.name)])
    warped_images = [torch.from_numpy(mediapy.read_image(imf)[:, :, :3] / 255.0).float() for imf in warped_image_filenames]

    # load points and simplices
    simplices_list = torch.load(output_folder / "nerfstudio" / "meshes/simplices.pt")

    visuals_3d_dir = output_folder / "visuals_3d"
    output_verts_dir = visuals_3d_dir / "verts"
    output_corrs_dir = visuals_3d_dir / "corrs"
    output_dense_points_dir = visuals_3d_dir / "dense_points"
    output_img_colors_dir = visuals_3d_dir / "img_colors"

    mesh_verts_list = torch.load(output_verts_dir / "verts.pt")
    warped_mesh_verts_list = torch.load(output_verts_dir / "warped_verts.pt")

    coarse_corrs_3d = torch.load(output_corrs_dir / "corrs.pt")
    refined_corrs_3d = torch.load(output_corrs_dir / "warped_corrs.pt")
    
    coarse_dense_points_3d = torch.load(output_dense_points_dir / "dense_points.pt")
    warped_dense_points_3d = torch.load(output_dense_points_dir / "warped_dense_points.pt")

    images_colors = torch.load(output_img_colors_dir / "img_colors.pt")
    warped_images_colors = torch.load(output_img_colors_dir / "warped_img_colors.pt")

    # define colors
    point_colors = torch.rand((m, 3)).to(device)
    sparse_point_colors = [point_colors[points_mask[ndx]] for ndx in range(n)]
    mesh_colors = [np.random.rand(3) for _ in range(n)]

    if server is None:
        viser_server = viser.ViserServer()
    else:
        viser_server = server

    viser_server.add_frame(f"{str(output_method)}", show_axes=False, visible=visible)
    viser_server.add_frame(f"{str(output_method)}/coarse", show_axes=False, visible=False)
    view_pcs_cameras(viser_server, 
                         images, # cameras
                         coarse_cameras,
                         coarse_dense_points_3d, # dense pc
                         images_colors, 
                         coarse_corrs_3d, # sparse pc
                         sparse_point_colors,
                         mesh_verts_list, # mesh
                         simplices_list,
                         mesh_colors,
                         prefix=f"{str(output_method)}/coarse")

    view_pcs_cameras(viser_server, 
                        warped_images, # cameras
                        refined_cameras,
                        warped_dense_points_3d, # dense pc
                        warped_images_colors, 
                        refined_corrs_3d, # sparse pc
                        sparse_point_colors,
                        warped_mesh_verts_list, # mesh
                        simplices_list,
                        mesh_colors,
                        prefix=f"{str(output_method)}/refined")
    
    if return_early:
        return

    if server is None:
        while True:
            pass
    

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()