"""
Main script.
"""

from pathlib import Path
import json
from datetime import datetime


import cv2 
import mediapy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import shutil
import cv2

import tyro
import viser
from toon3d.utils.draw_utils import get_images_with_keypoints
from toon3d.utils.viser_utils import view_pcs_cameras
from toon3d.utils.camera_utils import Cameras
from toon3d.utils.io_utils import make_transforms_json, make_dense_points_3d, make_plys, load_dataset
from toon3d.warp.warp_mesh import WarpMesh, draw_tris, draw_points, FrameMeshes, get_simplices
from toon3d.warp.arap_utils import face_verts_arap_loss, calc_area_loss, signed_areas


from nerfstudio.utils.rich_utils import CONSOLE


def main(
    data_prefix: Path = Path("data/processed"),
    dataset: str = "bobs-burgers-dining",
    device: str = "cpu",
    niters: int = 2000,
    lr: float = 0.01,
    affine_loss_mult: float = 1000.0,
    scale_loss_mult: float = 1e3,
    scale_loss_target: float = 1.0,
    offset_loss_mult: float = 1e3,
    focal_mag_mult: float = 1e-3,
    aspect_mult: float = 1,
    up_mult: float = 0,
    niters_warp: int = 1000,
    lr_warp: float = 0.01,
    arap_mult: float = 1, 
    zs_diff_mult: float = 1,
    output_prefix: Path = Path("outputs"),
    output_method: Path = Path("run"),
    nerfstudio_folder: Path = Path("data/nerfstudio"),
    ply_downscale_factor: int = 4,
    view_point_cloud: bool = True,
    save_geometry: bool = True,
    write_to_nerfstudio: bool = True,
    port: int = 7007,
    seed: int = None,
):
    """_summary_
    Script to run our SfM on cartoons.

    Args:
        data_prefix (Path, optional): path to dataset directory. Defaults to Path("data/processed").
        dataset (str, optional): datset within dataset_prefix directory. Defaults to "bobs-burgers-dining".
        device (str, optional): Defaults to "cpu".
        niters (int, optional): number of iters for coarse optimization. Defaults to 2000.
        lr (float, optional): learning rate of coarse optimization. Defaults to 0.01.
        affine_loss_mult (float, optional): optimization objective. Defaults to 1000.0.
        scale_loss_mult (float, optional): how close scene scale should remain to scale_loss_target. Defaults to 1e3.
        scale_loss_target (float, optional): mean scale of scene. Defaults to 1.0.
        offset_loss_mult (float, optional): how strong to push back negative z values. Defaults to 1e3.
        focal_mag_mult (float, optional): how close we want cameras to their content. Defaults to 1e-3.
        aspect_mult (float, optional): how similar we want cameras to their original image aspect ratio. Defaults to 1.
        up_mult (float, optional): how close we want the cameras to be in the upright position. Defaults to 0.
        niters_warp (int, optional): number of iters for dense optimization Defaults to 1000.
        lr_warp (float, optional): learning rate of dense optimization. Defaults to 0.01.
        arap_mult (float, optional): how much we allow uv warping. Defaults to 1.
        zs_diff_mult (float, optional): how much we allow z warping. Defaults to 1.
        output_prefix (Path, optional): path for outputs of method Defaults to Path("outputs").
        output_method (Path, optional): dataset of specific experiment outputs. Defaults to Path("run").
        nerfstudio_folder (Path, optional): nerfstudio data path. Defaults to Path("data/nerfstudio").
        ply_downscale_factor (int, optional): every other increment for pointcloud. Defaults to 4.
        view_point_cloud (bool, optional): view pc in viser. Defaults to True.
        save_geometry (bool, optional): save point cloud for export. Defaults to True.
        write_to_nerfstudio (bool, optional): write to nerfstudio folder to run toon3d method. Defaults to True.
        port (int, optional): port for viser server. Defaults to 7007.
        seed (int, optional): random seed for experiments. Defaults to None.

    Returns:
        None
    """

    output_folder = output_prefix / dataset / output_method / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_folder.mkdir(parents=True)

    images, heights, widths, points, points_normed, points_mask, depths, masks = load_dataset(data_prefix / dataset, device)
    n, m = points.shape[:2]

    # for ablations
    if seed is not None:
        torch.manual_seed(seed)

    # offset relative depths
    depth_offset = 0.2
    depths = [depths[i] + depth_offset for i in range(n)]

    # draw keypoints on image
    point_colors = torch.rand((m, 3)).to(device)
    colors = point_colors[None].repeat(n, 1, 1)
    colors[~points_mask] = torch.tensor([1, 1, 1]).float().to(device)
    output_images_dir = Path(output_folder / "images")
    output_images_dir.mkdir(parents=True)
    for i in range(n):
        keypoint_image = get_images_with_keypoints(
            images[i].permute(2, 0, 1)[None], points_normed[i][None].cpu(), colors[i][None], keypoint_size=5
        )[0].permute(1, 2, 0)
        
        mediapy.write_image(output_images_dir / f"{i:02d}.png", keypoint_image)

    # make borders
    border_masks = []
    for ndx in range(n):
        pts = points[ndx][points_mask[ndx]]
        spls = get_simplices(pts)

        mask_mesh = WarpMesh(pts, spls, heights[ndx], widths[ndx], pts, device=device)
        border_mask = mask_mesh.apply_warp(torch.zeros(heights[ndx], widths[ndx], 1))[..., 0]
        border_mask = 1 - border_mask.cpu().numpy()

        masks[ndx] = masks[ndx] * border_mask
        border_masks.append(border_mask)

    ########################################
    ###### coarse camera optimization ######
    ########################################
    CONSOLE.print("[bold green] Camera optimization")

    # make pairs for comparison in optimization
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs.append([i, j])

    pairs = torch.tensor(pairs).T
    pairs_mask = torch.logical_and(points_mask[pairs][0], points_mask[pairs][1])[...,None] * 1

    # extract manual correspodences for optimization
    us = points[..., 0].int()
    vs = points[..., 1].int()

    us_normed = us / (widths[..., None] - 1) * 2 - 1
    vs_normed = vs / (heights[..., None] - 1) * 2 - 1

    zs = torch.stack([depths[ndx][vs[ndx], us[ndx]] for ndx in range(n)])

    dzs = nn.Parameter(torch.zeros([n]).float().to(device)) # delta zs
    szs = nn.Parameter(torch.ones([n]).float().to(device)) # scale zs

    # make cameras and optimizer
    coarse_cameras = Cameras(n).to(device)

    # load cameras if exist
    optimizer = torch.optim.Adam(list(coarse_cameras.parameters()) + [szs, dzs], lr=lr)

    # optimization loop
    pbar = tqdm(range(niters))

    for i in pbar:

        new_zs = zs * szs[...,None] + dzs[...,None]
        points_3d = coarse_cameras(us_normed, vs_normed, new_zs)
        paired_points = points_3d[pairs]
        affine_loss = affine_loss_mult * torch.mean(((paired_points[0] - paired_points[1]) ** 2) * pairs_mask)

        scale_loss = scale_loss_mult * ((torch.mean(szs) - scale_loss_target) ** 2 + torch.mean(torch.relu(-szs) ** 2))
        offset_loss = offset_loss_mult * torch.mean(torch.relu(-dzs) ** 2)
        focal_mag = focal_mag_mult * (torch.mean(coarse_cameras.fxs) + torch.mean(coarse_cameras.fys))
        focal_fix = aspect_mult * torch.mean((coarse_cameras.fys / coarse_cameras.fxs - widths / heights) ** 2)
        up_loss = up_mult * torch.mean((torch.abs((coarse_cameras.Rs @ torch.tensor([0, 1, 0]).float().to(device))[:,1] - 1)))
        loss = affine_loss + scale_loss + offset_loss + focal_mag + focal_fix + up_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep first point cloud fixed
        with torch.no_grad():
            coarse_cameras.quats[0] = torch.tensor([1, 0, 0, 0])
            coarse_cameras.ts[0] = torch.zeros([3])

            pbar.set_description(f"Loss: {loss:.2e}, 3D: {affine_loss:.2e}", refresh=True)

            # unbounded scale and offset
            if scale_loss_mult == 0:
                szs[:] = 1
            if offset_loss_mult == 0:
                dzs[:] = 0

    # make new cameras
    refined_cameras = Cameras(n, 
                            coarse_cameras.fxs.detach().clone(), 
                            coarse_cameras.fys.detach().clone(), 
                            coarse_cameras.quats.detach().clone(), 
                            coarse_cameras.ts.detach().clone()).to(device)

    new_zs = zs * szs[...,None] + dzs[...,None]

    ################################
    ###### make triangulation ######
    ################################
    CONSOLE.print("[bold green] Image triangulation")

    frame_meshes = FrameMeshes(points, points_mask, depths)

    # face verts for ARAP
    face_verts_packed = frame_meshes.points_packed[frame_meshes.simplices_packed].mT

    ###################################
    ###### epipolar optimization ######
    ###################################
    CONSOLE.print("[bold green] Image Warping")

    dzs_warp = nn.Parameter(dzs.detach().clone()) # delta zs
    szs_warp = nn.Parameter(szs.detach().clone()) # scale zs

    # set up optimizer
    optimizer = torch.optim.Adam(list(frame_meshes.parameters()) + list(refined_cameras.parameters()) + [szs_warp, dzs_warp], lr=lr_warp)

    # optimization loop
    pbar = tqdm(range(niters_warp))

    tri_areas = signed_areas(face_verts_packed)

    for i in pbar:

        # arap loss
        warped_face_verts_packed = frame_meshes.warped_points_packed[frame_meshes.simplices_packed].mT
        arap_loss = arap_mult * face_verts_arap_loss(face_verts_packed, warped_face_verts_packed)

        # area loss
        area_loss = calc_area_loss(warped_face_verts_packed, min_area=tri_areas * 0.1)

        # szs, dzs packed for depth_ranking
        szs_warp_packed = szs_warp.repeat_interleave(torch.tensor(frame_meshes.num_points_per_mesh).to(device)).detach()
        dzs_warp_packed = dzs_warp.repeat_interleave(torch.tensor(frame_meshes.num_points_per_mesh).to(device)).detach()

        # z_diff_loss
        zs_packed = frame_meshes.zs_packed * szs_warp_packed + dzs_warp_packed
        warped_zs_packed = frame_meshes.warped_zs_packed * szs_warp_packed + dzs_warp_packed
        zs_diff_loss = zs_diff_mult * torch.mean((zs_packed - warped_zs_packed).abs())

        # affine loss
        warped_us = frame_meshes.warped_corr_points_padded[...,0]
        warped_vs = frame_meshes.warped_corr_points_padded[...,1]
        warped_us_normed = warped_us / (widths[..., None] - 1) * 2 - 1
        warped_vs_normed = warped_vs / (heights[..., None] - 1) * 2 - 1
        warped_zs = frame_meshes.warped_corr_zs_padded * szs_warp[...,None] + dzs_warp[...,None]

        points_3d = refined_cameras(warped_us_normed, warped_vs_normed, warped_zs)
        paired_points = points_3d[pairs]
        affine_loss = affine_loss_mult * torch.mean(((paired_points[0] - paired_points[1]) ** 2) * pairs_mask)

        scale_loss = scale_loss_mult * ((torch.mean(szs_warp) - scale_loss_target) ** 2 + torch.mean(torch.relu(-szs_warp) ** 2))
        offset_loss = offset_loss_mult * torch.mean(torch.relu(-dzs_warp) ** 2)
        focal_mag = focal_mag_mult * (torch.mean(refined_cameras.fxs) + torch.mean(refined_cameras.fys))
        focal_fix = aspect_mult * torch.mean((refined_cameras.fys / refined_cameras.fxs - widths / heights) ** 2)
        up_loss = up_mult * torch.mean((torch.abs((refined_cameras.Rs @ torch.tensor([0, 1, 0]).float().to(device))[:,1] - 1)))

        loss = 0.0
        loss += affine_loss + scale_loss + offset_loss + focal_mag + focal_fix
        loss += arap_loss + area_loss + zs_diff_loss + up_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            refined_cameras.quats[0] = torch.tensor([1, 0, 0, 0])
            refined_cameras.ts[0] = torch.zeros([3])

            # unbounded scale and offset
            if scale_loss_mult == 0:
                szs_warp[:] = 1

            if offset_loss_mult == 0:
                dzs_warp[:] = 0

            corrs_lthan_min = frame_meshes.corr_zs_padded + frame_meshes.delta_corr_zs_padded < 0.1
            if torch.any(corrs_lthan_min):
                frame_meshes.delta_corr_zs_padded[corrs_lthan_min] = 0.1 - frame_meshes.corr_zs_padded[corrs_lthan_min]

        pbar.set_description(f"Loss: {loss:.2e}, 3D: {affine_loss:.2e}, ARAP: {arap_loss:.2e}, Z: {zs_diff_loss:.2e}", refresh=True)

    ### create meshes for warping ###
    all_meshes = []

    for ndx in range(n):
        warped_mesh_points = frame_meshes.warped_points_list[ndx]
        uv_points = frame_meshes.points_list[ndx]
        simplices = frame_meshes.simplices_list[ndx]

        mesh = WarpMesh(warped_mesh_points, simplices, heights[ndx], widths[ndx], uv_points, device=device)

        all_meshes.append(mesh)

    # warp images and depths
    warped_images = []
    warped_depths = []
    warped_masks = []

    warp_visuals_dir = output_folder / "visuals_warp"
    output_warped_images_dir = Path(warp_visuals_dir / "warped_images")
    output_warped_depths_dir = Path(warp_visuals_dir / "warped_depths")
    output_warped_masks_dir = Path(warp_visuals_dir / "warped_masks")
    output_warp_colormap_dir = Path(warp_visuals_dir / "warp_colormaps")

    warp_visuals_dir.mkdir(parents=True, exist_ok=True)
    output_warped_images_dir.mkdir(parents=True)
    output_warped_depths_dir.mkdir(parents=True)
    output_warped_masks_dir.mkdir(parents=True)
    output_warp_colormap_dir.mkdir(parents=True)

    CONSOLE.print("[bold green] Rendering Images")

    for ndx in range(len(all_meshes)):
        mesh = all_meshes[ndx]
        image = images[ndx]
        depth_map = depths[ndx]
        delta_zs = frame_meshes.delta_zs_list[ndx]
        mask = torch.from_numpy(masks[ndx]).float()

        warped_image = mesh.apply_warp(image).cpu().detach()
        warped_depth_uv = mesh.apply_warp(depth_map[...,None]).cpu().detach()[...,0]
        warped_depth_z = mesh.vertex_coloring(delta_zs).cpu().detach()
        warped_mask = mesh.apply_warp(mask[...,None]).cpu().detach()[...,0]

        warped_depth = warped_depth_uv + warped_depth_z
        warped_mask = warped_mask.to(torch.uint8).numpy()
        warped_mask[warped_mask < 100] = 0
        warped_mask[warped_mask >= 100] = 255

        wire_img = draw_tris(mesh, image = warped_image.numpy())
        points_img = draw_points(mesh, image=wire_img, points=frame_meshes.warped_corr_points_list[ndx], colors=point_colors[points_mask[ndx]].tolist())

        border_mask = border_masks[ndx]
        image[border_mask == 0] = 1
        warp_transition = ((warped_image * 0.5) + (image * 0.5)).numpy()

        warp_transition = draw_tris(mesh, points= mesh.uv_points, image = warp_transition, color = (0.5, 0.5, 0.5)) # draw old verts
        warp_transition = draw_points(mesh, image=warp_transition, points=frame_meshes.corr_points_list[ndx], colors=((1+point_colors[points_mask[ndx]])/2).tolist())

        warp_transition = draw_tris(mesh, image = warp_transition)
        warp_transition = draw_points(mesh, image=warp_transition, points=frame_meshes.warped_corr_points_list[ndx], colors=point_colors[points_mask[ndx]].tolist())
        
        colors_ = ((1+point_colors[points_mask[ndx]])/2).tolist()
        for old_point, new_point, color_ in zip(mesh.uv_points, mesh.points, colors_):
            x0, y0 = old_point[0].int().item(), old_point[1].int().item()
            x1, y1 = new_point[0].int().item(), new_point[1].int().item()

            warp_transition = cv2.arrowedLine(warp_transition, (x0, y0), (x1, y1), color_, 2) 

        mediapy.write_image(output_warp_colormap_dir / f"{ndx:02d}.png", warp_transition)

        # save warps to outputs
        mediapy.write_image(output_warped_images_dir / f"{ndx:02d}.png", warped_image.numpy())
        mediapy.write_image(output_warped_images_dir / f"wire_{ndx:02d}.png", points_img)
        mediapy.write_image(output_warped_images_dir / f"duo_{ndx:02d}.png", np.concatenate([warped_image.numpy(), points_img], axis=1))

        mediapy.write_image(output_warped_depths_dir / f"{ndx:02d}.png", warped_depth)
        mediapy.write_image(output_warped_depths_dir / f"zdepth_{ndx:02d}.png", warped_depth_z.numpy())
        mediapy.write_image(output_warped_masks_dir / f"{ndx:02d}.png", warped_mask)

        warped_images.append(warped_image)
        warped_depths.append(warped_depth)
        warped_masks.append(warped_mask)


    # rescale depths
    depths = [(depth_map * sz[...,None]) + dz[...,None] for depth_map, dz, sz in zip(depths, dzs, szs)]
    warped_depths = [(depth_map.to(device) * sz[...,None]) + dz[...,None] for depth_map, dz, sz in zip(warped_depths, dzs_warp, szs_warp)]


    #######################
    ###### make json ######
    #######################

    nerfstudio_dir = output_folder / "nerfstudio"
    nerfstudio_images_dir = nerfstudio_dir / "images"
    nerfstudio_depths_dir = nerfstudio_dir / "depths"
    nerfstudio_masks_dir = nerfstudio_dir / "masks"
    nerfstudio_points_dir = nerfstudio_dir / "points"
    nerfstudio_meshes_dir = nerfstudio_dir / "meshes"
    nerfstudio_plys_dir = nerfstudio_dir / "plys"

    nerfstudio_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_images_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_depths_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_masks_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_points_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_meshes_dir.mkdir(parents=True, exist_ok=True)
    nerfstudio_plys_dir.mkdir(parents=True, exist_ok=True)

    # store mesh points
    
    meshes_points_dict = {"corr_points": frame_meshes.corr_points_list, 
                    "boundary_points": [], 
                    "inner_points": [],
                    "points": frame_meshes.points_list}

    warped_meshes_points_dict = {"corr_points": frame_meshes.warped_corr_points_list, 
                            "boundary_points": [], # grad still attached to boundary_points
                            "inner_points": [], # grad still attached to inner_points
                            "points": frame_meshes.warped_points_list
                            }

    # json
    transforms = make_transforms_json(
        refined_cameras.fxs,
        refined_cameras.fys,
        refined_cameras.Rs,
        refined_cameras.ts,
        widths,
        heights,
    )
    with open(nerfstudio_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=4)

    # points
    torch.save(points, output_folder / "nerfstudio" / "points/points.pt")
    torch.save(points_mask, output_folder / "nerfstudio" / "points/points_mask.pt")

    # mesh points
    torch.save(meshes_points_dict, output_folder / "nerfstudio" / "meshes/meshes_points.pt")
    torch.save(warped_meshes_points_dict, output_folder / "nerfstudio" / "meshes/warped_meshes_points.pt")
    torch.save(frame_meshes.simplices_list, output_folder / "nerfstudio" / "meshes/simplices.pt")

    # images
    for i, img in enumerate(images):
        mediapy.write_image(nerfstudio_images_dir / f"{i:05}.png", warped_images[i].cpu().numpy())

    # depths
    for i in range(n):
        np.save(nerfstudio_depths_dir / f"{i:05d}", warped_depths[i][..., None].cpu().detach().numpy())

    # masks
    for i in range(n):
        mediapy.write_image(nerfstudio_masks_dir / f"{i:05}.png", warped_masks[i])

    # camera params and szs/dzs
    camera_params_dir = output_folder / "camera_params"
    output_cameras_dir = Path(camera_params_dir / "cameras")
    output_szs_dir = Path(camera_params_dir / "szs")
    output_dzs_dir = Path(camera_params_dir / "dzs")

    camera_params_dir.mkdir(parents=True)
    output_cameras_dir.mkdir(parents=True)
    output_szs_dir.mkdir(parents=True)
    output_dzs_dir.mkdir(parents=True)

    torch.save(coarse_cameras.state_dict(), output_cameras_dir / "coarse_cameras.pt")
    torch.save(refined_cameras.state_dict(), output_cameras_dir / "refined_cameras.pt")

    torch.save(szs, output_szs_dir / "szs.pt")
    torch.save(szs_warp, output_szs_dir / "szs_warp.pt")

    torch.save(dzs, output_dzs_dir / "dzs.pt")
    torch.save(dzs_warp, output_dzs_dir / "dzs_warp.pt")

    
    # coarse
    valid_points_3d = [mask[::ply_downscale_factor, ::ply_downscale_factor].T.flatten() != 0 for mask in masks]
    coarse_dense_points_3d = make_dense_points_3d(coarse_cameras, depths, ply_downscale_factor)
    coarse_dense_points_3d = [coarse_dense_points_3d[ndx][valid_points_3d[ndx]] for ndx in range(n)]
    images_colors = [image.permute(1, 0, 2)[::ply_downscale_factor, ::ply_downscale_factor].cpu().flatten(0, 1) for image in images]
    images_colors = [images_colors[ndx][valid_points_3d[ndx]] for ndx in range(n)]

    # refined
    warped_valid_points_3d = [mask[::ply_downscale_factor, ::ply_downscale_factor].T.flatten() > 0.1 for mask in warped_masks]
    warped_dense_points_3d = make_dense_points_3d(refined_cameras, warped_depths, ply_downscale_factor)
    warped_dense_points_3d = [warped_dense_points_3d[ndx][warped_valid_points_3d[ndx]] for ndx in range(n)]
    warped_images_colors = [image.permute(1, 0, 2)[::ply_downscale_factor, ::ply_downscale_factor].cpu().flatten(0, 1) for image in warped_images]
    warped_images_colors = [warped_images_colors[ndx][warped_valid_points_3d[ndx]] for ndx in range(n)]
    

    # verts and sparse points
    mesh_verts_list = []
    warped_mesh_verts_list = []

    coarse_corrs_3d = []
    refined_corrs_3d = []

    for ndx in range(n):
        us = frame_meshes.points_list[ndx][...,0]
        vs = frame_meshes.points_list[ndx][...,1]
        us_normed = us / (widths[ndx] - 1) * 2 - 1
        vs_normed = vs / (heights[ndx] - 1) * 2 - 1
        zs = frame_meshes.zs_list[ndx] * szs[ndx] + dzs[ndx]
        points_3d = coarse_cameras(us_normed, vs_normed, zs)[ndx]

        mesh_verts_list.append(points_3d)
        coarse_corrs_3d.append(points_3d[:frame_meshes.num_corrs_per_mesh[ndx]])

        warped_us = frame_meshes.warped_points_list[ndx][...,0]
        warped_vs = frame_meshes.warped_points_list[ndx][...,1]
        warped_us_normed = warped_us / (widths[ndx] - 1) * 2 - 1
        warped_vs_normed = warped_vs / (heights[ndx] - 1) * 2 - 1
        warped_zs = frame_meshes.warped_zs_list[ndx] * szs_warp[ndx] + dzs_warp[ndx]
        warped_points_3d = refined_cameras(warped_us_normed, warped_vs_normed, warped_zs)[ndx]

        warped_mesh_verts_list.append(warped_points_3d)
        refined_corrs_3d.append(warped_points_3d[:frame_meshes.num_corrs_per_mesh[ndx]])


    visuals_3d_dir = output_folder / "visuals_3d"
    output_verts_dir = visuals_3d_dir / "verts"
    output_corrs_dir = visuals_3d_dir / "corrs"
    output_dense_points_dir = visuals_3d_dir / "dense_points"
    output_img_colors_dir = visuals_3d_dir / "img_colors"

    visuals_3d_dir.mkdir(parents=True, exist_ok=True)
    output_corrs_dir.mkdir(parents=True, exist_ok=True)
    output_verts_dir.mkdir(parents=True, exist_ok=True)
    output_dense_points_dir.mkdir(parents=True, exist_ok=True)
    output_img_colors_dir.mkdir(parents=True, exist_ok=True)

    torch.save(mesh_verts_list, output_verts_dir / "verts.pt")
    torch.save(warped_mesh_verts_list, output_verts_dir / "warped_verts.pt")

    torch.save(coarse_corrs_3d, output_corrs_dir / "corrs.pt")
    torch.save(refined_corrs_3d, output_corrs_dir / "warped_corrs.pt")
    
    torch.save(coarse_dense_points_3d, output_dense_points_dir / "dense_points.pt")
    torch.save(warped_dense_points_3d, output_dense_points_dir / "warped_dense_points.pt")

    torch.save(images_colors, output_img_colors_dir / "img_colors.pt")
    torch.save(warped_images_colors, output_img_colors_dir / "warped_img_colors.pt")

    if write_to_nerfstudio or save_geometry:
        CONSOLE.print("[bold green] Writing plys")

        # create plys (point clouds)
        plys = make_plys(warped_images_colors, warped_dense_points_3d)
        for i, ply in enumerate(plys):
            with open(output_folder / "nerfstudio" / "plys" / f"{i:05d}.ply", "w") as f:
                f.write(ply)

        # make a point cloud ply with all the plys combined ...
        with open(output_folder / "nerfstudio" / "plys" / "all.ply", "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {sum([len(ply.splitlines()) - 14 for ply in plys])}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uint8 red\n")
            f.write("property uint8 green\n")
            f.write("property uint8 blue\n")
            f.write("end_header\n")

            for ply in plys:
                for line in ply.splitlines()[14:]:
                    f.write(line + "\n")

    if write_to_nerfstudio:
        # remove folder if exists
        nerfstudio_folder.mkdir(exist_ok=True, parents=True)
        if nerfstudio_folder:
            if (nerfstudio_folder / dataset).exists():
                shutil.rmtree(nerfstudio_folder / dataset)
            shutil.copytree(output_folder / "nerfstudio", nerfstudio_folder / dataset)

    if view_point_cloud:

        sparse_point_colors = [point_colors[points_mask[ndx]] for ndx in range(n)]
        mesh_colors = [np.random.rand(3) for _ in range(n)]
        
        server = viser.ViserServer(request_share_url=True, port=port)

        view_pcs_cameras(server, 
                         images, # cameras
                         coarse_cameras,
                         coarse_dense_points_3d, # dense pc
                         images_colors, 
                         coarse_corrs_3d, # sparse pc
                         sparse_point_colors,
                         mesh_verts_list, # mesh
                         frame_meshes.simplices_list,
                         mesh_colors,
                         visible=False,
                         prefix="coarse")
        
        view_pcs_cameras(server, 
                         warped_images, # cameras
                         refined_cameras,
                         warped_dense_points_3d, # dense pc
                         warped_images_colors, 
                         refined_corrs_3d, # sparse pc
                         sparse_point_colors,
                         warped_mesh_verts_list, # mesh
                         frame_meshes.simplices_list,
                         mesh_colors,
                         prefix="refined")
        
        while True:
            pass
    
    metrics = None
    return metrics


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()