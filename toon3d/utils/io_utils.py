"""
Output utils.
"""

import io
import torch
import numpy as np
import cv2

import mediapy
import json
from toon3d.utils.depth_utils import create_discontinuity_mask

def make_transforms_json(fxs, fys, Rs, ts, widths, heights):
    fxs, fys, Rs, ts = fxs.detach().cpu(), fys.detach().cpu(), Rs.detach().cpu(), ts.detach().cpu()
    transforms = {
        "camera_model": "OPENCV",
        "ply_file_path": "plys/all.ply",
        "points": "points/points.pt",
        "points_mask": "points/points_mask.pt",
        "meshes_points": "meshes/meshes_points.pt",
        "warped_meshes_points": "meshes/warped_meshes_points.pt",
        "simplices": "meshes/simplices.pt",
    }

    frames = []
    for i, (R, t) in enumerate(zip(Rs, ts)):
        flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
        R = R @ flip
        transform_matrix = torch.cat([torch.cat([R, t[...,None]], 1), torch.tensor([[0, 0, 0, 1]])], 0)
        width = widths[i].item()
        height = heights[i].item()
        fx = fxs[i] / 2 * widths[i]
        fy = fys[i] / 2 * heights[i]
        frame = {
            "file_path": f"images/{i:05d}.png",
            "fl_x": fx.item(),
            "fl_y": fy.item(),
            "cx": width // 2,
            "cy": height // 2,
            "w": int(width),
            "h": int(height),
            "transform_matrix": transform_matrix.tolist(),
            "depth_file_path": f"depths/{i:05d}.npy",
            "mask_path": f"masks/{i:05d}.png",
        }
        frames.append(frame)
    transforms["frames"] = frames

    return transforms

def make_dense_points_3d(cameras, depths, factor=4):
    dense_points_3d = []

    for ndx, depth_map in enumerate(depths): 
        height, width = depth_map.shape
        xs = torch.arange(width).to(depth_map.device)
        ys = torch.arange(height).to(depth_map.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        grid_x = (grid_x / (width - 1) * 2) - 1
        grid_y = (grid_y / (height - 1) * 2) - 1

        x = grid_x[::factor, ::factor].flatten()
        y = grid_y[::factor, ::factor].flatten()
        z = depth_map[::factor, ::factor].T.flatten()

        dense_points_3d.append(cameras(x, y, z)[ndx])

    return dense_points_3d

def make_plys(images_colors, points_3d):
    n = len(images_colors)
    plys = []
    for i in range(n):
        image_colors = images_colors[i]
        ply_points = points_3d[i]

        ply = io.StringIO()
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write(f"element vertex {len(ply_points)}\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        ply.write("property uint8 red\n")
        ply.write("property uint8 green\n")
        ply.write("property uint8 blue\n")
        ply.write("end_header\n")

        for point, color in zip(ply_points, image_colors):
            x, y, z = point.to(torch.float)
            r, g, b = (color * 255).to(torch.uint8)
            ply.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b}\n")

        plys.append(ply.getvalue())
        ply.close()

    return plys

def load_dataset(data_path, device="cpu", dilate_mask_iters=4, dilate_lines_thresh=0.05, dilate_lines_iters=3, min_valid_points=3):
    images_path = data_path / "images"
    metadata_path = data_path / "metadata.json"
    points_path = data_path / "points.json"
    depths_path = data_path / "depths"

    metadata_json = json.loads(metadata_path.read_text())

    points_json = json.loads(points_path.read_text())
    valid_images = points_json["validImages"]
    points_list = [[[p["x"], p["y"]] for p in points] for points in points_json["points"]]
    valid_points_list = points_json["validPoints"]
    # update valid_images based on valid_points_list and min_valid_points
    valid_images = [valid_images[i] and sum(valid_points_list[i]) >= min_valid_points for i in range(len(valid_images))]
    valid_polygons = points_json["polygons"]
    image_filenames = sorted(list(images_path.glob("*.png")))

    # only keep the valid indices, based on valid_images
    metadata_json["frames"] = [
        metadata_json["frames"][i] for i in range(len(metadata_json["frames"])) if valid_images[i]
    ]
    points_list = [points_list[i] for i in range(len(points_list)) if valid_images[i]]
    valid_points_list = [valid_points_list[i] for i in range(len(valid_points_list)) if valid_images[i]]
    valid_polygons = [valid_polygons[i] for i in range(len(valid_polygons)) if valid_images[i]]
    image_filenames = [image_filenames[i] for i in range(len(image_filenames)) if valid_images[i]]

    images = [torch.from_numpy(mediapy.read_image(imf)[:, :, :3] / 255.0).float() for imf in image_filenames]
    n = len(images)
    heights = torch.tensor([image.shape[0] for image in images]).to(device)
    widths = torch.tensor([image.shape[1] for image in images]).to(device)

    m = max([len(p) for p in points_list])

    points = torch.full([n, m, 2], -1).float().to(device)
    for i in range(n):
        for j in range(len(points_list[i])):
            points[i, j, 0] = points_list[i][j][0]
            points[i, j, 1] = points_list[i][j][1]

    # points_mask padded with False
    points_mask = torch.zeros_like(points[:, :, 0]) == 1
    for i in range(n):
        for j in range(len(points_list[i])):
            points_mask[i, j] = valid_points_list[i][j]

    # remove points that have no pair
    points_mask.T[(points_mask * 1).sum(0) < 2][:] = False

    depths = []
    for i in range(len(valid_images)):
        depths.append(torch.load(depths_path / f"{i:05d}.pt").squeeze().to(device))
    # only keep the valid indices, based on valid_images
    depths = [depths[i] for i in range(len(depths)) if valid_images[i]]
    max_depth = max([torch.max(depth) for depth in depths])
    depths = [depth / max_depth for depth in depths]

    masks = []
    for i in range(n):
        mask_image = np.ones((heights[i], widths[i]), dtype=np.uint8) * 255
        for j in range(len(metadata_json["frames"][i]["masks"])):
            mask = metadata_json["frames"][i]["masks"][j]
            for k in range(len(mask["polygons"])):
                contour = np.array(mask["polygons"][k]).reshape(-1, 2).astype(np.int32)
                temp_mask_image = np.zeros((heights[i], widths[i]), dtype=np.uint8)
                if valid_polygons[i][j][k]:
                    cv2.fillPoly(temp_mask_image, [contour], 1)
                temp_mask_image = cv2.dilate(temp_mask_image, np.ones((5, 5), np.uint8), iterations=dilate_mask_iters)
                mask_image[temp_mask_image == 1] = 0
        masks.append(mask_image)

    # mask out parts of the depth map where they are big discontinuities
    for i in range(n):
        depth_mask = 1 - create_discontinuity_mask(depths[i][None], dilate_lines_thresh, dilate_lines_iters)[0].float()
        masks[i] *= depth_mask.cpu().numpy().astype(np.uint8)

    shapes = torch.cat([widths[:, None], heights[:, None]], -1).unsqueeze(1).repeat(1, m, 1).to(device)
    points_normed = (points / shapes) * 2 - 1

    return images, heights, widths, points, points_normed, points_mask, depths, masks