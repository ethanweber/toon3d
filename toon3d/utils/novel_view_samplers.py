"""
Code to sample novel views given the training camera distribution.
"""

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix


def sample_interpolated_camera(dataset_cameras: Cameras) -> Cameras:
    """
    Sample a camera by interpolating between two random cameras in the dataset.
    dataset_cameras is shape (num_cameras, ...) and the output is shape (...) since we lose the batch dimension.
    """
    num_cameras = dataset_cameras.shape[0]
    idx1, idx2 = torch.randperm(num_cameras)[:2]
    camera1 = dataset_cameras[idx1]
    camera2 = dataset_cameras[idx2]
    pose1 = camera1.camera_to_worlds.cpu().numpy()
    pose2 = camera2.camera_to_worlds.cpu().numpy()
    quat1 = quaternion_from_matrix(pose1[:3, :3])
    quat2 = quaternion_from_matrix(pose2[:3, :3])
    t = torch.rand(1).item()
    quat = quaternion_slerp(quat1, quat2, t)
    R = quaternion_matrix(quat)[:3, :3]
    T = pose1[:3, 3:4] * (1 - t) + pose2[:3, 3:4] * t
    R = torch.from_numpy(R)
    T = torch.from_numpy(T)

    fx1 = camera1.fx.item()
    fx2 = camera2.fx.item()
    t_fx = torch.rand(1).item()
    fx = fx1 * (1 - t_fx) + fx2 * t_fx
    fy1 = camera1.fy.item()
    fy2 = camera2.fy.item()
    t_fy = torch.rand(1).item()
    fy = fy1 * (1 - t_fy) + fy2 * t_fy

    cx = 256.0
    cy = 256.0
    width = 512
    height = 512
    camera_to_worlds = camera1.camera_to_worlds.clone().cpu()
    camera_to_worlds[:3, :3] = R
    camera_to_worlds[:3, 3:4] = T
    camera_perturb = Cameras(
        camera_to_worlds=camera_to_worlds,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )
    return camera_perturb
