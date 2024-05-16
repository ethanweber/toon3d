"""
Code for drawing on images.
"""


import cv2
import numpy as np

import torch


def draw_lines_on_image(image, lines, colors, thickness=2):
    """
    Image of shape [h,w,3] in range [0,255].
    Lines as list of [(x,y),(x,y)] coordinates as pixels.
    Colors in range [0,255].
    """
    new_image = np.ascontiguousarray(image, dtype=np.uint8)
    for (p0, p1), color in zip(lines, colors):
        cv2.line(new_image, p0, p1, color, thickness)
    return new_image


def draw_keypoints_on_image(image, keypoints, colors, radius=3, thickness=-1):
    """
    Image of shape [h,w,3] in range [0,255].
    Keypoints as (x,y) coordinates as pixels.
    Colors in range [0,255].
    """
    new_image = np.ascontiguousarray(image, dtype=np.uint8)
    for keypoint, color in zip(keypoints, colors):
        cv2.circle(new_image, keypoint, radius, color, thickness)
    return new_image


def get_images_with_keypoints(images, keypoints, colors, keypoint_size=10, lines=None, line_colors=None):
    """Returns the batch of images with keypoints drawn in the colors.
    Images of shape [b, c, h, w] in range [0,1].
    Keypoints of shape [b, n, 2] as (x,y) coordinates in range [-1,1].
    Colors of shape [b, n, 3] in range (0, 1).
    Lines of shape [b, n, 2, 2] as (x,y) coordinates in range [-1,1].
    """
    device = images.device
    b, _, h, w = images.shape
    new_images = []
    for idx in range(b):
        im = np.ascontiguousarray(
            (images[idx].permute(1, 2, 0).detach().clone().cpu().numpy() * 255.0).astype("uint8")
        ).astype("uint8")

        if lines is not None:
            li = (lines[idx] * 0.5 + 0.5) * torch.tensor([w - 1, h - 1], device=device)
            if line_colors is None:
                co = [(0, 255, 255) for _ in range(len(li))]
            else:
                co = (line_colors[idx] * 255.0).detach().clone().cpu().numpy().astype("uint8")
                co = [(int(r), int(g), int(b)) for r, g, b in co]
            li = [((int(p[0, 0]), int(p[0, 1])), (int(p[1, 0]), int(p[1, 1]))) for p in li]
            im = draw_lines_on_image(im, li, co)

        if keypoints is not None:
            ke = (keypoints[idx] * 0.5 + 0.5) * torch.tensor([w - 1, h - 1], device=device)
            ke = [(int(x), int(y)) for x, y in ke]
            co = (colors[idx] * 255.0).detach().clone().cpu().numpy().astype("uint8")
            co = [(int(r), int(g), int(b)) for r, g, b in co]
            im = draw_keypoints_on_image(im, ke, co, radius=keypoint_size)

        new_images.append(im)
    new_images = np.stack(new_images, axis=0)
    new_images = torch.tensor(new_images).permute(0, 3, 1, 2).float().to(device) / 255.0
    return new_images
