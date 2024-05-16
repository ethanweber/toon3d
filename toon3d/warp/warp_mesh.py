"""
Code for warping a 2D mesh.
"""

import cv2 
import torch
import torch.nn as nn
from scipy.spatial import Delaunay
import numpy as np

def get_simplices(points):
    tri = Delaunay(points.detach().cpu().numpy())
    simplices = torch.tensor(tri.simplices, device=points.device)
    return simplices

def get_half_edges(faces):
    half_edges = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=0).permute(1, 0, 2).flatten(0, 1)
    return half_edges

def get_edge_twins(half_edges, edge_pairs=None):
    if edge_pairs is None:
        edge_pairs = get_edge_pairs(half_edges)

    twins = torch.full((len(half_edges),), -1).to(half_edges.device)
    twins[edge_pairs[0]] = edge_pairs[1]
    twins[edge_pairs[1]] = edge_pairs[0]

    return twins

def get_edge_pairs(half_edges):
    unique, occurances = torch.unique(half_edges.sort().values, dim=0, return_inverse=True)
    expanded_occurances = occurances[...,None].expand(-1, len(occurances))
    matches = expanded_occurances == expanded_occurances.T

    diagonal = torch.eye(len(matches), dtype=torch.bool) # don't let it match to itself
    matches[diagonal] = False

    edge_pairs = matches.nonzero().T
    return edge_pairs

### drawing utils

def draw_tris(mesh, image=None, points=None, color=None, thickness=1):

    edges = mesh.half_edges.sort(1).values.unique(dim=0)
    if points is None:
        points = mesh.points

    edge_points = points[edges]

    if color is None:
        color = (0, 0, 0)

    if image is None:
        image = mesh.image[0].cpu().numpy()

    image = image.copy()

    for edge_point in edge_points:
        x1, y1, x2, y2 = edge_point.flatten().int()
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        image = cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)
    
    return image

def draw_points(mesh, image=None, points=None, colors=None):

    if image is None:
        image = mesh.image[0].cpu().numpy()

    image = image.copy()
    
    if colors is None:
        colors = [(1.0, 0, 0) * len(points)]

    for point, color in zip(points, colors):
        x, y = point[0].int().item(), point[1].int().item()
        image = cv2.circle(image, (x, y), 2, color, thickness=5)
    return image

class FrameMeshes(nn.Module):
    def __init__(self, corr_points_padded, corr_masks_padded, depths):
        super().__init__()

        self.n = len(corr_points_padded)

        self.corr_points_padded = corr_points_padded.clone().float()
        self.corr_masks_padded = corr_masks_padded.clone()

        # make lists
        self.corr_points_list = [pts[pts_mask] for pts, pts_mask in zip(corr_points_padded, corr_masks_padded)]
        self.simplices_list = []

        for ndx in range(self.n):

            corr_points = self.corr_points_list[ndx]

            simplices = get_simplices(corr_points)
            self.simplices_list.append(simplices)

        self.points_list = []
        for corr_points in self.corr_points_list:
            self.points_list.append(corr_points)

        # num coords
        self.num_corrs_per_mesh = [len(cp) for cp in self.corr_points_list]
        self.num_points_per_mesh = [len(p) for p in self.points_list]

        # packed 
        self.corr_points_packed = torch.cat(self.corr_points_list)
        self.points_packed = torch.cat(self.points_list)

        # zs
        self.corr_zs_list = [depths[ndx][self.corr_points_list[ndx][...,1].int(), self.corr_points_list[ndx][...,0].int()] for ndx in range(self.n)]
        self.zs_list = []
        for corr_zs in self.corr_zs_list:
            self.zs_list.append(corr_zs)

        self.corr_zs_padded = torch.stack([depths[ndx][self.corr_points_padded[ndx,...,1].int(), self.corr_points_padded[ndx,...,0].int()] for ndx in range(self.n)])
        self.corr_zs_packed = self.corr_zs_padded[self.corr_masks_padded]
        self.zs_packed = torch.cat(self.zs_list)

        # deltas
        self.delta_corr_points_padded = nn.Parameter(torch.zeros_like(self.corr_points_padded))
        self.delta_corr_zs_padded = nn.Parameter(torch.zeros_like(self.corr_zs_padded))

        # pack simplices
        simplices_shifts = torch.cat([torch.tensor([0]), torch.tensor(self.num_points_per_mesh).cumsum(0)[:-1]])
        self.simplices_shifted_list = [simplices + shift for simplices, shift in zip(self.simplices_list, simplices_shifts)]
        self.simplices_packed = torch.cat(self.simplices_shifted_list)

        # import pdb; pdb.set_trace()


    ### delta points ###
        
    @property
    def delta_corr_points_packed(self):
        return self.delta_corr_points_padded[self.corr_masks_padded]

    ### warped points ###
    
    #packed
    @property
    def warped_corr_points_packed(self):
        return self.corr_points_packed + self.delta_corr_points_packed
    
    @property
    def warped_points_packed(self):
        return torch.cat(self.warped_points_list)
    
    # padded
    @property
    def warped_corr_points_padded(self):
        return self.corr_points_padded + self.delta_corr_points_padded
    
    # list
    @property
    def warped_corr_points_list(self):
        return self.warped_corr_points_packed.split(self.num_corrs_per_mesh)
    
    @property
    def warped_points_list(self):
        warped_points_list = []
        for warped_corr_points in self.warped_corr_points_list:
            warped_points_list.append(warped_corr_points)

        return warped_points_list
    
    ### delta zs ###

    # packed
    @property
    def delta_corr_zs_packed(self):
        return self.delta_corr_zs_padded[self.corr_masks_padded]
    
    @property
    def delta_zs_packed(self):
        return torch.cat(self.delta_zs_list)

    # list
    @property
    def delta_corr_zs_list(self):
        return self.delta_corr_zs_packed.split(self.num_corrs_per_mesh)
    
    @property
    def delta_zs_list(self):
        delta_zs_list = []
        for delta_corr_zs in self.delta_corr_zs_list:
            delta_zs_list.append(delta_corr_zs)

        return delta_zs_list
    
    ### warped zs ###

    # padded
    @property
    def warped_corr_zs_padded(self):
        return self.corr_zs_padded + self.delta_corr_zs_padded
    
    # packed
    @property
    def warped_zs_packed(self):
        return self.zs_packed + self.delta_zs_packed

    # list
    @property
    def warped_corr_zs_list(self):
        return (self.corr_zs_packed + self.delta_corr_zs_packed).split(self.num_corrs_per_mesh)
    
    @property
    def warped_zs_list(self):
        return self.warped_zs_packed.split(self.num_points_per_mesh)
    
def barycentric_coordinates_batch(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w

class WarpMesh(nn.Module):
    def __init__(self, points, simplices, height, width, uv_points=None, device="cpu"):
        """
        Creates a Mesh designed to fit an input image with triangulation
        """
        super().__init__()

        self.height = height
        self.width = width

        self.points = points.to(device).float()
        self.faces = simplices.to(device)

        if uv_points is None:
            uv_points = self.points.clone()
        self.uv_points = uv_points.to(device)

        self.points_normed =  self.points.clone() / torch.tensor([width - 1, height - 1]).to(device) * 2 - 1
        self.uv_points_normed =  self.uv_points.clone() / torch.tensor([width - 1, height - 1]).to(device) * 2 - 1

        self.eps = 1 / (max(height - 1, width - 1))

        self.half_edges = get_half_edges(self.faces)
        self.edge_pairs = get_edge_pairs(self.half_edges)
        self.edge_twins = get_edge_twins(self.half_edges, self.edge_pairs)

        self.device = device

    def apply_warp(self, image):
        """Applies warp to image

        Args:
            Image (torch.tensor): Image size (H, W, c)

        Returns:
            Warped Image (torch.tensor): size (H, W, c)
        """
        image = image.permute(2, 0, 1)[None].to(self.device)

        target_image = torch.ones_like(image)
        image_coords = torch.meshgrid(torch.linspace(-1, 1, self.height), torch.linspace(-1, 1, self.width), indexing="ij")
        image_coords = torch.flip(torch.stack(image_coords, dim=-1), dims=[-1]).to(self.device)  # stored as (x,y) coordinates

        for idx0, idx1, idx2 in self.faces:

            # skip if all simplices are zero because thats how we pad
            if idx0 == idx1 == idx2 == 0:
                continue

            # source image (from which we are copying)
            S_a = self.uv_points_normed[idx0][None, None].repeat(self.height, self.width, 1)  # (H, W, 2,)
            S_b = self.uv_points_normed[idx1][None, None].repeat(self.height, self.width, 1)  # (H, W, 2,)
            S_c = self.uv_points_normed[idx2][None, None].repeat(self.height, self.width, 1)  # (H, W, 2,)

            # destination image (to which we are copying)
            a = self.points_normed[idx0][None, None].repeat(self.height, self.width, 1)  # (H, W, 2,)
            b = self.points_normed[idx1][None, None].repeat(self.height, self.width, 1)  # (H, W, 2,)
            c = self.points_normed[idx2][None, None].repeat(self.height, self.width, 1)  # (H, W, 2,)

            # (u, v, w)
            # https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
            v0 = b - a
            v1 = c - a
            v2 = image_coords - a

            # replace dot without the function
            d00 = (v0 * v0).sum(-1)
            d01 = (v0 * v1).sum(-1)
            d11 = (v1 * v1).sum(-1)
            d20 = (v2 * v0).sum(-1)
            d21 = (v2 * v1).sum(-1)

            denom = d00 * d11 - d01 * d01
            w_b = (d11 * d20 - d01 * d21) / denom
            w_c = (d00 * d21 - d01 * d20) / denom
            w_a = 1.0 - w_b - w_c

            grid_sample_coordinates = S_a * w_a.unsqueeze(-1) + S_b * w_b.unsqueeze(-1) + S_c * w_c.unsqueeze(-1)
            _grid = grid_sample_coordinates.unsqueeze(0).float()

            pixels = torch.nn.functional.grid_sample(image, _grid, align_corners=True)[0]

            valid_mask = (w_b >= 0 - self.eps) * (w_b <= 1 + self.eps) * (w_c >= 0 - self.eps) * (w_c <= 1 + self.eps) * (w_b + w_c <= 1 + self.eps)
            mask = valid_mask.float().unsqueeze(0)

            target_image = target_image * (1.0 - mask) + pixels * mask

        return target_image[0].permute(1, 2, 0)
    
    def vertex_coloring(self, vertex_colors):
        img = torch.zeros((self.height, self.width), dtype=torch.float32).to(self.device)

        for face in self.faces:
            a, b, c = self.points[face]
            color_a, color_b, color_c = vertex_colors[face]

            min_x = max(int(torch.min(torch.tensor([a[0], b[0], c[0]]))), 0)
            max_x = min(int(torch.max(torch.tensor([a[0], b[0], c[0]]))), self.width - 1)
            min_y = max(int(torch.min(torch.tensor([a[1], b[1], c[1]]))), 0)
            max_y = min(int(torch.max(torch.tensor([a[1], b[1], c[1]]))), self.height - 1)

            xx, yy = torch.meshgrid(torch.arange(min_x, max_x + 1).to(self.device), torch.arange(min_y, max_y + 1).to(self.device), indexing='xy')
            xx, yy = xx.flatten(), yy.flatten()
            p = torch.stack([xx, yy], dim=-1).float()

            u, v, w = barycentric_coordinates_batch(p, a, b, c)

            mask = (u >= 0) & (v >= 0) & (w >= 0) & (u <= 1) & (v <= 1) & (w <= 1)

            if mask.any():
                color = u[mask] * color_a + v[mask] * color_b + w[mask] * color_c
                img[yy[mask].long(), xx[mask].long()] = color

        return img