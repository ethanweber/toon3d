"""
Camera utilities
"""

import torch
import torch.nn as nn


class Cameras(nn.Module):
    def __init__(self, n, fxs=None, fys=None, quats=None, ts=None):
        super().__init__()

        self.n = n

        # camera intrinsics
        if fxs is None: fxs = torch.full([n], 1.0)
        if fys is None: fys = torch.full([n], 1.0)

        self.fxs = nn.Parameter(fxs)
        self.fys = nn.Parameter(fys)

        # camera extrinsics
        if quats is None: quats = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).repeat(self.n, 1)
        if ts is None: ts = torch.zeros([self.n, 3], dtype=torch.float32)

        self.quats = nn.Parameter(quats)
        self.ts = nn.Parameter(ts)
        
    @property
    def Rs(self):
        quats = self.quats / torch.norm(self.quats, dim=1)[...,None]
        qr, qi, qj, qk = quats.split(1, -1)

        Rs = (
            torch.stack(
                [
                    torch.stack([0.5 - (qj**2 + qk**2), (qi * qj) - (qr * qk), (qi * qk) + (qr * qj)], -1),
                    torch.stack([(qi * qj) + (qr * qk), 0.5 - (qi**2 + qk**2), (qj * qk) - (qr * qi)], -1),
                    torch.stack([(qi * qk) - (qr * qj), (qj * qk) + (qr * qi), 0.5 - (qi**2 + qj**2)], -1),
                ],
                -1,
            ).mT
            * 2
        )

        return Rs.squeeze(1)

    def forward(self, us, vs, zs, ndx=None):
        """
        us (batch, num_pts)
        vs (batch, num_pts)
        zs (batch, num_pts)

        returns points_3d (batch, num_points, 3)
        """
        if ndx is None:
            points_backprojected = self.backproject(us, vs, zs).permute(0, 2, 1) # (batch, 3, num_points)
            points_3d = (self.Rs @ points_backprojected) + self.ts[...,None]  # (batch, 3, num_points)
            points_3d = points_3d.permute(0, 2, 1)  # (batch, num_points, 3)
        
        else:
            points_backprojected = self.backproject(us, vs, zs, ndx).permute(1, 0) # (3, num_points)
            points_3d = (self.Rs[ndx] @ points_backprojected) + self.ts[ndx,...,None]  # (3, num_points)
            points_3d =  points_3d.permute(1, 0) # (num_points, 3)

        return points_3d

    def backproject(self, us, vs, zs, ndx=None):
        """
        us (batch, num_pts)
        vs (batch, num_pts)
        zs (batch, num_pts)

        or

        ndx = int
        us (num_pts)
        vs (num_pts)
        zs (num_pts)

        returns points_backprojected (batch, num_points, 3)
        """
        assert us.shape == vs.shape and vs.shape == zs.shape, f"us ({us.shape}), vs ({vs.shape}), zs ({zs.shape}) must all be same shape"
        if ndx is None:
            points_intrinsics = torch.stack([us / self.fxs[...,None], vs / self.fys[...,None]], -1) # (batch, num_points, 2)
            points_homogenous = nn.functional.pad(points_intrinsics, (0, 1, 0, 0), mode="constant", value=1) # (batch, num_points, 3)
            points_backprojected = points_homogenous * zs[...,None] # (batch, num_points, 3)
        else:
            points_intrinsics = torch.stack([us / self.fxs[ndx], vs / self.fys[ndx]], -1) # (num_points, 2)
            points_homogenous = nn.functional.pad(points_intrinsics, (0, 1, 0, 0), mode="constant", value=1) # (num_points, 3)
            points_backprojected = points_homogenous * zs[...,None] # (num_points, 3)

        return points_backprojected
