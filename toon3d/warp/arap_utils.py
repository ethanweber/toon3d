import torch

def batch_rigid_transform(A: torch.Tensor, B: torch.Tensor):
    """
    Calculates Rigid Transformation from A -> B

    Args:
        A (torch.Tensor): batch of N shapes with P points (N, 2, P)
        B (torch.Tensor): batch of N shapes with P points (N, 2, P)

    Returns:
        tuple: A tuple containing two elements:
            - R (torch.Tensor): rotation matrix (N, 2, 2)
            - t (torch.Tensor): translation vector (N, 2, 1).
    """
    assert A.shape == B.shape

    # find mean column wise
    centroid_A = torch.mean(A, axis=-1, keepdim=True)
    centroid_B = torch.mean(B, axis=-1, keepdim=True)

    H = (A - centroid_A) @ (B - centroid_B).mT

    # find rotation
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.mT @ U.mT
    U[R.det() < 0,...,1] *= -1 # fix flip
    R = Vt.mT @ U.mT

    t = -R @ centroid_A + centroid_B
    return R, t

def signed_areas(tri_verts: torch.Tensor):
    """
    Find the signed areas of a batch of triangles using the wedge product

    Given a batch of triangle coordiantes, each entry with:
    [[x0, x1, x2],
     [y0, y1, y2]]

    We find its signed area by first considering two vectors that form the triangle
    BA = [[x1 - x0], [y1 - y0]]
    CA = [[x2 - x0], [y2 - y0]]

    and then taking their wedge product to find the signed area
    area = BA ∧ CA = (BA.x * CA.y) - (CA.x * BA.y) = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    Args:
        tri_verts (torch.Tensor): a batch of N triangle vertex coordinates on a 2D plane (N, 2, 3)

    Returns:
        areas (torch.Tensor): batch of signed areas of triangles (N)
    """
    x0 = tri_verts[...,0,0]
    x1 = tri_verts[...,0,1]
    x2 = tri_verts[...,0,2]

    y0 = tri_verts[...,1,0]
    y1 = tri_verts[...,1,1]
    y2 = tri_verts[...,1,2]

    areas = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    return areas

def calc_area_loss(tri_verts: torch.Tensor, min_area: int = 50):
    """Calculates the area loss of a batch of vertices

    The area loss for each triangle boils down to max(0, min_area - area), 

    Args:
        tri_verts (torch.Tensor): a batch of N triangle vertex coordinates on a 2D plane (N, 2, 3)
        min_area (int, optional): Minimum area a triangle can be before loss is applied. Defaults to 50.

    Returns:
        torch.Tensor: Sum over batch of area losses (1)
    """
    areas = signed_areas(tri_verts)
    areas_loss = torch.sum(torch.relu(min_area - areas).abs())
    return areas_loss

def face_verts_arap_loss(stable_face_verts: torch.Tensor, transformed_face_verts: torch.Tensor):
    """Calculates As-Rigid-As-Possible (ARAP) loss

    Given a set of original vertices (stable_face_verts), 
    we find the rigid transformation (rigid_face_verts) that most closely corresponds to their transformation (transformed_face_verts)
    we then take the L2 distance between the rigid_face_verts and transformed_face_verts

    Args:
        stable_face_verts (torch.Tensor): a batch of N triangle vertex coordinates on a 2D plane (N, 2, 3)
        transformed_face_verts (torch.Tensor): a batch of N triangle vertex coordinates on a 2D plane (N, 2, 3)

    Returns:
        torch.Tensor: ARAP loss (1)
    """
    
    Rs, ts = batch_rigid_transform(stable_face_verts, transformed_face_verts)
    rigid_face_verts = (Rs @ stable_face_verts + ts).detach()

    return torch.mean(((transformed_face_verts - rigid_face_verts) ** 2))