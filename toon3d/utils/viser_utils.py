import torch
import numpy as np

import viser

def view_pcs_cameras(server,
                     images, # images
                     cameras,
                     dense_points_3d, # dense pc
                     images_colors, 
                     sparse_points_3d, # sparse pc
                     point_colors, 
                     mesh_verts_list, # mesh
                     simplices_list,
                     mesh_colors,
                     visible=True,
                     prefix="data"):

    heights = torch.tensor([image.shape[0] for image in images])

    gui_point_size = server.add_gui_slider(
        f"{prefix} point size", min=0.001, max=1.0, step=1000, initial_value=0.01
    )

    def draw() -> None:
        # from our coords to viser coords X, Y, Z -> -Z, X, -Y
        flip = torch.tensor([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).float()

        server.add_frame(f"{prefix}/mesh", show_axes=False, visible=False)
        server.add_frame(f"{prefix}/dense-pc", show_axes=False, visible=visible)
        server.add_frame(f"{prefix}/sparse-pc", show_axes=False, visible=visible)
        server.add_frame(f"{prefix}/cameras", show_axes=False, visible=visible)

        for ndx in range(len(images)):
            server.add_mesh_simple(
                f"{prefix}/mesh/img-{ndx}/fill",
                vertices=5 * (flip @ mesh_verts_list[ndx].cpu().T).T.detach().numpy(),
                faces=simplices_list[ndx].cpu().detach().numpy(),
                color=mesh_colors[ndx],
                wxyz=viser.transforms.SO3.from_x_radians(0).wxyz,
                opacity=0.4,
                flat_shading=True,
                side='double',
            )

            #wireframe
            server.add_mesh_simple(
                f"{prefix}/mesh/img-{ndx}/wireframe",
                vertices=5 * (flip @ mesh_verts_list[ndx].cpu().T).T.detach().numpy(),
                faces=simplices_list[ndx].cpu().detach().numpy(),
                wireframe=True,
                color=(0, 0, 0),
                wxyz=viser.transforms.SO3.from_x_radians(0).wxyz,
            )

            server.add_point_cloud(
                f"{prefix}/dense-pc/img-{ndx}",
                colors=images_colors[ndx].detach().numpy(),
                points=5 * (flip @ dense_points_3d[ndx].cpu().T).T.detach().numpy(),
                point_size=gui_point_size.value,
                point_shape="circle"
            )

            server.add_point_cloud(
                f"{prefix}/sparse-pc/img-{ndx}",
                colors=point_colors[ndx].cpu().detach().numpy(),
                points=5 * (flip @ sparse_points_3d[ndx].cpu().T).T.detach().numpy(),
                point_size=0.2,
                point_shape="sparkle",
            )

            server.add_camera_frustum(
                f"{prefix}/cameras/img-{ndx}",
                fov=np.arctan2(heights[ndx] / 2, cameras.fxs[ndx].item()),
                aspect=cameras.fys[ndx].item() / cameras.fxs[ndx].item(),
                scale=0.1,
                wxyz=viser.transforms.SO3.from_matrix(flip @ cameras.Rs[ndx].cpu().detach().numpy()).wxyz,
                position=5 * ((flip @ cameras.ts[ndx].cpu().flatten().detach().numpy())),
                image=images[ndx].numpy(),
            )

    gui_point_size.on_update(lambda _: draw())
    draw()