#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None, obj_level=None, obj_id=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # generate object_mask corresponding to the object id
    if obj_id is not None:
        if obj_level == "small":
            object_mask = pc.small_object_id == obj_id
        elif obj_level == "middle":
            object_mask = pc.middle_object_id == obj_id
        elif obj_level == "default":
            object_mask = pc.default_object_id == obj_id
    else:
        object_mask = torch.ones_like(pc.default_object_id, dtype=torch.float32, device="cuda").bool()
    
    # if object_mask.sum() == 0:  
    #     print(f"{obj_level} object {obj_id} has zero points in the scene")
    assert object_mask.sum() > 0
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz[object_mask], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz[object_mask], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz[object_mask]
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz[object_mask], to_homogenous(pc.get_xyz[object_mask]).unsqueeze(-1)).squeeze(-1))
    else:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz[object_mask]
        else:
            means3D = pc.get_xyz[object_mask] + d_xyz[object_mask]
    opacity = pc.get_opacity[object_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        print("Not implemented for single object rendering !!!")
        assert 0
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if torch.is_tensor(d_xyz) is False:
            scales = pc.get_scaling[object_mask] + d_scaling
            rotations = pc.get_rotation[object_mask] + d_rotation
        else:
            scales = pc.get_scaling[object_mask] + d_scaling[object_mask]
            rotations = pc.get_rotation[object_mask] + d_rotation[object_mask]
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            print("Not implemented for single object rendering !!!")
            assert 0
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features[object_mask]
    else:
        print("Not implemented for single object rendering !!!")
        assert 0
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth}
