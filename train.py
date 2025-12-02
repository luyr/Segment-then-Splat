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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up or dataset.deform == False:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # random sample 5 objects to render
        object_masks = viewpoint_cam.object_masks
        default_masks = object_masks['default']
        middle_masks = object_masks['middle']
        small_masks = object_masks['small']
        selected_small_objects = []
        selected_middle_objects = []
        selected_default_objects = [] 
        if iteration < opt.stage1_iters: # only render small objects
            if iteration == 1:
                print("===== stage 1: only render small objects =====")
            if len(small_masks) != 0:
                selected_small_objects = torch.randint(0, len(small_masks), (dataset.num_sample_objects,))
        elif iteration >= opt.stage1_iters and iteration < opt.stage1_iters + opt.stage2_iters: # only render small and middle objects
            if iteration == 1 + opt.stage1_iters:
                print("===== stage 2: only render small and middle objects =====")
            if len(small_masks) != 0:
                selected_small_objects = torch.randint(0, len(small_masks), (dataset.num_sample_objects,))
            if len(middle_masks) != 0:
                selected_middle_objects = torch.randint(0, len(middle_masks), (dataset.num_sample_objects,))
        else: # render all objects
            if iteration == 1 + opt.stage1_iters + opt.stage2_iters:
                print("===== stage 3: render all objects =====")
            if len(small_masks) != 0:
                selected_small_objects = torch.randint(0, len(small_masks), (dataset.num_sample_objects,))
            if len(middle_masks) != 0:
                selected_middle_objects = torch.randint(0, len(middle_masks), (dataset.num_sample_objects,))
            if len(default_masks) != 0:
                selected_default_objects = torch.randint(0, len(default_masks), (dataset.num_sample_objects,))
                
        
        for obj_id in selected_small_objects:
            # print("Rendering small object {}".format(obj_id))
            try:
                if small_masks[obj_id].sum() == 0: # skip empty mask
                    continue
                rendered_obj = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, obj_level="small", obj_id=obj_id)["render"]
                small_mask = small_masks[obj_id].to('cuda')
                # calculate IoU between rendered_obj and small_mask, first binarize the rendered_obj
                if iteration > opt.partial_mask_iter:
                    with torch.no_grad():
                        rendered_obj_mask = (rendered_obj > 0).bool()
                        iou = (rendered_obj_mask * small_mask).sum() / (rendered_obj_mask + small_mask).sum()
                        # print(rendered_obj_mask.dtype, small_mask.dtype)
                        if iou < opt.partial_mask_iou:
                            continue
                loss += (1.0 - opt.lambda_dssim) * l1_loss(rendered_obj, gt_image * small_mask) + \
                    opt.lambda_dssim * (1.0 - ssim(rendered_obj, gt_image * small_mask))
            except AssertionError:
                # print(f"small object {obj_id} has zero points in the scene, skip")
                pass
        for obj_id in selected_middle_objects:
            # print("Rendering middle object {}".format(obj_id))
            try:
                if middle_masks[obj_id].sum() == 0: # skip empty mask
                    continue
                rendered_obj = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, obj_level="middle", obj_id=obj_id)["render"]
                middle_mask = middle_masks[obj_id].to('cuda')
                # calculate IoU between rendered_obj and middle_mask, first binarize the rendered_obj
                if iteration > opt.partial_mask_iter:
                    with torch.no_grad():
                        rendered_obj_mask = (rendered_obj > 0).bool()
                        iou = (rendered_obj_mask * middle_mask).sum() / (rendered_obj_mask + middle_mask).sum()
                        # print(rendered_obj_mask.dtype, middle_mask.dtype)
                        if iou < opt.partial_mask_iou:
                            continue
                loss += (1.0 - opt.lambda_dssim) * l1_loss(rendered_obj, gt_image * middle_mask) + \
                    opt.lambda_dssim * (1.0 - ssim(rendered_obj, gt_image * middle_mask))
            except AssertionError:
                # print(f"middle object {obj_id} has zero points in the scene, skip")
                pass
        for obj_id in selected_default_objects:
            # print("Rendering default object {}".format(obj_id))
            try:
                if default_masks[obj_id].sum() == 0: # skip empty mask
                    continue
                rendered_obj = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, obj_level="default", obj_id=obj_id)["render"]
                default_mask = default_masks[obj_id].to('cuda')
                # calculate IoU between rendered_obj and default_mask, first binarize the rendered_obj
                if iteration > opt.partial_mask_iter:
                    with torch.no_grad():
                        rendered_obj_mask = (rendered_obj > 0).bool()
                        iou = (rendered_obj_mask * default_mask).sum() / (rendered_obj_mask + default_mask).sum()
                        # print(rendered_obj_mask.dtype, default_mask.dtype)
                        if iou < opt.partial_mask_iou:
                            continue
                loss += (1.0 - opt.lambda_dssim) * l1_loss(rendered_obj, gt_image * default_mask) + \
                    opt.lambda_dssim * (1.0 - ssim(rendered_obj, gt_image * default_mask))
            except AssertionError:
                print(f"default object {obj_id} has zero points in the scene, skip")
                pass
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof, dataset.deform)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                torch.save(gaussians.default_object_id, os.path.join(args.model_path, "default_object_id_{}.pth".format(iteration)))
                torch.save(gaussians.middle_object_id, os.path.join(args.model_path, "middle_object_id_{}.pth".format(iteration)))
                torch.save(gaussians.small_object_id, os.path.join(args.model_path, "small_object_id_{}.pth".format(iteration)))

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, is_deform=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    if is_deform:
                        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    else:
                        d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 70001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
