USE_WANDB = False

if USE_WANDB:
    import wandb
else:
    import wandb_stub as wandb

import os
import math
import glob
import subprocess
import random
from typing import *
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
from PIL import Image

from clip_loss import CLIPLoss
from grid_utils import grid_construction_sphere_big, get_grid_normal, grid_construction_sphere_small
from sdf_utils import generate_image

# NOTE: speed up
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

clip_loss = CLIPLoss()

writer = SummaryWriter()


class SDFOptimizer:
    def __init__(
        self,
        bounding_box_min_x=-2.,
        bounding_box_min_y=-2.,
        bounding_box_min_z=-2.,
        bounding_box_max_x=2.,
        bounding_box_max_y=2.,
        bounding_box_max_z=2.,
        out_img_width=256,
        out_img_height=256,
    ):
        self.bounding_box_min_x = bounding_box_min_x
        self.bounding_box_min_y = bounding_box_min_y
        self.bounding_box_min_z = bounding_box_min_z
        self.bounding_box_max_x = bounding_box_max_x
        self.bounding_box_max_y = bounding_box_max_y
        self.bounding_box_max_z = bounding_box_max_z
        self.out_img_width = out_img_width
        self.out_img_height = out_img_height

        self.grid_res_x = self.grid_res_y = self.grid_res_z = None
        self.voxel_size = None

    def update_res(
        self,
        grid_res: int,
    ):
        self.grid_res_x = self.grid_res_y = self.grid_res_z = grid_res
        self.voxel_size = Tensor([4. / (self.grid_res_x - 1)])

    def generate_initial_grid(self, ):
        grid_initial = grid_construction_sphere_big(
            self.grid_res_x,
            self.bounding_box_min_x,
            self.bounding_box_max_x,
        )
        grid_initial.requires_grad = True

        return grid_initial

    def get_camera_angle_list(
        self,
        num_cameras: int,
        mapping_span: float,
        shuffle_order: bool = False,
        mapping_type: str = "linear",
    ):
        if num_cameras == 1:
            camera_angle_list = [Tensor([0, 0, 5])]
        else:
            if mapping_type == "linear":
                # lin_space1 = list(np.linspace(0, 1, num_cameras)**2 * mapping_span/2)
                # lin_space2 = list(np.linspace(1, 0, num_cameras)**2 * mapping_span/2 + mapping_span/2)
                # camera_angle_list = [
                #     Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
                #     for a in lin_space1 + lin_space2
                # ]

                lin_space = list(np.linspace(0, 1, num_cameras) * mapping_span)
                camera_angle_list = [
                    Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
                    for a in lin_space
                ]

            elif mapping_type == "sdfdiff":
                camera_angle_list = [
                    Tensor([0, 0, 5]),  # 0
                    Tensor([0.1, 5, 0]),
                    Tensor([5, 0, 0]),
                    Tensor([0, 0, -5]),
                    Tensor([0.1, -5, 0]),
                    Tensor([-5, 0, 0]),  # 5
                    Tensor([5 / math.sqrt(2), 0, 5 / math.sqrt(2)]),
                    Tensor([5 / math.sqrt(2), 5 / math.sqrt(2), 0]),
                    Tensor([0, 5 / math.sqrt(2), 5 / math.sqrt(2)]),
                    Tensor([-5 / math.sqrt(2), 0, -5 / math.sqrt(2)]),
                    Tensor([-5 / math.sqrt(2), -5 / math.sqrt(2), 0]),  #10
                    Tensor([0, -5 / math.sqrt(2), -5 / math.sqrt(2)]),
                    Tensor([-5 / math.sqrt(2), 0, 5 / math.sqrt(2)]),
                    Tensor([-5 / math.sqrt(2), 5 / math.sqrt(2), 0]),
                    Tensor([0, -5 / math.sqrt(2), 5 / math.sqrt(2)]),
                    Tensor([5 / math.sqrt(2), 0, -5 / math.sqrt(2)]),
                    Tensor([5 / math.sqrt(2), -5 / math.sqrt(2), 0]),
                    Tensor([0, 5 / math.sqrt(2), -5 / math.sqrt(2)]),
                    Tensor(
                        [5 / math.sqrt(3), 5 / math.sqrt(3),
                         5 / math.sqrt(3)]),
                    Tensor([
                        5 / math.sqrt(3), 5 / math.sqrt(3), -5 / math.sqrt(3)
                    ]),
                    Tensor([
                        5 / math.sqrt(3), -5 / math.sqrt(3), 5 / math.sqrt(3)
                    ]),
                    Tensor([
                        -5 / math.sqrt(3), 5 / math.sqrt(3), 5 / math.sqrt(3)
                    ]),
                    Tensor([
                        -5 / math.sqrt(3), -5 / math.sqrt(3), 5 / math.sqrt(3)
                    ]),
                    Tensor([
                        -5 / math.sqrt(3), 5 / math.sqrt(3), -5 / math.sqrt(3)
                    ]),
                    Tensor([
                        5 / math.sqrt(3), -5 / math.sqrt(3), -5 / math.sqrt(3)
                    ]),
                    Tensor([
                        -5 / math.sqrt(3), -5 / math.sqrt(3), -5 / math.sqrt(3)
                    ])
                ]
                num_cameras = len(camera_angle_list)

                print(
                    f"WARNING: using `sdfdiff` angles. Num cameras is {num_cameras}"
                )

        if shuffle_order:
            random.shuffle(camera_angle_list)

        return camera_angle_list

    def generate_image(
        self,
        grid_initial,
        camera_angle,
    ):
        # for cam_iter in range(max_num_iters_per_camera):
        image = generate_image(
            self.bounding_box_min_x,
            self.bounding_box_min_y,
            self.bounding_box_min_z,
            self.bounding_box_max_x,
            self.bounding_box_max_y,
            self.bounding_box_max_z,
            self.voxel_size,
            self.grid_res_x,
            self.grid_res_y,
            self.grid_res_z,
            self.out_img_width,
            self.out_img_height,
            grid_initial,
            camera_angle,
        )

        return image

    def loss_fn(
        self,
        output,
        grid,
        prompt=None,
    ):
        grid_res_x = self.grid_res_x
        grid_res_y = self.grid_res_y
        grid_res_z = self.grid_res_z

        output = output[None, None].repeat(1, 3, 1, 1)  # CLIP expects RGB
        image_loss = clip_loss.compute(
            output,
            prompt,
        )

        wandb.log({"image loss": image_loss})

        [grid_normal_x, grid_normal_y,
         grid_normal_z] = get_grid_normal(grid, self.voxel_size, grid_res_x,
                                          grid_res_y, grid_res_z)
        sdf_loss = torch.abs(torch.pow(grid_normal_x[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                    + torch.pow(grid_normal_y[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                    + torch.pow(grid_normal_z[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2) - 1).mean() #/ ((grid_res-1) * (grid_res-1) * (grid_res-1))

        return image_loss, sdf_loss

    def compute_loss(
        self,
        prompt,
        image_initial,
        grid_initial,
    ):
        image_loss, sdf_loss = self.loss_fn(
            image_initial,
            grid_initial,
            prompt,
        )

        return image_loss, sdf_loss


def create_experiment_dir(experiment_name: str, ):
    out_dir = os.path.join(
        "./experiments",
        experiment_name,
    )

    num_out_gen_dirs = len(glob.glob(f"{out_dir}*"))
    if num_out_gen_dirs:
        results_dir = out_dir + f"-{num_out_gen_dirs}"
    else:
        results_dir = out_dir

    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def clip_sdf_optimization(
    prompt: str,
    optim_config: SimpleNamespace,
    experiment_name: str = None,
    sdf_grid_res_list: List[int] = [8, 16, 24, 32, 40, 48, 56, 64],
    on_update: Callable[[Tensor], None]=None
):
    """
    Optimize a 3D SDF grid given a prompt. The process is:
    1. Define and create experiment directory
    2. Set grid parameters
    3. Create initial SDF grid
    4. Create optimizer
    5. Optimize, i.e.:
        for each resolution in sdf_grid_res_list
            while the resolution is not increased
                get camera views
                for each camera view
                    generate image
                    compute loss
                    update optimizer
                    print results
                    compute stop criteria

            update resolution and optimizer

        Generate final visualizations
    """
    wandb.init(
        project='neural-sculpture',
        entity='viccpoes',
        config=vars(optim_config),
    )

    if experiment_name is None:
        experiment_name = "_".join(prompt.split(" "))

    results_dir = create_experiment_dir(experiment_name=experiment_name, )

    sdf_optimizer = SDFOptimizer()
    sdf_optimizer.update_res(sdf_grid_res_list[0])
    grid_initial = sdf_optimizer.generate_initial_grid()

    optimizer = torch.optim.Adam(
        [grid_initial],
        lr=optim_config.learning_rate,
        eps=1e-8,
    )

    global_iter = 0
    for grid_res_idx, sdf_grid_res in enumerate(sdf_grid_res_list):
        increase_res = False

        sdf_grid_res_iter = 0
        while not increase_res:
            cam_view_loss = 0

            num_cameras = min(
                optim_config.camera.max_num_cameras,
                optim_config.camera.init_num_cameras * (sdf_grid_res_iter + 1),
            )
            camera_angle_list = sdf_optimizer.get_camera_angle_list(
                num_cameras=num_cameras,
                mapping_span=optim_config.camera.mapping_span,
                shuffle_order=optim_config.camera.shuffle_order,
                mapping_type=optim_config.camera.mapping_type,
            )

            step_size = int(np.ceil(optim_config.batch_size / 2))
            for cam_view_idx in range(0, num_cameras, step_size):
                init_loss = None
                tolerance = optim_config.init_tolerance / (grid_res_idx + 1)

                cam_iter = 0
                while True:
                    cam_iter += 1

                    image_initial = sdf_optimizer.generate_image(
                        grid_initial,
                        camera_angle_list[cam_view_idx],
                    )

                    image_loss, sdf_loss = sdf_optimizer.compute_loss(
                        prompt,
                        image_initial,
                        grid_initial,
                    )

                    conv_input = (grid_initial).unsqueeze(0).unsqueeze(0)
                    conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0],
                                                             [0, 1, 0],
                                                             [0, 0, 0]],
                                                            [[0, 1, 0],
                                                             [1, -6, 1],
                                                             [0, 1, 0]],
                                                            [[0, 0, 0],
                                                             [0, 1, 0],
                                                             [0, 0, 0]]]]])

                    conv_result = (F.conv3d(conv_input, conv_filter)**2)
                    lp_loss = conv_result.mean()

                    image_loss *= torch.prod(
                        torch.tensor(image_initial.shape).detach().clone())
                    sdf_loss *= torch.prod(
                        torch.tensor(grid_initial.shape).detach().clone())
                    lp_loss *= torch.prod(
                        torch.tensor(conv_result.shape).detach().clone())

                    image_loss *= optim_config.loss.image_loss_weight
                    sdf_loss *= optim_config.loss.sdf_loss_weight
                    lp_loss *= optim_config.loss.lp_loss_weight

                    cam_view_loss += image_loss + sdf_loss + lp_loss

                    if init_loss is not None:
                        if image_loss < init_loss - tolerance:
                            break

                    if cam_iter >= optim_config.max_iters_per_cam:
                        break

                    if init_loss is None:
                        init_loss = image_loss

                    global_iter += 1

                    if global_iter % optim_config.batch_size == 0:
                        cam_view_loss /= optim_config.batch_size

                        optimizer.zero_grad()
                        cam_view_loss.backward()
                        optimizer.step()
                        if on_update:
                            on_update(grid_initial)

                        cam_view_loss = 0

                        # NOTE: clears jupyter notebook cell output
                        clear_output(wait=True)

                        writer.add_scalar(f'loss-{sdf_grid_res}/image_loss',
                                          image_loss.item(), global_iter)
                        writer.add_scalar(f'loss-{sdf_grid_res}/sdf_loss',
                                          sdf_loss, global_iter)
                        writer.add_scalar(f'loss-{sdf_grid_res}/lp_loss',
                                          lp_loss, global_iter)
                        writer.add_scalar(f'loss-{sdf_grid_res}/cam_view_loss',
                                          cam_view_loss, global_iter)

                        print("\n")
                        print("")
                        print("image loss: ", image_loss)
                        print("image weight: ",
                              torch.prod(torch.tensor(image_initial.shape)))
                        print("")
                        print("sdf loss: ", sdf_loss)
                        print("sdf weight: ",
                              torch.prod(torch.tensor(grid_initial.shape)))
                        print("")
                        print("lp loss: ", lp_loss)
                        print("lp weight: ",
                              torch.prod(torch.tensor(conv_input.shape)))
                        print("")
                        print("loss: ", cam_view_loss)
                        print("")
                        print("tolerance:", tolerance)
                        print("")
                        print("sdf grid res:", sdf_grid_res, " - "
                              "iteration:", sdf_grid_res_iter, " - ",
                              "cam view idx", cam_view_idx, " - ",
                              "cam iters:", cam_iter + 1)
                        print("")

                        torchvision.utils.save_image(
                            image_initial.detach(),
                            "./" + results_dir + "/" + "final_cam_" +
                            str(sdf_grid_res).zfill(4) + "-" +
                            str(global_iter).zfill(4) + ".jpg",
                            nrow=8,
                            padding=2,
                            normalize=False,
                            range=None,
                            scale_each=False,
                            pad_value=0)

                        # NOTE: jupyter notebook display
                        image_initial_array = image_initial.detach().cpu(
                        ).numpy() * 255
                        display(
                            Image.fromarray(
                                image_initial_array.astype(np.uint8)))

                torch.save(
                    grid_initial,
                    os.path.join(results_dir, "grid.pt"),
                )

            if sdf_grid_res_iter >= optim_config.iters_per_res:
                increase_res = True

            sdf_grid_res_iter += 1

        # NOTE: do not update in the last iteration
        if grid_res_idx + 1 >= len(sdf_grid_res_list):
            break

        update_grid_res = sdf_grid_res_list[grid_res_idx + 1]
        sdf_optimizer.update_res(grid_res=update_grid_res, )

        with torch.no_grad():
            # Update the sdf grid
            grid_initial = torch.nn.functional.interpolate(
                grid_initial[None, None, :],
                size=(update_grid_res, ) * 3,
                mode='trilinear',
            )[0, 0, :]

        grid_initial.requires_grad = True
        optim_state = optimizer.state_dict()['state']

        exp_avg = optim_state[0]['exp_avg'].clone()
        optim_state[0]['exp_avg'] = torch.nn.functional.interpolate(
            exp_avg[None, None, :],
            size=(update_grid_res, ) * 3,
            mode='trilinear',
        )[0, 0, :]

        exp_avg_sq = optim_state[0]['exp_avg_sq'].clone()
        optim_state[0]['exp_avg_sq'] = torch.nn.functional.interpolate(
            exp_avg_sq[None, None, :],
            size=(update_grid_res, ) * 3,
            mode='trilinear',
        )[0, 0, :]

        optim_config.learning_rate /= 1.1
        optimizer = torch.optim.Adam(
            [grid_initial],
            lr=optim_config.learning_rate,
            eps=1e-8,
        )

        optimizer.state_dict()['state'] = optim_state

        grid_res_idx += 1

    return results_dir


def generate_visualizations(results_dir):
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    out_video_path = os.path.join(vis_dir, "generation.mp4")
    print(f"Generating {out_video_path}")

    cmd = ("ffmpeg -y "
           "-r 16 "
           f"-pattern_type glob -i '{results_dir}/*.jpg' "
           "-vcodec libx264 "
           "-crf 25 "
           "-pix_fmt yuv420p "
           f"{out_video_path}")

    subprocess.check_call(cmd, shell=True)

    num_cameras = 128
    camera_angle_list = [
        torch.tensor([5 * math.cos(a), 0, 5 * math.sin(a)]).cuda()
        for a in np.linspace(0, 2 * math.pi, num_cameras)
    ]

    grid = torch.load(os.path.join(results_dir, "grid.pt"))
    sdf_optimizer = SDFOptimizer()
    sdf_optimizer.update_res(grid.shape[0])
    for cam in range(num_cameras):
        image_initial = sdf_optimizer.generate_image(
            grid,
            camera_angle_list[cam],
        )

        torchvision.utils.save_image(
            image_initial.detach(),
            os.path.join(vis_dir, f"{str(cam).zfill(3)}.jpg"),
            nrow=8,
            padding=2,
            normalize=False,
            range=None,
            scale_each=False,
            pad_value=0,
        )

    out_video_path = os.path.join(vis_dir, "visualization.mp4")
    print(f"Generating {out_video_path}")

    cmd = ("ffmpeg -y "
           "-r 16 "
           f"-pattern_type glob -i '{vis_dir}/*.jpg' "
           "-vcodec libx264 "
           "-crf 25 "
           "-pix_fmt yuv420p "
           f"{out_video_path}")

    subprocess.check_call(cmd, shell=True)

    print("\n----- END -----")


# class SDFCLIP:
#     def __init__(
#         self,
#         prompt: str = "a bunny rabbit mesh rendered with maya zbrush",
#         # prompt: str = "a cube mesh rendered with maya zbrush",
#         out_dir: str = "./results",
#         out_img_width: int = 256,
#         out_img_height: int = 256,
#         use_single_cam: bool = False,
#         print_jupyter: bool = False,
#         save_results: bool = True,
#     ):
#         self.prompt = prompt
#         self.out_img_width = out_img_width
#         self.out_img_height = out_img_height
#         self.use_single_cam = use_single_cam
#         self.print_jupyter = print_jupyter
#         self.save_results = save_results

#         self.out_dir = os.path.join(out_dir, '_'.join(self.prompt.split(" ")))

#         self.sdf_grid_res_list = [8, 16, 24, 32, 40, 48, 56, 64]
#         # self.sdf_grid_res_list = [8, 12, 16, 24, 32, 40]

#         self.clip_loss = CLIPLoss(
#             text_target=prompt,
#             device=device,
#         )

#     def loss_fn(
#         self,
#         output,
#         grid,
#         voxel_size,
#         grid_res_x,
#         grid_res_y,
#         grid_res_z,
#         width,
#         height,
#         prompt=None,
#     ):

#         output = output[None, None].repeat(1, 3, 1, 1)  # CLIP expects RGB
#         image_loss = self.clip_loss.compute(
#             output,
#             prompt,
#         )

#         [grid_normal_x, grid_normal_y,
#          grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x,
#                                           grid_res_y, grid_res_z)
#         sdf_loss = torch.abs(torch.pow(grid_normal_x[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
#                                     + torch.pow(grid_normal_y[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
#                                     + torch.pow(grid_normal_z[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2) - 1).mean() #/ ((grid_res-1) * (grid_res-1) * (grid_res-1))

#         return image_loss, sdf_loss

#     def get_camera_angle_list(
#         self,
#         num_cameras=1,
#         mapping_view=math.pi / 4,
#     ):
#         if num_cameras == 1:
#             camera_angle_list = [Tensor([0, 0, 5])]
#         else:
#             # lin_space1 = list(np.linspace(0, 1, num_cameras)**2 * mapping_view/2)
#             # lin_space2 = list(np.linspace(1, 0, num_cameras)**2 * mapping_view/2 + mapping_view/2)
#             # camera_angle_list = [
#             #     Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
#             #     for a in lin_space1 + lin_space2
#             # ]

#             lin_space = list(np.linspace(0, 1, num_cameras) * mapping_view)
#             camera_angle_list = [
#                 Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
#                 for a in lin_space
#             ]

#             random.shuffle(camera_angle_list)

#             # first_half_camera_list = camera_angle_list[0:int(num_cameras/1)]
#             # second_half_camera_list = camera_angle_list[int(num_cameras/2)::]
#             # camera_angle_list = first_half_camera_list + second_half_camera_list

#             # camera_angle_list = [
#             #     Tensor([0, 0, 5]),  # 0
#             #     Tensor([0.1, 5, 0]),
#             #     Tensor([5, 0, 0]),
#             #     Tensor([0, 0, -5]),
#             #     Tensor([0.1, -5, 0]),
#             #     Tensor([-5, 0, 0]),  # 5
#             #     Tensor([5 / math.sqrt(2), 0, 5 / math.sqrt(2)]),
#             #     Tensor([5 / math.sqrt(2), 5 / math.sqrt(2), 0]),
#             #     Tensor([0, 5 / math.sqrt(2), 5 / math.sqrt(2)]),
#             #     Tensor([-5 / math.sqrt(2), 0, -5 / math.sqrt(2)]),
#             #     Tensor([-5 / math.sqrt(2), -5 / math.sqrt(2), 0]),  #10
#             #     Tensor([0, -5 / math.sqrt(2), -5 / math.sqrt(2)]),
#             #     Tensor([-5 / math.sqrt(2), 0, 5 / math.sqrt(2)]),
#             #     Tensor([-5 / math.sqrt(2), 5 / math.sqrt(2), 0]),
#             #     Tensor([0, -5 / math.sqrt(2), 5 / math.sqrt(2)]),
#             #     Tensor([5 / math.sqrt(2), 0, -5 / math.sqrt(2)]),
#             #     Tensor([5 / math.sqrt(2), -5 / math.sqrt(2), 0]),
#             #     Tensor([0, 5 / math.sqrt(2), -5 / math.sqrt(2)]),
#             #     Tensor([5 / math.sqrt(3), 5 / math.sqrt(3), 5 / math.sqrt(3)]),
#             #     Tensor([5 / math.sqrt(3), 5 / math.sqrt(3),
#             #             -5 / math.sqrt(3)]),
#             #     Tensor([5 / math.sqrt(3), -5 / math.sqrt(3),
#             #             5 / math.sqrt(3)]),
#             #     Tensor([-5 / math.sqrt(3), 5 / math.sqrt(3),
#             #             5 / math.sqrt(3)]),
#             #     Tensor(
#             #         [-5 / math.sqrt(3), -5 / math.sqrt(3), 5 / math.sqrt(3)]),
#             #     Tensor(
#             #         [-5 / math.sqrt(3), 5 / math.sqrt(3), -5 / math.sqrt(3)]),
#             #     Tensor(
#             #         [5 / math.sqrt(3), -5 / math.sqrt(3), -5 / math.sqrt(3)]),
#             #     Tensor(
#             #         [-5 / math.sqrt(3), -5 / math.sqrt(3), -5 / math.sqrt(3)])
#             # ]
#             # num_cameras = len(camera_angle_list)

#         return camera_angle_list

#     def run(
#         self,
#         learning_rate: float = 0.006,
#         image_loss_weight: float = 1 / 1000,
#         sdf_loss_weight: float = 1 / 1000,
#         lp_loss_weight: float = 1 / 1000,
#         max_std_res_loss: float = 0.6,
#         num_std_res_samples: float = 4,
#         max_num_iters_per_camera: int = 1,
#         view_batch_size: int = 4,
#     ):
#         num_out_gen_dirs = len(glob.glob(f"{self.out_dir}*"))
#         if num_out_gen_dirs:
#             results_dir = self.out_dir + f"-{num_out_gen_dirs}"
#         else:
#             results_dir = self.out_dir

#         os.makedirs(results_dir, exist_ok=True)

#         bounding_box_min_x = -2.
#         bounding_box_min_y = -2.
#         bounding_box_min_z = -2.
#         bounding_box_max_x = 2.
#         bounding_box_max_y = 2.
#         bounding_box_max_z = 2.

#         grid_res_idx = 0
#         grid_res_x = grid_res_y = grid_res_z = self.sdf_grid_res_list[
#             grid_res_idx]
#         voxel_size = Tensor([4. / (grid_res_x - 1)])

#         grid_initial = grid_construction_sphere_big(
#             grid_res_x,
#             bounding_box_min_x,
#             bounding_box_max_x,
#         )
#         grid_initial.requires_grad = True

#         optimizer = torch.optim.Adam(
#             [grid_initial],
#             lr=learning_rate,
#             eps=1e-8,
#         )

#         # NOTE: from the paper
#         # -   We should focus on the views with large losses. Our
#         #     approach first calculates the average loss for all
#         #     the camera views from the result of the previous iteration.
#         # -   If a loss for a view is greater
#         #     than the average loss, then during the current iteration,
#         #     we update SDF until the loss for this view is less than the
#         #     average (with 20 max updates).
#         # -   For the other views, we update the SDF five times. If one
#         #     update increases the loss, then we switch to the next view
#         #     directly. We stop our optimization process when the loss is
#         #     smaller than a given tolerance or the step length is too small.

#         try:
#             global_iter = 0
#             for grid_res_idx, sdf_grid_res in enumerate(
#                     self.sdf_grid_res_list):

#                 std_res_loss = None
#                 avg_cam_view_loss = None
#                 increase_res = False

#                 sdf_grid_res_iter = 0
#                 while not increase_res:
#                     cam_view_loss = 0

#                     if self.use_single_cam:
#                         num_cameras = 1
#                         mapping_view = math.pi / 4
#                     else:
#                         num_cameras = min(64, 16 * (sdf_grid_res_iter + 1))
#                         # mapping_view = min(2*math.pi, (sdf_grid_res_iter + 1) * math.pi / 2)
#                         mapping_view = 2 * math.pi
#                         # prompt_prefix = [ 'the front view of '] * int(num_cameras/4) + ['a side view of '] * int(num_cameras/4) + ['the back view of '] * int(num_cameras/4) + ['a side view of '] * int(num_cameras/4)
#                         # max_num_cameras = 16
#                         # num_cameras = min(((grid_res_idx + 1) * 3), max_num_cameras)
#                         # mapping_view = min(2 * math.pi,
#                         #                    (grid_res_idx + 1) * math.pi / 2)

#                     print(f"NUM CAMERAS {num_cameras}")

#                     cam_view_loss_list = [None] * num_cameras
#                     camera_angle_list = self.get_camera_angle_list(
#                         num_cameras,
#                         mapping_view,
#                     )

#                     if sdf_grid_res_iter % 2 == 0:
#                         camera_angle_list = camera_angle_list[::-1]

#                     for cam_view_idx in range(
#                             0, num_cameras, int(np.ceil(view_batch_size / 2))):
#                         # prev_cam_loss = None

#                         init_loss = None
#                         tolerance = 0.01 / (grid_res_idx + 1)

#                         cam_iter = 0
#                         while True:
#                             cam_iter += 1

#                             # for cam_iter in range(max_num_iters_per_camera):
#                             image_initial = generate_image(
#                                 bounding_box_min_x,
#                                 bounding_box_min_y,
#                                 bounding_box_min_z,
#                                 bounding_box_max_x,
#                                 bounding_box_max_y,
#                                 bounding_box_max_z,
#                                 voxel_size,
#                                 grid_res_x,
#                                 grid_res_y,
#                                 grid_res_z,
#                                 self.out_img_width,
#                                 self.out_img_height,
#                                 grid_initial,
#                                 camera_angle_list[cam_view_idx],
#                             )

#                             image_loss, sdf_loss = self.loss_fn(
#                                 image_initial,
#                                 grid_initial,
#                                 voxel_size,
#                                 grid_res_x,
#                                 grid_res_y,
#                                 grid_res_z,
#                                 self.out_img_width,
#                                 self.out_img_height,
#                                 # prompt_prefix[cam_view_idx] + self.prompt,
#                             )

#                             image_loss *= torch.prod(
#                                 torch.tensor(
#                                     image_initial.shape).detach().clone())
#                             sdf_loss *= torch.prod(
#                                 torch.tensor(
#                                     grid_initial.shape).detach().clone())

#                             image_loss *= image_loss_weight
#                             sdf_loss *= sdf_loss_weight

#                             print("IMG LOSS", image_loss)
#                             print("INIT", init_loss)
#                             if init_loss is not None:
#                                 print("INIT - TOLERANCE",
#                                       init_loss - tolerance)

#                             if init_loss is not None:
#                                 if image_loss < init_loss - tolerance:
#                                     break
#                                 if cam_iter >= 32:
#                                     break

#                             if init_loss is None:
#                                 init_loss = image_loss

#                             conv_input = (
#                                 grid_initial).unsqueeze(0).unsqueeze(0)
#                             conv_filter = torch.cuda.FloatTensor(
#                                 [[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#                                    [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
#                                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])

#                             conv_result = (F.conv3d(conv_input,
#                                                     conv_filter)**2)
#                             lp_loss = conv_result.mean()
#                             lp_loss *= torch.prod(
#                                 torch.tensor(
#                                     conv_result.shape).detach().clone())
#                             lp_loss *= lp_loss_weight

#                             cam_view_loss += image_loss + sdf_loss + lp_loss

#                             global_iter += 1

#                             # if global_iter > 1 and global_iter % 128 == 0:
#                             #     increase_res = True
#                             #     break

#                             # if avg_cam_view_loss is not None:
#                             # if cam_iter >= min(max_num_iters_per_camera, 3) and cam_view_loss < avg_cam_view_loss:
#                             #     break

#                             # if prev_cam_loss is not None:
#                             #     if cam_iter >= 5 and prev_cam_loss > cam_view_loss:
#                             #         break

#                             cam_view_loss_list[cam_view_idx] = cam_view_loss

#                             # prev_cam_loss = cam_view_loss

#                             if global_iter > 1 and global_iter % view_batch_size == 0:
#                                 cam_view_loss /= view_batch_size

#                                 optimizer.zero_grad()
#                                 cam_view_loss.backward()
#                                 optimizer.step()

#                                 cam_view_loss = 0

#                                 # if global_iter % 4 == 0:
#                                 if self.print_jupyter:
#                                     clear_output(wait=True)

#                                 writer.add_scalar(
#                                     f'loss-{sdf_grid_res}/image_loss',
#                                     image_loss.item(), global_iter)
#                                 writer.add_scalar(
#                                     f'loss-{sdf_grid_res}/sdf_loss', sdf_loss,
#                                     global_iter)
#                                 writer.add_scalar(
#                                     f'loss-{sdf_grid_res}/lp_loss', lp_loss,
#                                     global_iter)
#                                 writer.add_scalar(
#                                     f'loss-{sdf_grid_res}/cam_view_loss',
#                                     cam_view_loss, global_iter)

#                                 if std_res_loss is None:
#                                     writer.add_scalar(
#                                         f'metrics-{sdf_grid_res}/std_res_loss',
#                                         -1, global_iter)
#                                 else:
#                                     writer.add_scalar(
#                                         f'metrics-{sdf_grid_res}/std_ress_loss',
#                                         std_res_loss, global_iter)
#                                 if std_res_loss is None:
#                                     writer.add_scalar(
#                                         f'metrics-{sdf_grid_res}/avg_cam_view_loss',
#                                         -1, global_iter)
#                                 else:
#                                     writer.add_scalar(
#                                         f'metrics-{sdf_grid_res}/avg_cam_view_loss',
#                                         avg_cam_view_loss, global_iter)

#                                 print("\n")
#                                 print("")
#                                 print("image loss: ", image_loss)
#                                 print(
#                                     "image weight: ",
#                                     torch.prod(
#                                         torch.tensor(image_initial.shape)))
#                                 print("")
#                                 print("sdf loss: ", sdf_loss)
#                                 print(
#                                     "sdf weight: ",
#                                     torch.prod(torch.tensor(
#                                         grid_initial.shape)))
#                                 print("")
#                                 print("lp loss: ", lp_loss)
#                                 print(
#                                     "lp weight: ",
#                                     torch.prod(torch.tensor(conv_input.shape)))
#                                 print("")
#                                 print("loss: ", cam_view_loss)
#                                 print("")
#                                 print("STD:", std_res_loss)
#                                 print("")
#                                 print("AVG:", avg_cam_view_loss)
#                                 print("")
#                                 print("sdf grid res:", sdf_grid_res, " - "
#                                       "iteration:", sdf_grid_res_iter, " - ",
#                                       "cam view idx", cam_view_idx, " - ",
#                                       "cam iters:", cam_iter + 1)
#                                 print("")

#                                 if self.save_results:
#                                     torchvision.utils.save_image(
#                                         image_initial.detach(),
#                                         "./" + results_dir + "/" +
#                                         "final_cam_" +
#                                         str(sdf_grid_res).zfill(4) + "-" +
#                                         str(global_iter).zfill(4) + ".jpg",
#                                         nrow=8,
#                                         padding=2,
#                                         normalize=False,
#                                         range=None,
#                                         scale_each=False,
#                                         pad_value=0)

#                                 if self.print_jupyter:
#                                     image_initial_array = image_initial.detach(
#                                     ).cpu().numpy() * 255
#                                     display(
#                                         Image.fromarray(
#                                             image_initial_array.astype(
#                                                 np.uint8)))

#                     # camera_angle_list = camera_angle_list[::-1]

#                     # avg_cam_view_loss = torch.tensor(cam_view_loss_list).mean()

#                     if sdf_grid_res_iter >= 5:
#                         increase_res = True

#                     sdf_grid_res_iter += 1

#                 if grid_res_idx + 1 >= len(self.sdf_grid_res_list):
#                     break

#                 grid_res_update_x = grid_res_update_y = grid_res_update_z = self.sdf_grid_res_list[
#                     grid_res_idx + 1]
#                 voxel_size_update = (bounding_box_max_x - bounding_box_min_x
#                                      ) / (grid_res_update_x - 1)
#                 # linear_space_x = torch.linspace(0, grid_res_update_x - 1,
#                 #                                 grid_res_update_x)
#                 # linear_space_y = torch.linspace(0, grid_res_update_y - 1,
#                 #                                 grid_res_update_y)
#                 # linear_space_z = torch.linspace(0, grid_res_update_z - 1,
#                 #                                 grid_res_update_z)
#                 # first_loop = linear_space_x.repeat(
#                 #     grid_res_update_y * grid_res_update_z,
#                 #     1).t().contiguous().view(-1).unsqueeze_(1)
#                 # second_loop = linear_space_y.repeat(
#                 #     grid_res_update_z,
#                 #     grid_res_update_x).t().contiguous().view(-1).unsqueeze_(1)
#                 # third_loop = linear_space_z.repeat(grid_res_update_x *
#                 #                                    grid_res_update_y).unsqueeze_(1)
#                 # loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()
#                 # min_x = Tensor([bounding_box_min_x]).repeat(
#                 #     grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
#                 # min_y = Tensor([bounding_box_min_y]).repeat(
#                 #     grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
#                 # min_z = Tensor([bounding_box_min_z]).repeat(
#                 #     grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
#                 # bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1)

#                 # # Get the position of the grid points in the refined grid
#                 # points = bounding_min_matrix + voxel_size_update * loop
#                 # voxel_min_point_index_x = torch.floor(
#                 #     (points[:, 0].unsqueeze_(1) - min_x) /
#                 #     voxel_size).clamp(max=grid_res_x - 2)
#                 # voxel_min_point_index_y = torch.floor(
#                 #     (points[:, 1].unsqueeze_(1) - min_y) /
#                 #     voxel_size).clamp(max=grid_res_y - 2)
#                 # voxel_min_point_index_z = torch.floor(
#                 #     (points[:, 2].unsqueeze_(1) - min_z) /
#                 #     voxel_size).clamp(max=grid_res_z - 2)
#                 # voxel_min_point_index = torch.cat(
#                 #     (voxel_min_point_index_x, voxel_min_point_index_y,
#                 #      voxel_min_point_index_z), 1)
#                 # voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

#                 # # Compute the sdf value of the grid points in the refined grid
#                 # grid_initial_update = calculate_sdf_value(
#                 #     grid_initial, points, voxel_min_point, voxel_min_point_index,
#                 #     voxel_size, grid_res_x, grid_res_y,
#                 #     grid_res_z).view(grid_res_update_x, grid_res_update_y,
#                 #                      grid_res_update_z)

#                 # Update the grid resolution for the refined sdf grid
#                 grid_res_x = grid_res_update_x
#                 grid_res_y = grid_res_update_y
#                 grid_res_z = grid_res_update_z

#                 # Update the voxel size for the refined sdf grid
#                 voxel_size = voxel_size_update

#                 with torch.no_grad():
#                     # Update the sdf grid
#                     grid_initial = torch.nn.functional.interpolate(
#                         grid_initial[None, None, :],
#                         size=(grid_res_x, grid_res_y, grid_res_z),
#                         mode='trilinear',
#                     )[0, 0, :]

#                 grid_initial.requires_grad = True
#                 optim_state = optimizer.state_dict()['state']

#                 exp_avg = optim_state[0]['exp_avg'].clone()
#                 optim_state[0]['exp_avg'] = torch.nn.functional.interpolate(
#                     exp_avg[None, None, :],
#                     size=(grid_res_x, grid_res_y, grid_res_z),
#                     mode='trilinear',
#                 )[0, 0, :]

#                 exp_avg_sq = optim_state[0]['exp_avg_sq'].clone()
#                 optim_state[0]['exp_avg_sq'] = torch.nn.functional.interpolate(
#                     exp_avg_sq[None, None, :],
#                     size=(grid_res_x, grid_res_y, grid_res_z),
#                     mode='trilinear',
#                 )[0, 0, :]

#                 learning_rate /= 1.1
#                 optimizer = torch.optim.Adam(
#                     [grid_initial],
#                     lr=learning_rate,
#                     eps=1e-8,
#                 )

#                 optimizer.state_dict()['state'] = optim_state

#                 # Double the size of the image
#                 if self.out_img_width < 256:
#                     self.out_img_width = int(self.out_img_width * 2)
#                     self.out_img_height = int(self.out_img_height * 2)

#                 grid_res_idx += 1

#         except Exception as e:
#             print(e)

#         self.vis_dir = os.path.join(results_dir, "visualizations")
#         os.makedirs(self.vis_dir, exist_ok=True)

#         out_video_path = os.path.join(self.vis_dir, "generation.mp4")
#         print(f"Generating {out_video_path}")

#         cmd = ("ffmpeg -y "
#                "-r 16 "
#                f"-pattern_type glob -i '{results_dir}/*.jpg' "
#                "-vcodec libx264 "
#                "-crf 25 "
#                "-pix_fmt yuv420p "
#                f"{out_video_path}")

#         subprocess.check_call(cmd, shell=True)

#         num_cameras = 128
#         camera_angle_list = [
#             torch.tensor([5 * math.cos(a), 0, 5 * math.sin(a)]).cuda()
#             for a in np.linspace(0, 2 * math.pi, num_cameras)
#         ]

#         out_img_width = 512
#         out_img_height = 512

#         for cam in range(num_cameras):
#             image_initial = generate_image(
#                 bounding_box_min_x,
#                 bounding_box_min_y,
#                 bounding_box_min_z,
#                 bounding_box_max_x,
#                 bounding_box_max_y,
#                 bounding_box_max_z,
#                 voxel_size,
#                 grid_res_x,
#                 grid_res_y,
#                 grid_res_z,
#                 out_img_width,
#                 out_img_height,
#                 grid_initial,
#                 camera_angle_list[cam],
#             )

#             torchvision.utils.save_image(
#                 image_initial.detach(),
#                 os.path.join(self.vis_dir, f"{str(cam).zfill(3)}.jpg"),
#                 nrow=8,
#                 padding=2,
#                 normalize=False,
#                 range=None,
#                 scale_each=False,
#                 pad_value=0,
#             )

#         out_video_path = os.path.join(self.vis_dir, "visualization.mp4")
#         print(f"Generating {out_video_path}")

#         cmd = ("ffmpeg -y "
#                "-r 16 "
#                f"-pattern_type glob -i '{self.vis_dir}/*.jpg' "
#                "-vcodec libx264 "
#                "-crf 25 "
#                "-pix_fmt yuv420p "
#                f"{out_video_path}")

#         subprocess.check_call(cmd, shell=True)

#         print("\n----- END -----")

#         return grid_initial

if __name__ == "__main__":
    optim_config = SimpleNamespace(
        learning_rate=0.01,
        batch_size=1,
        init_tolerance=0.1,
        iters_per_res=6,
        max_iters_per_cam=32,
        camera=SimpleNamespace(
            max_num_cameras=64,
            init_num_cameras=16,
            mapping_span=2 * math.pi,
            shuffle_order=False,
            mapping_type="linear",
        ),
        loss=SimpleNamespace(
            image_loss_weight=1 / 1000,
            sdf_loss_weight=1 / 1000,
            lp_loss_weight=1 / 1000,
        ),
    )

    clip_sdf_optimization(
        prompt="3D bunny rabbit mesh rendered with maya zbrush",
        optim_config=optim_config,
        experiment_name="test",
        sdf_grid_res_list=[8, 16, 24, 32, 40, 48, 56, 64],
    )