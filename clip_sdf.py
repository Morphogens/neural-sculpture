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
# import wandb
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
        config: SimpleNamespace,
        sdf_grid_res_list: List[int] = [8, 16, 24, 32, 40, 48, 56, 64],
        out_img_width=256,
        out_img_height=256,
    ):
        self.config = config
        self.sdf_grid_res_list = sdf_grid_res_list
        self.out_img_width = out_img_width
        self.out_img_height = out_img_height

        self.bounding_box_min_x = -2.
        self.bounding_box_min_y = -2.
        self.bounding_box_min_z = -2.
        self.bounding_box_max_x = 2
        self.bounding_box_max_y = 2
        self.bounding_box_max_z = 2

        self.results_dir = None

        self.grid_res_x = self.grid_res_y = self.grid_res_z = None
        self.voxel_size = None

        self.update_res(self.sdf_grid_res_list[0])
        self.initialize_optim()

        # wandb.init(
        #     project='neural-sculpture',
        #     entity='viccpoes',
        #     config=vars(self.config),
        # )

    def update_res(
        self,
        grid_res: int,
    ):
        self.grid_res_x = self.grid_res_y = self.grid_res_z = grid_res
        self.voxel_size = Tensor([4. / (self.grid_res_x - 1)])

    def generate_initial_grid(self, ):
        grid = grid_construction_sphere_big(
            self.grid_res_x,
            self.bounding_box_min_x,
            self.bounding_box_max_x,
        )
        grid.requires_grad = True

        return grid

    def initialize_optim(self, ):
        self.grid = self.generate_initial_grid()
        self.optimizer = torch.optim.Adam(
            [self.grid],
            lr=self.config.learning_rate,
            eps=1e-8,
        )

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
                    ]),
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
        camera_angle,
    ):
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
            self.grid,
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

        # wandb.log({"image loss": image_loss})

        [grid_normal_x, grid_normal_y,
         grid_normal_z] = get_grid_normal(grid, self.voxel_size, grid_res_x,
                                          grid_res_y, grid_res_z)
        sdf_loss = torch.abs(torch.pow(grid_normal_x[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                    + torch.pow(grid_normal_y[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                    + torch.pow(grid_normal_z[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2) - 1).mean() #/ ((grid_res-1) * (grid_res-1) * (grid_res-1))

        return image_loss, sdf_loss

    def compute_losses(
        self,
        gen_img,
        prompt,
    ):
        image_loss, sdf_loss = self.loss_fn(
            gen_img,
            self.grid,
            prompt,
        )

        conv_input = (self.grid).unsqueeze(0).unsqueeze(0)
        conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0],
                                                 [0, 0, 0]],
                                                [[0, 1, 0], [1, -6, 1],
                                                 [0, 1, 0]],
                                                [[0, 0, 0], [0, 1, 0],
                                                 [0, 0, 0]]]]])

        conv_result = (F.conv3d(conv_input, conv_filter)**2)
        lp_loss = conv_result.mean()

        image_loss *= torch.prod(torch.tensor(gen_img.shape).detach().clone())
        sdf_loss *= torch.prod(torch.tensor(self.grid.shape).detach().clone())
        lp_loss *= torch.prod(torch.tensor(conv_result.shape).detach().clone())

        image_loss *= self.config.loss.image_loss_weight
        sdf_loss *= self.config.loss.sdf_loss_weight
        lp_loss *= self.config.loss.lp_loss_weight

        return image_loss, sdf_loss, lp_loss

    @staticmethod
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
        self,
        prompt: str,
        experiment_name: str = None,
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
        if experiment_name is None:
            experiment_name = "_".join(prompt.split(" "))

        self.results_dir = self.create_experiment_dir(
            experiment_name=experiment_name, )

        global_iter = 0
        for grid_res_idx, sdf_grid_res in enumerate(self.sdf_grid_res_list):
            increase_res = False

            sdf_grid_res_iter = 0
            while not increase_res:
                cam_view_loss = 0

                num_cameras = min(
                    self.config.camera.max_num_cameras,
                    self.config.camera.init_num_cameras *
                    (sdf_grid_res_iter + 1),
                )
                camera_angle_list = self.get_camera_angle_list(
                    num_cameras=num_cameras,
                    mapping_span=self.config.camera.mapping_span,
                    shuffle_order=self.config.camera.shuffle_order,
                    mapping_type=self.config.camera.mapping_type,
                )

                if sdf_grid_res_iter % 2 == 0:
                    camera_angle_list = camera_angle_list[::-1]

                step_size = int(np.ceil(self.config.batch_size / 2))
                for cam_view_idx in range(0, num_cameras, step_size):
                    init_loss = None
                    tolerance = self.config.init_tolerance / (grid_res_idx + 1)

                    cam_iter = 0
                    while True:

                        gen_img = self.generate_image(
                            camera_angle_list[cam_view_idx], )

                        image_loss, sdf_loss, lp_loss = self.compute_losses(
                            gen_img,
                            prompt,
                        )

                        cam_view_loss += image_loss + sdf_loss + lp_loss

                        if init_loss is not None:
                            if image_loss < init_loss - tolerance:
                                break

                        if cam_iter >= self.config.max_iters_per_cam:
                            break

                        if init_loss is None:
                            init_loss = image_loss

                        global_iter += 1

                        if global_iter % self.config.batch_size == 0:
                            cam_view_loss /= self.config.batch_size

                            self.optimizer.zero_grad()
                            cam_view_loss.backward()
                            self.optimizer.step()

                            # NOTE: clears jupyter notebook cell output
                            clear_output(wait=True)

                            writer.add_scalar(
                                f'loss-{sdf_grid_res}/image_loss',
                                image_loss.item(),
                                global_iter,
                            )
                            writer.add_scalar(
                                f'loss-{sdf_grid_res}/sdf_loss',
                                sdf_loss,
                                global_iter,
                            )
                            writer.add_scalar(
                                f'loss-{sdf_grid_res}/lp_loss',
                                lp_loss,
                                global_iter,
                            )
                            writer.add_scalar(
                                f'loss-{sdf_grid_res}/cam_view_loss',
                                cam_view_loss,
                                global_iter,
                            )

                            print("\n\n")
                            print("image loss: ", image_loss, "\n")
                            print("sdf loss: ", sdf_loss, "\n")
                            print("lp loss: ", lp_loss, "\n")
                            print("loss: ", cam_view_loss, "\n")
                            print("tolerance:", tolerance, "\n")
                            print("")
                            print("sdf grid res:", sdf_grid_res, " - "
                                  "iteration:", sdf_grid_res_iter, " - ",
                                  "cam view idx", cam_view_idx, " - ",
                                  "cam iters:", cam_iter + 1)

                            torchvision.utils.save_image(
                                gen_img.detach(),
                                "./" + self.results_dir + "/" + "final_cam_" +
                                str(sdf_grid_res).zfill(4) + "-" +
                                str(global_iter).zfill(4) + ".jpg",
                                nrow=8,
                                padding=2,
                                normalize=False,
                                range=None,
                                scale_each=False,
                                pad_value=0,
                            )

                            # NOTE: jupyter notebook display
                            image_initial_array = gen_img.detach().cpu().numpy(
                            ) * 255
                            display(
                                Image.fromarray(
                                    image_initial_array.astype(np.uint8)))

                            cam_view_loss = 0

                        cam_iter += 1

                    torch.save(
                        self.grid,
                        os.path.join(self.results_dir, "grid.pt"),
                    )

                if sdf_grid_res_iter >= self.config.iters_per_res:
                    increase_res = True

                sdf_grid_res_iter += 1

            # NOTE: do not update in the last iteration
            if grid_res_idx + 1 >= len(self.sdf_grid_res_list):
                break

            update_grid_res = self.sdf_grid_res_list[grid_res_idx + 1]
            self.update_res(grid_res=update_grid_res, )

            with torch.no_grad():
                # Update the sdf grid
                self.grid = torch.nn.functional.interpolate(
                    self.grid[None, None, :],
                    size=(update_grid_res, ) * 3,
                    mode='trilinear',
                )[0, 0, :]

            self.grid.requires_grad = True
            optim_state = self.optimizer.state_dict()['state']

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

            self.config.learning_rate /= 1.2
            self.optimizer = torch.optim.Adam(
                [self.grid],
                lr=self.config.learning_rate,
                eps=1e-8,
            )

            self.optimizer.state_dict()['state'] = optim_state

            grid_res_idx += 1

    def optimize_view(
        self,
        prompt,
        camera_angle,
    ):
        gen_img = self.generate_image(camera_angle, )

        image_loss, sdf_loss, lp_loss = self.compute_losses(
            gen_img,
            prompt,
        )

        cam_view_loss = image_loss + sdf_loss + lp_loss

        self.optimizer.zero_grad()
        cam_view_loss.backward()
        self.optimizer.step()

        return self.grid

    def generate_visualizations(self, ):
        vis_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        out_video_path = os.path.join(vis_dir, "generation.mp4")
        print(f"Generating {out_video_path}")

        cmd = ("ffmpeg -y "
               "-r 16 "
               f"-pattern_type glob -i '{self.results_dir}/*.jpg' "
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

        grid = torch.load(os.path.join(self.results_dir, "grid.pt"))
        self.update_res(grid.shape[0])
        for cam in range(num_cameras):
            gen_img = self.generate_image(camera_angle_list[cam], )

            torchvision.utils.save_image(
                gen_img.detach(),
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


if __name__ == "__main__":
    optim_config = SimpleNamespace(
        learning_rate=0.01,
        batch_size=1,
        init_tolerance=-0.5,
        iters_per_res=6,
        max_iters_per_cam=8,
        camera=SimpleNamespace(
            max_num_cameras=8,
            init_num_cameras=8,
            mapping_span=math.pi / 2,
            shuffle_order=False,
            mapping_type="linear",
        ),
        loss=SimpleNamespace(
            image_loss_weight=1 / 1000,
            sdf_loss_weight=1 / 1000,
            lp_loss_weight=1 / 1000,
        ),
    )

    sdf_optimizer = SDFOptimizer(config=optim_config, )

    sdf_optimizer.clip_sdf_optimization(
        prompt="3D bunny rabbit mesh rendered with maya zbrush",
        experiment_name="test",
    )