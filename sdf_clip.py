import os
import math
import glob
import subprocess
from typing import *

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from IPython.display import display, clear_output
from PIL import Image

from clip_loss import CLIPLoss
from grid_utils import grid_construction_sphere_big, get_grid_normal
from sdf_utils import calculate_sdf_value, compute_intersection_pos, generate_image

# NOTE: speed up
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor


class SDFCLIP:
    def __init__(
        self,
        prompt: str = "a bunny rabbit mesh rendered with maya zbrush",
        out_dir: str = "./results",
        out_img_width: int = 256,
        out_img_height: int = 256,
        use_single_cam: bool = True,
        print_jupyter: bool = False,
        save_results: bool = True,
    ):
        self.prompt = prompt
        self.out_img_width = out_img_width
        self.out_img_height = out_img_height
        self.use_single_cam = use_single_cam
        self.print_jupyter = print_jupyter
        self.save_results = save_results

        self.out_dir = os.path.join(out_dir, '_'.join(self.prompt.split(" ")))

        self.sdf_grid_res_list = [8, 16, 24, 32, 40, 48, 56, 64]

        self.clip_loss = CLIPLoss(
            text_target=prompt,
            device=device,
        )

    def loss_fn(
        self,
        output,
        target,
        grid,
        voxel_size,
        grid_res_x,
        grid_res_y,
        grid_res_z,
        width,
        height,
    ):

        output = output[None, None].repeat(1, 3, 1, 1)  # CLIP expects RGB
        image_loss = self.clip_loss.compute(output)

        [grid_normal_x, grid_normal_y,
         grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x,
                                          grid_res_y, grid_res_z)
        sdf_loss = torch.abs(torch.pow(grid_normal_x[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                    + torch.pow(grid_normal_y[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                    + torch.pow(grid_normal_z[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2) - 1).mean() #/ ((grid_res-1) * (grid_res-1) * (grid_res-1))

        return image_loss, sdf_loss

    def run(
        self,
        learning_rate: float = 0.01,
        image_loss_weight: float = 1 / 1000,
        sdf_loss_weight: float = 1 / 1000,
        lp_loss_weight: float = 1 / 1000,
        max_std_res_loss: float = 0.8,
        num_std_res_samples: float = 3,
        max_num_iters_per_camera: int = 20,
    ):
        num_out_gen_dirs = len(glob.glob(f"{self.out_dir}*"))
        if num_out_gen_dirs:
            results_dir = self.out_dir + f"-{num_out_gen_dirs}"

        os.makedirs(results_dir, exist_ok=True)

        bounding_box_min_x = -2.
        bounding_box_min_y = -2.
        bounding_box_min_z = -2.
        bounding_box_max_x = 2.
        bounding_box_max_y = 2.
        bounding_box_max_z = 2.

        grid_res_idx = 0
        grid_res_x = grid_res_y = grid_res_z = self.sdf_grid_res_list[
            grid_res_idx]
        voxel_size = Tensor([4. / (grid_res_x - 1)])

        grid_initial = grid_construction_sphere_big(
            grid_res_x,
            bounding_box_min_x,
            bounding_box_max_x,
        )
        grid_initial.requires_grad = True

        if self.use_single_cam:
            num_cameras = 1
            camera_angle_list = [Tensor([0, 0, 5])]
        else:
            # num_cameras = 8
            # camera_angle_list = [
            #     Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
            #     for a in np.linspace(0, math.pi / 4, num_cameras)
            # ]
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
                Tensor([5 / math.sqrt(3), 5 / math.sqrt(3), 5 / math.sqrt(3)]),
                Tensor([5 / math.sqrt(3), 5 / math.sqrt(3),
                        -5 / math.sqrt(3)]),
                Tensor([5 / math.sqrt(3), -5 / math.sqrt(3),
                        5 / math.sqrt(3)]),
                Tensor([-5 / math.sqrt(3), 5 / math.sqrt(3),
                        5 / math.sqrt(3)]),
                Tensor(
                    [-5 / math.sqrt(3), -5 / math.sqrt(3), 5 / math.sqrt(3)]),
                Tensor(
                    [-5 / math.sqrt(3), 5 / math.sqrt(3), -5 / math.sqrt(3)]),
                Tensor(
                    [5 / math.sqrt(3), -5 / math.sqrt(3), -5 / math.sqrt(3)]),
                Tensor(
                    [-5 / math.sqrt(3), -5 / math.sqrt(3), -5 / math.sqrt(3)])
            ]
            num_cameras = len(camera_angle_list)

        optimizer = torch.optim.Adam(
            [grid_initial],
            lr=learning_rate,
            eps=1e-8,
        )

        # NOTE: from the paper
        # -   We should focus on the views with large losses. Our
        #     approach first calculates the average loss for all
        #     the camera views from the result of the previous iteration.
        # -   If a loss for a view is greater
        #     than the average loss, then during the current iteration,
        #     we update SDF until the loss for this view is less than the
        #     average (with 20 max updates).
        # -   For the other views, we update the SDF five times. If one
        #     update increases the loss, then we switch to the next view
        #     directly. We stop our optimization process when the loss is
        #     smaller than a given tolerance or the step length is too small.

        global_iter = 0
        cam_view_loss_list = [None] * num_cameras

        for sdf_grid_res in self.sdf_grid_res_list:
            res_loss_list = []
            std_res_loss = None
            avg_cam_view_loss = None
            increase_res = False

            sdf_grid_res_iter = 0
            while not increase_res:
                for cam_view_idx in range(num_cameras):
                    # prev_cam_loss = None
                    for cam_iter in range(max_num_iters_per_camera):
                        image_initial = generate_image(
                            bounding_box_min_x,
                            bounding_box_min_y,
                            bounding_box_min_z,
                            bounding_box_max_x,
                            bounding_box_max_y,
                            bounding_box_max_z,
                            voxel_size,
                            grid_res_x,
                            grid_res_y,
                            grid_res_z,
                            self.out_img_width,
                            self.out_img_height,
                            grid_initial,
                            camera_angle_list[cam_view_idx],
                        )

                        image_loss, sdf_loss = self.loss_fn(
                            image_initial,
                            None,
                            grid_initial,
                            voxel_size,
                            grid_res_x,
                            grid_res_y,
                            grid_res_z,
                            self.out_img_width,
                            self.out_img_height,
                        )

                        image_loss *= torch.prod(
                            torch.tensor(image_initial.shape).detach().clone())
                        sdf_loss *= torch.prod(
                            torch.tensor(grid_initial.shape).detach().clone())

                        image_loss *= image_loss_weight
                        sdf_loss *= sdf_loss_weight

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
                        lp_loss *= torch.prod(
                            torch.tensor(conv_result.shape).detach().clone())
                        lp_loss *= lp_loss_weight

                        cam_view_loss = image_loss + sdf_loss + lp_loss

                        if avg_cam_view_loss is not None:
                            if cam_iter >= 5 and cam_view_loss < avg_cam_view_loss:
                                break

                        # if prev_cam_loss is not None:
                        #     if cam_iter >= 5 and prev_cam_loss > cam_view_loss:
                        #         break

                        optimizer.zero_grad()
                        cam_view_loss.backward()
                        optimizer.step()

                        cam_view_loss_list[cam_view_idx] = cam_view_loss

                        # prev_cam_loss = cam_view_loss

                        global_iter += 1

                    camera_angle_list = camera_angle_list[::-1]

                    if True:  #global_iter % 5 == 0:
                        if self.print_jupyter:
                            clear_output(wait=True)

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
                        print("STD:", std_res_loss)
                        print("")
                        print("AVG:", avg_cam_view_loss)
                        print("")
                        print("sdf grid res:", sdf_grid_res, " - "
                              "iteration:", sdf_grid_res_iter + 1, " - ",
                              "cam view idx", cam_view_idx, " - ",
                              "cam iters:", cam_iter + 1)
                        print("")

                        if self.save_results:
                            torchvision.utils.save_image(
                                image_initial.detach(),
                                "./" + results_dir + "/" + "final_cam_" +
                                str(grid_res_x) + "_" + str(sdf_grid_res) +
                                "-" + str(cam_view_idx) + ".jpg",
                                nrow=8,
                                padding=2,
                                normalize=False,
                                range=None,
                                scale_each=False,
                                pad_value=0)

                        if self.print_jupyter:
                            image_initial_array = image_initial.detach().cpu(
                            ).numpy() * 255
                            display(
                                Image.fromarray(
                                    image_initial_array.astype(np.uint8)))

                avg_cam_view_loss = torch.tensor(cam_view_loss_list).mean()
                res_loss_list.append(avg_cam_view_loss)

                if len(res_loss_list) > num_std_res_samples:
                    std_res_loss = torch.tensor(
                        res_loss_list[-num_std_res_samples:]).std()
                    if std_res_loss < max_std_res_loss:
                        increase_res = True

                sdf_grid_res_iter += 1

        # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # #

        # global_iter = 0
        # while grid_res_idx + 1 < len(self.sdf_grid_res_list):
        #     # tolerance *= 1.05

        #     loss_camera = [100000] * len(camera_angle_list)
        #     average = 1000000
        #     res_iter = 0
        #     while sum(loss_camera) < average - tolerance or res_iter <= 1:
        #         average = sum(loss_camera)
        #         res_iter += 1

        #         for cam in range(len(camera_angle_list)):
        #             loss = math.inf
        #             prev_loss = math.inf

        #             cam_iter = 0
        #             while cam_iter < 5 and loss < prev_loss or cam_iter <= 1:
        #                 cam_iter += 1
        #                 prev_loss = loss

        #                 image_initial = generate_image(
        #                     bounding_box_min_x,
        #                     bounding_box_min_y,
        #                     bounding_box_min_z,
        #                     bounding_box_max_x,
        #                     bounding_box_max_y,
        #                     bounding_box_max_z,
        #                     voxel_size,
        #                     grid_res_x,
        #                     grid_res_y,
        #                     grid_res_z,
        #                     self.out_img_width,
        #                     self.out_img_height,
        #                     grid_initial,
        #                     camera_angle_list[cam],
        #                 )

        #                 image_loss, sdf_loss = self.loss_fn(
        #                     image_initial,
        #                     None,
        #                     grid_initial,
        #                     voxel_size,
        #                     grid_res_x,
        #                     grid_res_y,
        #                     grid_res_z,
        #                     self.out_img_width,
        #                     self.out_img_height,
        #                 )

        #                 image_loss *= torch.prod(
        #                     torch.tensor(image_initial.shape).detach().clone())
        #                 sdf_loss *= torch.prod(
        #                     torch.tensor(grid_initial.shape).detach().clone())

        #                 image_loss *= image_loss_weight
        #                 sdf_loss *= sdf_loss_weight

        #                 conv_input = (grid_initial).unsqueeze(0).unsqueeze(0)
        #                 conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0],
        #                                                          [0, 1, 0],
        #                                                          [0, 0, 0]],
        #                                                         [[0, 1, 0],
        #                                                          [1, -6, 1],
        #                                                          [0, 1, 0]],
        #                                                         [[0, 0, 0],
        #                                                          [0, 1, 0],
        #                                                          [0, 0, 0]]]]])

        #                 conv_result = (F.conv3d(conv_input, conv_filter)**2)
        #                 lp_loss = conv_result.mean()
        #                 lp_loss *= torch.prod(
        #                     torch.tensor(conv_result.shape).detach().clone())
        #                 lp_loss *= lp_loss_weight

        #                 loss = image_loss + sdf_loss + lp_loss

        #                 loss_camera[cam] = image_loss + sdf_loss
        #                 loss_camera[cam] /= num_cameras

        #                 optimizer.zero_grad()
        #                 loss.backward()
        #                 optimizer.step()

        #                 global_iter += 1

        #             if True:  #global_iter % 5 == 0:
        #                 if self.print_jupyter:
        #                     clear_output(wait=True)

        #                 print("\n")
        #                 print("Tolerance / 2 ", tolerance / 2)
        #                 print("Diff ", average - sum(loss_camera))
        #                 print("Average ", average)
        #                 print("Loss camera ", sum(loss_camera))
        #                 print("")

        #                 print("image loss: ", image_loss)
        #                 print("image weight: ",
        #                       torch.prod(torch.tensor(image_initial.shape)))
        #                 print("sdf loss: ", sdf_loss)
        #                 print("sdf weight: ",
        #                       torch.prod(torch.tensor(grid_initial.shape)))
        #                 print("lp loss: ", lp_loss)
        #                 print("lp weight: ",
        #                       torch.prod(torch.tensor(conv_input.shape)))
        #                 print("")
        #                 print("loss: ", loss)
        #                 print("")
        #                 print("grid res: ", grid_res_x, " - ", "res iter:",
        #                       res_iter, " - ", "cam iter:", cam_iter, " - ",
        #                       "cam: ", cam, "\ncamera:",
        #                       camera_angle_list[cam])
        #                 print("")

        #                 if not self.print_jupyter:
        #                     torchvision.utils.save_image(
        #                         image_initial.detach(),
        #                         "./" + results_dir + "/" + "final_cam_" +
        #                         str(grid_res_x) + "_" + str(cam) + "_" +
        #                         str(res_iter) + "-" + str(cam_iter) + ".jpg",
        #                         nrow=8,
        #                         padding=2,
        #                         normalize=False,
        #                         range=None,
        #                         scale_each=False,
        #                         pad_value=0)

        #                 else:
        #                     image_initial_array = image_initial.detach().cpu(
        #                     ).numpy() * 255
        #                     display(
        #                         Image.fromarray(
        #                             image_initial_array.astype(np.uint8)))

            if grid_res_idx + 1 >= len(self.sdf_grid_res_list):
                break
            grid_res_update_x = grid_res_update_y = grid_res_update_z = self.sdf_grid_res_list[
                grid_res_idx + 1]
            voxel_size_update = (bounding_box_max_x -
                                 bounding_box_min_x) / (grid_res_update_x - 1)
            grid_initial_update = Tensor(grid_res_update_x, grid_res_update_y,
                                         grid_res_update_z)
            linear_space_x = torch.linspace(0, grid_res_update_x - 1,
                                            grid_res_update_x)
            linear_space_y = torch.linspace(0, grid_res_update_y - 1,
                                            grid_res_update_y)
            linear_space_z = torch.linspace(0, grid_res_update_z - 1,
                                            grid_res_update_z)
            first_loop = linear_space_x.repeat(
                grid_res_update_y * grid_res_update_z,
                1).t().contiguous().view(-1).unsqueeze_(1)
            second_loop = linear_space_y.repeat(
                grid_res_update_z,
                grid_res_update_x).t().contiguous().view(-1).unsqueeze_(1)
            third_loop = linear_space_z.repeat(grid_res_update_x *
                                               grid_res_update_y).unsqueeze_(1)
            loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()
            min_x = Tensor([bounding_box_min_x]).repeat(
                grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
            min_y = Tensor([bounding_box_min_y]).repeat(
                grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
            min_z = Tensor([bounding_box_min_z]).repeat(
                grid_res_update_x * grid_res_update_y * grid_res_update_z, 1)
            bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1)

            # Get the position of the grid points in the refined grid
            points = bounding_min_matrix + voxel_size_update * loop
            voxel_min_point_index_x = torch.floor(
                (points[:, 0].unsqueeze_(1) - min_x) /
                voxel_size).clamp(max=grid_res_x - 2)
            voxel_min_point_index_y = torch.floor(
                (points[:, 1].unsqueeze_(1) - min_y) /
                voxel_size).clamp(max=grid_res_y - 2)
            voxel_min_point_index_z = torch.floor(
                (points[:, 2].unsqueeze_(1) - min_z) /
                voxel_size).clamp(max=grid_res_z - 2)
            voxel_min_point_index = torch.cat(
                (voxel_min_point_index_x, voxel_min_point_index_y,
                 voxel_min_point_index_z), 1)
            voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

            # Compute the sdf value of the grid points in the refined grid
            grid_initial_update = calculate_sdf_value(
                grid_initial, points, voxel_min_point, voxel_min_point_index,
                voxel_size, grid_res_x, grid_res_y,
                grid_res_z).view(grid_res_update_x, grid_res_update_y,
                                 grid_res_update_z)

            # Update the grid resolution for the refined sdf grid
            grid_res_x = grid_res_update_x
            grid_res_y = grid_res_update_y
            grid_res_z = grid_res_update_z

            # Update the voxel size for the refined sdf grid
            voxel_size = voxel_size_update

            # Update the sdf grid
            grid_initial = grid_initial_update.data
            grid_initial.requires_grad = True

            exp_avg = optimizer.state_dict()['state'][0]['exp_avg'].clone()
            exp_avg = torch.nn.functional.interpolate(
                exp_avg[None, None, :],
                size=(grid_res_x, grid_res_y, grid_res_z),
            )[0, 0, :]

            exp_avg_sq = optimizer.state_dict(
            )['state'][0]['exp_avg_sq'].clone()
            exp_avg_sq = torch.nn.functional.interpolate(
                exp_avg_sq[None, None, :],
                size=(grid_res_x, grid_res_y, grid_res_z),
            )[0, 0, :]

            optimizer = torch.optim.Adam(
                [grid_initial],
                lr=learning_rate,
                eps=1e-8,
            )

            return optimizer
            optimizer.state_dict()['state'][0]['exp_avg'] = exp_avg
            optimizer.state_dict()['state'][0]['exp_avg_sq'] = exp_avg_sq

            # Double the size of the image
            if self.out_img_width < 256:
                self.out_img_width = int(self.out_img_width * 2)
                self.out_img_height = int(self.out_img_height * 2)

            # learning_rate /= 1.03
            grid_res_idx += 1

        self.vis_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)

        out_video_path = os.path.join(self.vis_dir, "generation.mp4")
        print(f"Generating {out_video_path}")

        cmd = ("ffmpeg -y "
               "-r 5 "
               f"-pattern_type glob -i '{results_dir}/*.jpg' "
               "-vcodec libx264 "
               "-crf 25 "
               "-pix_fmt yuv420p "
               f"{out_video_path}")

        subprocess.check_call(cmd, shell=True)

        num_cameras = 26
        camera_angle_list = [
            torch.tensor([5 * math.cos(a), 0, 5 * math.sin(a)]).cuda()
            for a in np.linspace(0, math.pi / 4, num_cameras)
        ]

        out_img_width = 512
        out_img_height = 512

        for cam in range(num_cameras):
            image_initial = generate_image(
                bounding_box_min_x,
                bounding_box_min_y,
                bounding_box_min_z,
                bounding_box_max_x,
                bounding_box_max_y,
                bounding_box_max_z,
                voxel_size,
                grid_res_x,
                grid_res_y,
                grid_res_z,
                out_img_width,
                out_img_height,
                grid_initial,
                camera_angle_list[cam],
            )

            torchvision.utils.save_image(
                image_initial.detach(),
                os.path.join(self.vis_dir, f"{cam}.jpg"),
                nrow=8,
                padding=2,
                normalize=False,
                range=None,
                scale_each=False,
                pad_value=0,
            )

        out_video_path = os.path.join(self.vis_dir, "visualization.mp4")
        print(f"Generating {out_video_path}")

        cmd = ("ffmpeg -y "
               "-r 5 "
               f"-pattern_type glob -i '{self.vis_dir}/*.jpg' "
               "-vcodec libx264 "
               "-crf 25 "
               "-pix_fmt yuv420p "
               f"{out_video_path}")

        subprocess.check_call(cmd, shell=True)

        print("\n----- END -----")

        return grid_initial


if __name__ == "__main__":
    sdf_clip = SDFCLIP()
    sdf_clip.run()