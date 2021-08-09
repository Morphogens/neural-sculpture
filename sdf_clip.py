import os
import math
from typing import *

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

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
        out_img_width: int = 256,
        out_img_height: int = 256,
        out_dir: str = "./results",
    ):
        self.out_img_width = out_img_width
        self.out_img_height = out_img_height
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

        self.use_single_cam = True
        self.voxel_res_list = [8, 16, 24, 32, 40, 48, 56, 64]

        self.clip_loss = CLIPLoss(
            text_target=prompt,
            device=device,
        )

    def run(
        self,
        learning_rate: float = 1e-1,
        tolerance: float = 8 / 10,
        num_iters_per_cam=1,
        num_iters_per_res: int = 64,
        image_loss_weight: float = 2000.,
        sdf_loss_weight: float = 1.,
        lp_loss_weight: float = 1.,
    ):
        bounding_box_min_x = -2.
        bounding_box_min_y = -2.
        bounding_box_min_z = -2.
        bounding_box_max_x = 2.
        bounding_box_max_y = 2.
        bounding_box_max_z = 2.

        grid_res_x = grid_res_y = grid_res_z = self.voxel_res_list.pop(0)
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
            num_cameras = 26
            camera_angle_list = [
                Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
                for a in np.linspace(0, math.pi / 4, num_cameras)
            ]

        image_loss_list = [math.inf] * num_cameras
        sdf_loss_list = [math.inf] * num_cameras

        optimizer = torch.optim.Adam(
            [grid_initial],
            lr=learning_rate,
            # eps=1e-2,
            eps=1e-8,
        )

        global_iter = 0

        while len(self.voxel_res_list):
            loss_camera = [1000] * num_cameras
            average = 100000

            tolerance *= 1.05

            for res_iter in range(num_iters_per_res):
                # if sum(loss_camera) > average - tolerance / 2:
                #     break

                average = sum(loss_camera)

                for cam in range(num_cameras):
                    prev_loss = math.inf

                    num = 0
                    while num < num_iters_per_cam:  # and loss < prev_loss:
                        loss = 0
                        num += 1
                        prev_loss = loss

                        # Generate images
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
                            camera_angle_list[cam],
                            1,
                            camera_angle_list,
                        )

                        image_loss, sdf_loss = self.loss_fn(
                            image_initial, None, grid_initial, voxel_size,
                            grid_res_x, grid_res_y, grid_res_z,
                            self.out_img_width, self.out_img_height)

                        image_loss *= image_loss_weight
                        sdf_loss *= sdf_loss_weight
                        # sdf_loss /= grid_res_x

                        image_loss_list[cam] = image_loss
                        sdf_loss_list[cam] = sdf_loss

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
                        lp_loss = torch.sum(
                            F.conv3d(conv_input, conv_filter)**2)

                        lp_loss *= lp_loss_weight
                        # lp_loss /= grid_res_x

                        # get total loss
                        loss += image_loss + sdf_loss + lp_loss

                        image_loss_list[cam] = image_loss / num_cameras
                        sdf_loss_list[cam] = sdf_loss / num_cameras

                        loss_camera[
                            cam] = image_loss_list[cam] + sdf_loss_list[cam]

                        if global_iter % 5 == 0:
                            # print out loss messages
                            print(
                                "\ngrid res:",
                                grid_res_x,
                                "iteration:",
                                res_iter,
                                "num:",
                                num,
                                "\ncamera:",
                                camera_angle_list[cam],
                                "\nloss:",
                                (loss / num_cameras).item(),
                                "\nimage_loss:",
                                image_loss.item(),
                                "\nsdf_loss",
                                sdf_loss.item(),
                                "\nlp_loss:",
                                lp_loss.item(),
                            )

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
                                camera_angle_list[cam],
                                0,
                                camera_angle_list,
                            )

                            torchvision.utils.save_image(
                                image_initial,
                                "./" + self.out_dir + "/" + "final_cam_" +
                                str(grid_res_x) + "_" + str(cam) + "_" +
                                str(res_iter) + "-" + str(num) + ".png",
                                nrow=8,
                                padding=2,
                                normalize=False,
                                range=None,
                                scale_each=False,
                                pad_value=0)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        global_iter += 1

            # num_iters_per_res = 10
            # for iter_i in range(num_iters_per_res):
            #     # loss = 0
            #     # avg_image_loss = 0
            #     # avg_sdf_loss = 0
            #     # avg_Lp_loss = 0
            #     # optimizer.zero_grad()

            #     for cam in range(num_cameras):
            #         for _ in range(5):
            #             loss = 0
            #             avg_image_loss = 0
            #             avg_sdf_loss = 0
            #             avg_Lp_loss = 0
            #             optimizer.zero_grad()

            #             # Generate images
            #             image_initial = self.generate_image(
            #                 bounding_box_min_x, bounding_box_min_y, bounding_box_min_z,
            #                 bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
            #                 voxel_size, grid_res_x, grid_res_y, grid_res_z, self.out_img_width, self.out_img_height,
            #                 grid_initial, camera_angle_list[cam], 1, camera_angle_list
            #             )

            #             # Perform backprobagation
            #             # compute image loss and sdf loss
            #             image_loss_list, sdf_loss_list = self.loss_fn(
            #                 image_initial, None, grid_initial,
            #                 voxel_size, grid_res_x, grid_res_y, grid_res_z,
            #                 self.out_img_width, self.out_img_height
            #             )

            #             conv_input = (grid_initial).unsqueeze(0).unsqueeze(0)
            #             conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [
            #                                                     [0, 1, 0], [1, -6, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])
            #             lp_loss = torch.sum(F.conv3d(conv_input, conv_filter) ** 2)

            #             # get total loss
            #             loss += (image_loss_list * 1000 + sdf_loss_list + lp_loss)

            #             avg_image_loss += image_loss
            #             avg_sdf_loss += sdf_loss_list
            #             avg_Lp_loss += lp_loss

            #             loss.backward()
            #             optimizer.step()

            #     # loss.backward()
            #     # optimizer.step()

            #     # print out loss messages
            #     print(
            #         '\n', iter_i,
            #         "\ngrid res:", grid_res_x,
            #         "\nloss:", loss.item(),
            #         "\nimage_loss:", avg_image_loss.item(),
            #         "\nsdf_loss", avg_sdf_loss.item(),
            #         "\nlp_loss:", avg_Lp_loss.item(),
            #     )

            #     if iter_i % 1 == 0:
            #         # genetate result images
            #         for cam in range(num_cameras)[:5]:
            #             image_initial = self.generate_image(
            #                 bounding_box_min_x, bounding_box_min_y, bounding_box_min_z,
            #                 bounding_box_max_x, bounding_box_max_y, bounding_box_max_z,
            #                 voxel_size, grid_res_x, grid_res_y, grid_res_z, self.out_img_width, self.out_img_height,
            #                 grid_initial, camera_angle_list[cam], 0, camera_angle_list
            #             )

            #             # img_path = "./" + self.out_dir + "/" + "final_cam_" + str(grid_res_x) + "_" + str(cam) + ".png"
            #             img_path = f"./{self.out_dir}/{grid_res_x}_{iter_i}_cam:{cam}.png"
            #             torchvision.utils.save_image(
            #                 image_initial, img_path, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0
            #             )
            # genetate result images
            # Save the final SDF result

            # with open(
            #         "./" + self.out_dir + "/" + str(grid_res_x) +
            #         "_best_sdf_bunny.pt", 'wb') as f:
            #     torch.save(grid_initial, f)

            # moves on to the next resolution stage
            if len(self.voxel_res_list):
                grid_res_update_x = grid_res_update_y = grid_res_update_z = self.voxel_res_list.pop(
                    0)
                voxel_size_update = (bounding_box_max_x - bounding_box_min_x
                                     ) / (grid_res_update_x - 1)
                grid_initial_update = Tensor(grid_res_update_x,
                                             grid_res_update_y,
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
                third_loop = linear_space_z.repeat(
                    grid_res_update_x * grid_res_update_y).unsqueeze_(1)
                loop = torch.cat((first_loop, second_loop, third_loop),
                                 1).cuda()
                min_x = Tensor([bounding_box_min_x]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z,
                    1)
                min_y = Tensor([bounding_box_min_y]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z,
                    1)
                min_z = Tensor([bounding_box_min_z]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z,
                    1)
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
                    grid_initial, points, voxel_min_point,
                    voxel_min_point_index, voxel_size, grid_res_x, grid_res_y,
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

                optimizer = torch.optim.Adam(
                    [grid_initial],
                    lr=learning_rate,
                    # eps=1e-2,
                    eps=1e-8,
                )

                # sdf_loss_weight /= 4

                # Double the size of the image
                if self.out_img_width < 256:
                    self.out_img_width = int(self.out_img_width * 2)
                    self.out_img_height = int(self.out_img_height * 2)

                # learning_rate /= 1.03

        print("----- END -----")

    # The energy E captures the difference between a rendered image and
    # a desired target image, and the rendered image is a function of the
    # SDF values. You could write E(SDF) = ||rendering(SDF)-target_image||^2.
    # In addition, there is a second term in the energy as you observed that
    # constrains the length of the normal of the SDF to 1. This is a regularization
    # term to make sure the output is still a valid SDF.
    # HI JOEL

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

        # image_loss = torch.sum(torch.abs(target - output))  # / (width * height)
        output = output[None, None].repeat(1, 3, 1, 1)  # CLIP expects RGB
        # output = augment(output, 8)
        image_loss = self.clip_loss.compute(output)

        [grid_normal_x, grid_normal_y,
         grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x,
                                          grid_res_y, grid_res_z)
        sdf_loss = torch.sum(
            torch.abs(
                torch.pow(
                    grid_normal_x[1:grid_res_x - 1, 1:grid_res_y - 1,
                                  1:grid_res_z - 1], 2) +
                torch.pow(
                    grid_normal_y[1:grid_res_x - 1, 1:grid_res_y - 1,
                                  1:grid_res_z - 1], 2) +
                torch.pow(
                    grid_normal_z[1:grid_res_x - 1, 1:grid_res_y - 1,
                                  1:grid_res_z - 1], 2) -
                1))  # / ((grid_res-1) * (grid_res-1) * (grid_res-1))

        # print("\n\nimage loss: ", image_loss)
        # print("sdf loss: ", sdf_loss)

        return image_loss, sdf_loss


if __name__ == "__main__":
    sdf_clip = SDFCLIP()
    sdf_clip.run()