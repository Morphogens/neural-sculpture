import os
import math

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

import renderer
from clip_loss import CLIPLoss
from grid_utils import grid_construction_sphere_big, get_grid_normal, get_intersection_normal

# NOTE: speed up
torch.backends.cudnn.benchmark = True


class SDFClip:
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('SDFClip device: ', self.device)

        if self.device == 'cuda':
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.FloatTensor

        self.clip_loss = CLIPLoss(
            text_target=prompt,
            device=self.device,
        )

    def run(
        self,
        learning_rate: float = 1e-2,
        tolerance: float = 8 / 10,
        num_iters_per_cam=5,
        num_iters_per_res: int = 64,
    ):
        bounding_box_min_x = -2.
        bounding_box_min_y = -2.
        bounding_box_min_z = -2.
        bounding_box_max_x = 2.
        bounding_box_max_y = 2.
        bounding_box_max_z = 2.

        grid_res_x = grid_res_y = grid_res_z = self.voxel_res_list.pop(0)
        voxel_size = self.Tensor([4. / (grid_res_x - 1)])

        grid_initial = grid_construction_sphere_big(
            grid_res_x,
            bounding_box_min_x,
            bounding_box_max_x,
        )
        grid_initial.requires_grad = True

        if self.use_single_cam:
            num_cameras = 1
            camera_angle_list = [self.Tensor([0, 0, 5])]
        else:
            num_cameras = 26
            camera_angle_list = [
                self.Tensor([5 * math.cos(a), 0, 5 * math.sin(a)])
                for a in np.linspace(0, math.pi / 4, num_cameras)
            ]

        image_loss_list = [math.inf] * num_cameras
        sdf_loss_list = [math.inf] * num_cameras

        iterations = 0
        while len(self.voxel_res_list):
            i = 0
            loss_camera = [1000] * num_cameras
            average = 100000
            num_steps = 64

            tolerance *= 1.05
            image_target = []

            optimizer = torch.optim.Adam(
                [grid_initial],
                lr=learning_rate,
                eps=1e-2,
            )

            while i < num_iters_per_res:  #sum(loss_camera) < average - tolerance / 2:
                average = sum(loss_camera)
                for cam in range(num_cameras):
                    prev_loss = math.inf

                    loss = 0
                    num = 0
                    while ((num < num_iters_per_cam) and loss < prev_loss):
                        num += 1
                        prev_loss = loss
                        iterations += 1

                        # Generate images
                        image_initial = self.generate_image(
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

                        # Perform backprobagation
                        # compute image loss and sdf loss
                        image_loss_list[cam], sdf_loss_list[
                            cam] = self.loss_fn(image_initial, None,
                                                grid_initial, voxel_size,
                                                grid_res_x, grid_res_y,
                                                grid_res_z, self.out_img_width,
                                                self.out_img_height)

                        clip_loss_val = image_loss_list[cam].item()
                        # compute laplacian loss
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
                        Lp_loss = torch.sum(
                            F.conv3d(conv_input, conv_filter)**2)

                        # get total loss
                        image_loss_list[cam] *= 2000
                        loss += image_loss_list[cam] + sdf_loss_list[
                            cam] + Lp_loss
                        image_loss_list[
                            cam] = image_loss_list[cam] / num_cameras
                        sdf_loss_list[cam] = sdf_loss_list[cam] / num_cameras

                        loss_camera[
                            cam] = image_loss_list[cam] + sdf_loss_list[cam]

                        # print out loss messages
                        print(
                            "\ngrid res:",
                            grid_res_x,
                            "iteration:",
                            i,
                            "num:",
                            num,
                            "\ncamera:",
                            camera_angle_list[cam],
                            "\nloss:",
                            (loss / num_cameras).item(),
                            "\nclip_loss_val",
                            clip_loss_val,
                            "\nimage_loss:",
                            image_loss_list[cam].item(),
                            "\nsdf_loss",
                            sdf_loss_list[cam].item(),
                            "\nlp_loss:",
                            Lp_loss.item(),
                        )

                        for cam in range(num_cameras):
                            image_initial = self.generate_image(
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
                                str(i) + "-" + str(num) + ".png",
                                nrow=8,
                                padding=2,
                                normalize=False,
                                range=None,
                                scale_each=False,
                                pad_value=0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(f"I have done {num} iterations per camera")
                i += 1
            print(f"I have done {i} iterations per grid")

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

            #             clip_loss_val = image_loss_list.item()
            #             # compute laplacian loss
            #             conv_input = (grid_initial).unsqueeze(0).unsqueeze(0)
            #             conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [
            #                                                     [0, 1, 0], [1, -6, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])
            #             Lp_loss = torch.sum(F.conv3d(conv_input, conv_filter) ** 2)

            #             # get total loss
            #             loss += (image_loss_list * 1000 + sdf_loss_list + Lp_loss)

            #             avg_image_loss += image_loss
            #             avg_sdf_loss += sdf_loss_list
            #             avg_Lp_loss += Lp_loss

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
            with open(
                    "./" + self.out_dir + "/" + str(grid_res_x) +
                    "_best_sdf_bunny.pt", 'wb') as f:
                torch.save(grid_initial, f)

            # moves on to the next resolution stage
            if len(self.voxel_res_list):
                grid_res_update_x = grid_res_update_y = grid_res_update_z = self.voxel_res_list.pop(
                    0)
                voxel_size_update = (bounding_box_max_x - bounding_box_min_x
                                     ) / (grid_res_update_x - 1)
                grid_initial_update = self.Tensor(grid_res_update_x,
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
                min_x = self.Tensor([bounding_box_min_x]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z,
                    1)
                min_y = self.Tensor([bounding_box_min_y]).repeat(
                    grid_res_update_x * grid_res_update_y * grid_res_update_z,
                    1)
                min_z = self.Tensor([bounding_box_min_z]).repeat(
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
                grid_initial_update = self.calculate_sdf_value(
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

                # Double the size of the image
                if self.out_img_width < 256:
                    self.out_img_width = int(self.out_img_width * 2)
                    self.out_img_height = int(self.out_img_height * 2)
                learning_rate /= 1.03

        print("----- END -----")

    # Do one more step for ray matching
    def calculate_sdf_value(
        self,
        grid,
        points,
        voxel_min_point,
        voxel_min_point_index,
        voxel_size,
        grid_res_x,
        grid_res_y,
        grid_res_z,
    ):
        string = ""

        # Linear interpolate along x axis the eight values
        tx = (points[:, 0] - voxel_min_point[:, 0]) / voxel_size
        string = string + "\n\nvoxel_size: \n" + str(voxel_size)
        string = string + "\n\ntx: \n" + str(tx)
        print(grid.shape)

        if self.device == 'cuda':
            tx = tx.cuda()
            x = voxel_min_point_index.long()[:, 0]
            y = voxel_min_point_index.long()[:, 1]
            z = voxel_min_point_index.long()[:, 2]

            string = string + "\n\nx: \n" + str(x)
            string = string + "\n\ny: \n" + str(y)
            string = string + "\n\nz: \n" + str(z)

            c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z]
            c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z]
            c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1]
            c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1,
                                                               z + 1]

            string = string + "\n\n(1 - tx): \n" + str((1 - tx))
            string = string + "\n\ngrid[x,y,z]: \n" + str(grid[x, y, z])
            string = string + "\n\ngrid[x+1,y,z]: \n" + str(grid[x + 1, y, z])
            string = string + "\n\nc01: \n" + str(c01)
            string = string + "\n\nc23: \n" + str(c23)
            string = string + "\n\nc45: \n" + str(c45)
            string = string + "\n\nc67: \n" + str(c67)

            # Linear interpolate along the y axis
            ty = (points[:, 1] - voxel_min_point[:, 1]) / voxel_size
            ty = ty.cuda()
            c0 = (1 - ty) * c01 + ty * c23
            c1 = (1 - ty) * c45 + ty * c67

            string = string + "\n\nty: \n" + str(ty)

            string = string + "\n\nc0: \n" + str(c0)
            string = string + "\n\nc1: \n" + str(c1)

            # Return final value interpolated along z
            tz = (points[:, 2] - voxel_min_point[:, 2]) / voxel_size
            tz = tz.cuda()
            string = string + "\n\ntz: \n" + str(tz)

        else:
            x = voxel_min_point_index.numpy()[:, 0]
            y = voxel_min_point_index.numpy()[:, 1]
            z = voxel_min_point_index.numpy()[:, 2]

            c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z]
            c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z]
            c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1]
            c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1,
                                                               z + 1]

            # Linear interpolate along the y axis
            ty = (points[:, 1] - voxel_min_point[:, 1]) / voxel_size
            c0 = (1 - ty) * c01 + ty * c23
            c1 = (1 - ty) * c45 + ty * c67

            # Return final value interpolated along z
            tz = (points[:, 2] - voxel_min_point[:, 2]) / voxel_size

        result = (1 - tz) * c0 + tz * c1

        return result

    def compute_intersection_pos(
        self,
        grid,
        intersection_pos_rough,
        voxel_min_point,
        voxel_min_point_index,
        ray_direction,
        voxel_size,
        mask,
        width,
        height,
    ):

        # Linear interpolate along x axis the eight values
        tx = (intersection_pos_rough[:, :, 0] -
              voxel_min_point[:, :, 0]) / voxel_size

        if self.device == 'cuda':

            x = voxel_min_point_index.long()[:, :, 0]
            y = voxel_min_point_index.long()[:, :, 1]
            z = voxel_min_point_index.long()[:, :, 2]

            c01 = (1 - tx) * grid[x, y, z].cuda() + tx * grid[x + 1, y,
                                                              z].cuda()
            c23 = (1 - tx) * grid[x, y + 1, z].cuda() + tx * grid[x + 1, y + 1,
                                                                  z].cuda()
            c45 = (1 - tx) * grid[x, y, z + 1].cuda() + tx * grid[x + 1, y, z +
                                                                  1].cuda()
            c67 = (1 - tx) * grid[x, y+1, z+1].cuda() + \
                tx * grid[x+1, y+1, z+1].cuda()

        else:
            x = voxel_min_point_index.numpy()[:, :, 0]
            y = voxel_min_point_index.numpy()[:, :, 1]
            z = voxel_min_point_index.numpy()[:, :, 2]

            c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z]
            c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z]
            c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1]
            c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1,
                                                               z + 1]

        # Linear interpolate along the y axis
        ty = (intersection_pos_rough[:, :, 1] -
              voxel_min_point[:, :, 1]) / voxel_size
        c0 = (1 - ty) * c01 + ty * c23
        c1 = (1 - ty) * c45 + ty * c67

        # Return final value interpolated along z
        tz = (intersection_pos_rough[:, :, 2] -
              voxel_min_point[:, :, 2]) / voxel_size

        sdf_value = (1 - tz) * c0 + tz * c1

        return (intersection_pos_rough + ray_direction * sdf_value.view(width, height, 1).repeat(1, 1, 3))\
            + (1 - mask.view(width, height, 1).repeat(1, 1, 3))

    def generate_image(
        self,
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
        width,
        height,
        grid,
        camera,
        back,
        camera_angle_list,
    ):

        # Get normal vectors for points on the grid
        [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(
            grid,
            voxel_size,
            grid_res_x,
            grid_res_y,
            grid_res_z,
        )

        # Generate rays
        w_h_3 = torch.zeros(width, height, 3).to(self.device)
        w_h = torch.zeros(width, height).to(self.device)

        eye_x = camera[0]
        eye_y = camera[1]
        eye_z = camera[2]

        # Do ray tracing in cpp
        outputs = renderer.ray_matching(
            w_h_3,
            w_h,
            grid,
            width,
            height,
            bounding_box_min_x,
            bounding_box_min_y,
            bounding_box_min_z,
            bounding_box_max_x,
            bounding_box_max_y,
            bounding_box_max_z,
            grid_res_x,
            grid_res_y,
            grid_res_z,
            eye_x,
            eye_y,
            eye_z,
        )

        # {intersection_pos, voxel_position, directions}
        intersection_pos_rough = outputs[0]
        voxel_min_point_index = outputs[1]
        ray_direction = outputs[2]

        # Initialize grid values and normals for intersection voxels
        intersection_grid_normal_x = self.Tensor(width, height, 8)
        intersection_grid_normal_y = self.Tensor(width, height, 8)
        intersection_grid_normal_z = self.Tensor(width, height, 8)
        intersection_grid = self.Tensor(width, height, 8)

        # Make the pixels with no intersections with rays be 0
        mask = (voxel_min_point_index[:, :, 0] != -1).type(self.Tensor)

        # Get the indices of the minimum point of the intersecting voxels
        x = voxel_min_point_index[:, :, 0].type(torch.cuda.LongTensor)
        y = voxel_min_point_index[:, :, 1].type(torch.cuda.LongTensor)
        z = voxel_min_point_index[:, :, 2].type(torch.cuda.LongTensor)
        x[x == -1] = 0
        y[y == -1] = 0
        z[z == -1] = 0

        # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
        # This line is equivalent to grid_normal_x[x,y,z]
        x1 = torch.index_select(
            grid_normal_x.view(-1), 0,
            z.view(-1) + grid_res_x * y.view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        x2 = torch.index_select(grid_normal_x.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x * y.view(-1) +
                                grid_res_x * grid_res_x * x.view(-1)).view(
                                    x.shape).unsqueeze_(2)
        x3 = torch.index_select(
            grid_normal_x.view(-1), 0,
            z.view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        x4 = torch.index_select(
            grid_normal_x.view(-1), 0,
            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        x5 = torch.index_select(
            grid_normal_x.view(-1), 0,
            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
            (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        x6 = torch.index_select(grid_normal_x.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x * y.view(-1) +
                                grid_res_x * grid_res_x *
                                (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        x7 = torch.index_select(
            grid_normal_x.view(-1), 0,
            z.view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                x.shape).unsqueeze_(2)
        x8 = torch.index_select(grid_normal_x.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x *
                                (y + 1).view(-1) + grid_res_x * grid_res_x *
                                (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        intersection_grid_normal_x = torch.cat(
            (x1, x2, x3, x4, x5, x6, x7, x8),
            2) + (1 - mask.view(width, height, 1).repeat(1, 1, 8))

        # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
        y1 = torch.index_select(
            grid_normal_y.view(-1), 0,
            z.view(-1) + grid_res_x * y.view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        y2 = torch.index_select(grid_normal_y.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x * y.view(-1) +
                                grid_res_x * grid_res_x * x.view(-1)).view(
                                    x.shape).unsqueeze_(2)
        y3 = torch.index_select(
            grid_normal_y.view(-1), 0,
            z.view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        y4 = torch.index_select(
            grid_normal_y.view(-1), 0,
            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        y5 = torch.index_select(
            grid_normal_y.view(-1), 0,
            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
            (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        y6 = torch.index_select(grid_normal_y.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x * y.view(-1) +
                                grid_res_x * grid_res_x *
                                (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        y7 = torch.index_select(
            grid_normal_y.view(-1), 0,
            z.view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                x.shape).unsqueeze_(2)
        y8 = torch.index_select(grid_normal_y.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x *
                                (y + 1).view(-1) + grid_res_x * grid_res_x *
                                (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        intersection_grid_normal_y = torch.cat(
            (y1, y2, y3, y4, y5, y6, y7, y8),
            2) + (1 - mask.view(width, height, 1).repeat(1, 1, 8))

        # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
        z1 = torch.index_select(
            grid_normal_z.view(-1), 0,
            z.view(-1) + grid_res_x * y.view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        z2 = torch.index_select(grid_normal_z.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x * y.view(-1) +
                                grid_res_x * grid_res_x * x.view(-1)).view(
                                    x.shape).unsqueeze_(2)
        z3 = torch.index_select(
            grid_normal_z.view(-1), 0,
            z.view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        z4 = torch.index_select(
            grid_normal_z.view(-1), 0,
            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
        z5 = torch.index_select(
            grid_normal_z.view(-1), 0,
            z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
            (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        z6 = torch.index_select(grid_normal_z.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x * y.view(-1) +
                                grid_res_x * grid_res_x *
                                (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        z7 = torch.index_select(
            grid_normal_z.view(-1), 0,
            z.view(-1) + grid_res_x * (y + 1).view(-1) +
            grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                x.shape).unsqueeze_(2)
        z8 = torch.index_select(grid_normal_z.view(-1), 0,
                                (z + 1).view(-1) + grid_res_x *
                                (y + 1).view(-1) + grid_res_x * grid_res_x *
                                (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        intersection_grid_normal_z = torch.cat(
            (z1, z2, z3, z4, z5, z6, z7, z8),
            2) + (1 - mask.view(width, height, 1).repeat(1, 1, 8))

        # Change from grid coordinates to world coordinates
        voxel_min_point = self.Tensor([
            bounding_box_min_x, bounding_box_min_y, bounding_box_min_z
        ]) + voxel_min_point_index * voxel_size

        intersection_pos = self.compute_intersection_pos(
            grid,
            intersection_pos_rough,
            voxel_min_point,
            voxel_min_point_index,
            ray_direction,
            voxel_size,
            mask,
            width,
            height,
        )

        intersection_pos = intersection_pos * mask.repeat(3, 1, 1).permute(
            1, 2, 0)
        shading = self.Tensor(width, height).fill_(0)

        # Compute the normal vectors for the intersecting points
        intersection_normal_x = get_intersection_normal(
            intersection_grid_normal_x, intersection_pos, voxel_min_point,
            voxel_size)
        intersection_normal_y = get_intersection_normal(
            intersection_grid_normal_y, intersection_pos, voxel_min_point,
            voxel_size)
        intersection_normal_z = get_intersection_normal(
            intersection_grid_normal_z, intersection_pos, voxel_min_point,
            voxel_size)

        # Put all the xyz-axis of the normal vectors into a single matrix
        intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
        intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
        intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
        intersection_normal = torch.cat(
            (intersection_normal_x_resize, intersection_normal_y_resize,
             intersection_normal_z_resize), 2)
        intersection_normal = intersection_normal / \
            torch.unsqueeze(torch.norm(intersection_normal,
                                    p=2, dim=2), 2).repeat(1, 1, 3)

        # Create the point light
        light_position = camera.repeat(width, height, 1)
        light_norm = torch.unsqueeze(
            torch.norm(light_position - intersection_pos, p=2, dim=2),
            2).repeat(1, 1, 3)
        light_direction_point = (light_position -
                                 intersection_pos) / light_norm

        # Create the directional light
        shading = 0
        light_direction = (camera / torch.norm(camera, p=2)).repeat(
            width, height, 1)
        l_dot_n = torch.sum(light_direction * intersection_normal,
                            2).unsqueeze_(2)
        shading += 10 * torch.max(
            l_dot_n,
            self.Tensor(width, height, 1).fill_(0))[:, :, 0] / torch.pow(
                torch.sum((light_position - intersection_pos) *
                          light_direction_point,
                          dim=2), 2)

        # Get the final image
        image = shading * mask
        image[mask == 0] = 0

        return image

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

    def sdf_diff(sdf1, sdf2):
        return torch.sum(torch.abs(sdf1 - sdf2)).item()


if __name__ == "__main__":
    sdf_clip = SDFClip()
    sdf_clip.run()