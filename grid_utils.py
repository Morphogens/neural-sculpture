import math

import torch


def grid_construction_sphere_big(
    grid_res: int,
    bounding_box_min: int,
    bounding_box_max: int,
    device: str = 'cuda',
):
    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)

    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)

    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1.6

    return grid.to(device)


def get_grid_normal(
    grid: torch.Tensor,
    voxel_size: float,
    grid_res_x: int,
    grid_res_y: int,
    grid_res_z: int,
):
    # largest index
    n_x = grid_res_x - 1
    n_y = grid_res_y - 1
    n_z = grid_res_z - 1

    # x-axis normal vectors
    x_norm_1 = torch.cat(
        (
            grid[1:, :, :],
            (3 * grid[n_x, :, :] - 3 * grid[n_x - 1, :, :] +
             grid[n_x - 2, :, :]).unsqueeze_(0),
        ),
        dim=0,
    )
    x_norm_2 = torch.cat(
        (
            (-3 * grid[1, :, :] + 3 * grid[0, :, :] +
             grid[2, :, :]).unsqueeze_(0),
            grid[:n_x, :, :],
        ),
        dim=0,
    )
    grid_normal_x = (x_norm_1 - x_norm_2) / (2 * voxel_size)

    # y-axis normal vectors
    y_norm_1 = torch.cat(
        (
            grid[:, 1:, :],
            (3 * grid[:, n_y, :] - 3 * grid[:, n_y - 1, :] +
             grid[:, n_y - 2, :]).unsqueeze_(1),
        ),
        dim=1,
    )
    y_norm_2 = torch.cat(
        (
            (-3 * grid[:, 1, :] + 3 * grid[:, 0, :] +
             grid[:, 2, :]).unsqueeze_(1),
            grid[:, :n_y, :],
        ),
        dim=1,
    )
    grid_normal_y = (y_norm_1 - y_norm_2) / (2 * voxel_size)

    # z-axis normal vectors
    z_norm_1 = torch.cat(
        (
            grid[:, :, 1:],
            (3 * grid[:, :, n_z] - 3 * grid[:, :, n_z - 1] +
             grid[:, :, n_z - 2]).unsqueeze_(2),
        ),
        dim=2,
    )
    z_norm_2 = torch.cat(
        (
            (-3 * grid[:, :, 1] + 3 * grid[:, :, 0] +
             grid[:, :, 2]).unsqueeze_(2),
            grid[:, :, :n_z],
        ),
        dim=2,
    )
    grid_normal_z = (z_norm_1 - z_norm_2) / (2 * voxel_size)

    return [grid_normal_x, grid_normal_y, grid_normal_z]


def get_intersection_normal(
    intersection_grid_normal,
    intersection_pos,
    voxel_min_point,
    voxel_size,
):

    # Compute parameters
    tx = (intersection_pos[:, :, 0] - voxel_min_point[:, :, 0]) / voxel_size
    ty = (intersection_pos[:, :, 1] - voxel_min_point[:, :, 1]) / voxel_size
    tz = (intersection_pos[:, :, 2] - voxel_min_point[:, :, 2]) / voxel_size

    intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 0] \
        + tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 1] \
        + (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:, :, 2] \
        + tz * ty * (1 - tx) * intersection_grid_normal[:, :, 3] \
        + (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:, :, 4] \
        + tz * (1 - ty) * tx * intersection_grid_normal[:, :, 5] \
        + (1 - tz) * ty * tx * intersection_grid_normal[:, :, 6] \
        + tz * ty * tx * intersection_grid_normal[:, :, 7]

    return intersection_normal


# def grid_construction_cube(
#     grid_res,
#     bounding_box_min,
#     bounding_box_max,
#     device='cuda',
# ):

#     # Construct the sdf grid for a cube with size 2
#     voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
#     cube_left_bound_index = float(grid_res - 1) / 4
#     cube_right_bound_index = float(grid_res - 1) / 4 * 3
#     cube_center = float(grid_res - 1) / 2

#     grid = torch.tensor(grid_res, grid_res, grid_res).to(device)
#     for i in range(grid_res):
#         for j in range(grid_res):
#             for k in range(grid_res):
#                 if (i >= cube_left_bound_index and i <= cube_right_bound_index
#                         and j >= cube_left_bound_index
#                         and j <= cube_right_bound_index
#                         and k >= cube_left_bound_index
#                         and k <= cube_right_bound_index):
#                     grid[i, j, k] = voxel_size * \
#                         max(abs(i - cube_center), abs(j - cube_center),
#                             abs(k - cube_center)) - 1
#                 else:
#                     grid[i, j, k] = math.sqrt(
#                         pow(
#                             voxel_size *
#                             (max(i - cube_right_bound_index,
#                                  cube_left_bound_index - i, 0)), 2) + pow(
#                                      voxel_size *
#                                      (max(j - cube_right_bound_index,
#                                           cube_left_bound_index - j, 0)), 2) +
#                         pow(
#                             voxel_size *
#                             (max(k - cube_right_bound_index,
#                                  cube_left_bound_index - k, 0)), 2))
#     return grid

# def grid_construction_torus(
#     grid_res,
#     bounding_box_min,
#     bounding_box_max,
#     device='cuda',
# ):

#     # radius of the circle between the two circles
#     radius_big = 1.5

#     # radius of the small circle
#     radius_small = 0.5

#     voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
#     grid = torch.tensor(grid_res, grid_res, grid_res).to(device)
#     for i in range(grid_res):
#         for j in range(grid_res):
#             for k in range(grid_res):
#                 x = bounding_box_min + voxel_size * i
#                 y = bounding_box_min + voxel_size * j
#                 z = bounding_box_min + voxel_size * k

#                 grid[i, j, k] = math.sqrt(
#                     math.pow((math.sqrt(math.pow(y, 2) + math.pow(z, 2)) -
#                               radius_big), 2) + math.pow(x, 2)) - radius_small

#     return grid

# def grid_construction_sphere_small(
#     grid_res,
#     bounding_box_min,
#     bounding_box_max,
#     device='cuda',
# ):

#     # Construct the sdf grid for a sphere with radius 1
#     linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
#     x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
#     y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
#     z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
#     grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1
#     return grid.to(device)
