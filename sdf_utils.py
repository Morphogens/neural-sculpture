from typing import *

import torch

import renderer
from grid_utils import get_grid_normal, get_intersection_normal

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# def sdf_diff(sdf1, sdf2):
#     return torch.sum(torch.abs(sdf1 - sdf2)).item()


def calculate_sdf_value(
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

    if device == 'cuda':
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
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1]

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
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1]

        # Linear interpolate along the y axis
        ty = (points[:, 1] - voxel_min_point[:, 1]) / voxel_size
        c0 = (1 - ty) * c01 + ty * c23
        c1 = (1 - ty) * c45 + ty * c67

        # Return final value interpolated along z
        tz = (points[:, 2] - voxel_min_point[:, 2]) / voxel_size

    result = (1 - tz) * c0 + tz * c1

    return result


def compute_intersection_pos(
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

    if device == 'cuda':

        x = voxel_min_point_index.long()[:, :, 0]
        y = voxel_min_point_index.long()[:, :, 1]
        z = voxel_min_point_index.long()[:, :, 2]

        c01 = (1 - tx) * grid[x, y, z].cuda() + tx * grid[x + 1, y, z].cuda()
        c23 = (1 - tx) * grid[x, y + 1, z].cuda() + tx * grid[x + 1, y + 1,
                                                              z].cuda()
        c45 = (1 - tx) * grid[x, y, z + 1].cuda() + tx * grid[x + 1, y,
                                                              z + 1].cuda()
        c67 = (1 - tx) * grid[x, y+1, z+1].cuda() + \
            tx * grid[x+1, y+1, z+1].cuda()

    else:
        x = voxel_min_point_index.numpy()[:, :, 0]
        y = voxel_min_point_index.numpy()[:, :, 1]
        z = voxel_min_point_index.numpy()[:, :, 2]

        c01 = (1 - tx) * grid[x, y, z] + tx * grid[x + 1, y, z]
        c23 = (1 - tx) * grid[x, y + 1, z] + tx * grid[x + 1, y + 1, z]
        c45 = (1 - tx) * grid[x, y, z + 1] + tx * grid[x + 1, y, z + 1]
        c67 = (1 - tx) * grid[x, y + 1, z + 1] + tx * grid[x + 1, y + 1, z + 1]

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
    grid: torch.Tensor,
    camera,
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
    w_h_3 = torch.zeros(width, height, 3).to(device)
    w_h = torch.zeros(width, height).to(device)

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
    intersection_grid_normal_x = Tensor(width, height, 8)
    intersection_grid_normal_y = Tensor(width, height, 8)
    intersection_grid_normal_z = Tensor(width, height, 8)
    intersection_grid = Tensor(width, height, 8)

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:, :, 0] != -1).type(Tensor)

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
    x2 = torch.index_select(
        grid_normal_x.view(-1), 0, (z + 1).view(-1) + grid_res_x * y.view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x3 = torch.index_select(
        grid_normal_x.view(-1), 0,
        z.view(-1) + grid_res_x * (y + 1).view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
                            grid_res_x * grid_res_x * x.view(-1)).view(
                                x.shape).unsqueeze_(2)
    x5 = torch.index_select(
        grid_normal_x.view(-1), 0,
        z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
        (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid_normal_x.view(-1), 0, (z + 1).view(-1) +
                            grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
                            (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(
        grid_normal_x.view(-1), 0,
        z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x *
        (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid_normal_x.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
                            grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                                x.shape).unsqueeze_(2)
    intersection_grid_normal_x = torch.cat(
        (x1, x2, x3, x4, x5, x6, x7, x8),
        2) + (1 - mask.view(width, height, 1).repeat(1, 1, 8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(
        grid_normal_y.view(-1), 0,
        z.view(-1) + grid_res_x * y.view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y2 = torch.index_select(
        grid_normal_y.view(-1), 0, (z + 1).view(-1) + grid_res_x * y.view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y3 = torch.index_select(
        grid_normal_y.view(-1), 0,
        z.view(-1) + grid_res_x * (y + 1).view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y4 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
                            grid_res_x * grid_res_x * x.view(-1)).view(
                                x.shape).unsqueeze_(2)
    y5 = torch.index_select(
        grid_normal_y.view(-1), 0,
        z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
        (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    y6 = torch.index_select(grid_normal_y.view(-1), 0, (z + 1).view(-1) +
                            grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
                            (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    y7 = torch.index_select(
        grid_normal_y.view(-1), 0,
        z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x *
        (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    y8 = torch.index_select(grid_normal_y.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
                            grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                                x.shape).unsqueeze_(2)
    intersection_grid_normal_y = torch.cat(
        (y1, y2, y3, y4, y5, y6, y7, y8),
        2) + (1 - mask.view(width, height, 1).repeat(1, 1, 8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(
        grid_normal_z.view(-1), 0,
        z.view(-1) + grid_res_x * y.view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z2 = torch.index_select(
        grid_normal_z.view(-1), 0, (z + 1).view(-1) + grid_res_x * y.view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z3 = torch.index_select(
        grid_normal_z.view(-1), 0,
        z.view(-1) + grid_res_x * (y + 1).view(-1) +
        grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z4 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
                            grid_res_x * grid_res_x * x.view(-1)).view(
                                x.shape).unsqueeze_(2)
    z5 = torch.index_select(
        grid_normal_z.view(-1), 0,
        z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
        (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    z6 = torch.index_select(grid_normal_z.view(-1), 0, (z + 1).view(-1) +
                            grid_res_x * y.view(-1) + grid_res_x * grid_res_x *
                            (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    z7 = torch.index_select(
        grid_normal_z.view(-1), 0,
        z.view(-1) + grid_res_x * (y + 1).view(-1) + grid_res_x * grid_res_x *
        (x + 1).view(-1)).view(x.shape).unsqueeze_(2)
    z8 = torch.index_select(grid_normal_z.view(-1), 0,
                            (z + 1).view(-1) + grid_res_x * (y + 1).view(-1) +
                            grid_res_x * grid_res_x * (x + 1).view(-1)).view(
                                x.shape).unsqueeze_(2)
    intersection_grid_normal_z = torch.cat(
        (z1, z2, z3, z4, z5, z6, z7, z8),
        2) + (1 - mask.view(width, height, 1).repeat(1, 1, 8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor([
        bounding_box_min_x, bounding_box_min_y, bounding_box_min_z
    ]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(
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

    intersection_pos = intersection_pos * mask.repeat(3, 1, 1).permute(1, 2, 0)
    shading = Tensor(width, height).fill_(0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x,
                                                    intersection_pos,
                                                    voxel_min_point,
                                                    voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y,
                                                    intersection_pos,
                                                    voxel_min_point,
                                                    voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z,
                                                    intersection_pos,
                                                    voxel_min_point,
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
    light_direction_point = (light_position - intersection_pos) / light_norm

    # Create the directional light
    # shading = 0
    light_direction = (camera / torch.norm(camera, p=2)).repeat(
        width, height, 1)
    l_dot_n = torch.sum(light_direction * intersection_normal, 2).unsqueeze_(2)
    shading += 10 * torch.max(
        l_dot_n,
        Tensor(width, height, 1).fill_(0))[:, :, 0] / torch.pow(
            torch.sum(
                (light_position - intersection_pos) * light_direction_point,
                dim=2), 2)

    # Get the final image
    image = shading * mask
    image[mask == 0] = 0
    image = shading / shading.max()

    return image
