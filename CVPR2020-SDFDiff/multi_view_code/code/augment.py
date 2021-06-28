import sys, random
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformations from https://github.com/reiinakano/neural-painters-pytorch
class RandomScale(nn.Module):
  """Module for randomly scaling an image"""
  def __init__(self, scales):
    """
    :param scales: list of scales to randomly choose from e.g. [0.8, 1.0, 1.2] will randomly scale an image by
      0.8, 1.0, or 1.2
    """
    super(RandomScale, self).__init__()

    self.scales = scales

  def forward(self, x: torch.Tensor):
    scale = self.scales[random.randint(0, len(self.scales)-1)]
    return F.interpolate(x, scale_factor=scale, mode='bilinear')


class RandomCrop(nn.Module):
  """Module for randomly cropping an image"""
  def __init__(self, size: int):
    """
    :param size: How much to crop from both sides. e.g. 8 will remove 8 pixels in both x and y directions.
    """
    super(RandomCrop, self).__init__()
    self.size = size

  def forward(self, x: torch.Tensor):
    batch_size, _, h, w = x.shape
    h_move = random.randint(0, self.size)
    w_move = random.randint(0, self.size)
    return x[:, :, h_move:h-self.size+h_move, w_move:w-self.size+w_move]


class RandomRotate(nn.Module):
  """Module for randomly rotating an image"""
  def __init__(self, angle=10, same_throughout_batch=False):
    """
    :param angle: Angle in degrees
    :param same_throughout_batch: Degree of rotation, although random, is kept the same throughout a single batch.
    """
    super(RandomRotate, self).__init__()
    self.angle=angle
    self.same_throughout_batch = same_throughout_batch

  def forward(self, img: torch.tensor):
    b, _, h, w = img.shape
    # create transformation (rotation)
    if not self.same_throughout_batch:
      angle = torch.randn(b, device=img.device) * self.angle
    else:
      angle = torch.randn(1, device=img.device) * self.angle
      angle = angle.repeat(b)
    center = torch.ones(b, 2, device=img.device)
    center[..., 0] = img.shape[3] / 2  # x
    center[..., 1] = img.shape[2] / 2  # y
    # define the scale factor
    scale = torch.ones(b, 2, device=img.device)
    M = kornia.get_rotation_matrix2d(center, angle, scale)
    img_warped = kornia.warp_affine(img, M, dsize=(h, w))
    return img_warped

# Define image augmentations
padder = nn.ConstantPad2d(12, 0.5)
rand_crop_8 = RandomCrop(8)
rand_scale = RandomScale([1 + (i-5)/50. for i in range(11)])
random_rotater = RandomRotate(angle=5, same_throughout_batch=False)
rand_crop_4 = RandomCrop(4)

def augment(img, n=1):
    imgs = img.repeat(n, 1, 1, 1)
    return rand_crop_4(random_rotater(rand_crop_8(rand_scale(padder(imgs)))))