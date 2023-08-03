import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_n_light_axix = 24
_activation = nn.ReLU()
_pool = nn.MaxPool2d(2)
_padding_mode = "reflect"

def _weight_norm(module):
  nn.init.orthogonal_(module.weight)
  module.bias.data.zero_()
  return nn.utils.weight_norm(module)

class ImageConv(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_a1 = _weight_norm(nn.Conv2d(3, 32, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a2 = _weight_norm(nn.Conv2d(32, 64, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a3 = _weight_norm(nn.Conv2d(64, 64, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a4 = _weight_norm(nn.Conv2d(64, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a5 = _weight_norm(nn.Conv2d(128, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a6 = _weight_norm(nn.Conv2d(128, 256, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a7 = _weight_norm(nn.Conv2d(256, 256, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a8 = _weight_norm(nn.Conv2d(256, 256, 3, padding=1, padding_mode=_padding_mode))

  def forward(self, x, debug=False):
    if debug:
      print("x", x.shape)
    n_batch = x.shape[0]
    n_max_images = x.shape[1]
    x = x.view(n_batch * n_max_images, x.shape[2], x.shape[3], x.shape[4])
    if debug:
      print("x", x.shape)
    x = _pool(_activation(self.conv_a1(x)))
    x = _activation(self.conv_a2(x))
    x = _pool(_activation(self.conv_a3(x)))
    x = _activation(self.conv_a4(x))
    x = _pool(_activation(self.conv_a5(x)))
    x = _activation(self.conv_a6(x))
    x = _pool(_activation(self.conv_a7(x)))
    x = _activation(self.conv_a8(x))
    img_size = x.shape[2]
    img_channel = x.shape[1]
    x = x.view(n_batch, n_max_images, img_channel, img_size, img_size)
    return x

class ANet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_b1 = _weight_norm(nn.Conv2d(64, 64, 3, padding=1, padding_mode=_padding_mode))
    self.conv_b2 = _weight_norm(nn.Conv2d(64, 64, 3, padding=1, padding_mode=_padding_mode))
    self.conv_b3 = _weight_norm(nn.Conv2d(64, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_b4 = _weight_norm(nn.Conv2d(128, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_b5 = _weight_norm(nn.Conv2d(128, 256, 3, padding=1, padding_mode=_padding_mode))
    self.conv_b6 = _weight_norm(nn.Conv2d(256, 256, 3, padding=1, padding_mode=_padding_mode))
    self.conv_trans_b1 = _weight_norm(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))
    self.conv_c1 = _weight_norm(nn.Conv2d(256, 256, 3, padding=1, padding_mode=_padding_mode))
    self.conv_c2 = _weight_norm(nn.Conv2d(256, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_trans_b2 = _weight_norm(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
    self.conv_c3 = _weight_norm(nn.Conv2d(128, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_c4 = _weight_norm(nn.Conv2d(128, 128, 3, padding=1, padding_mode=_padding_mode))
    self.conv_trans_b3 = _weight_norm(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
    self.conv_c5 = _weight_norm(nn.Conv2d(64, 64, 3, padding=1, padding_mode=_padding_mode))
    self.conv_c6 = _weight_norm(nn.Conv2d(64, 1, 3, padding=1, padding_mode=_padding_mode))

  def forward(self, images, light_dirs, debug=False):
    images = images.permute(0, 1, 3, 4, 2)
    if debug:
      print("images", images.shape)
    n_batch, n_max_images, img_size, _, _ = images.shape
    img_channel = 64
    x = torch.zeros(images.shape[:-1] + (img_channel, (_n_light_axix//2)**2), device=images.device)
    if debug:
      print("light_dirs", light_dirs)
    for i_image in range(n_max_images):
      x[:, i_image, :, :, 0, :] = 1.0 - 0.9 ** (i_image + 1)
    for i_batch in range(n_batch):
      for i_image in range(n_max_images):
        x[i_batch, i_image:, :, :, :, light_dirs[i_batch, i_image]] += \
          images[None, i_batch, i_image, :, :, :img_channel]
    if debug:
      print("x", x.shape, x.min(), x.max())
    x = x.view(n_batch * n_max_images * img_size * img_size, img_channel, _n_light_axix//2, _n_light_axix//2)
    x = _activation(self.conv_b1(x))
    x = _activation(self.conv_b2(x))
    x_1 = x
    if debug:
      print("x_1", x_1.shape)
    x = _pool(x)
    x = _activation(self.conv_b3(x))
    x = _activation(self.conv_b4(x))
    x_2 = x
    if debug:
      print("x_2", x_2.shape)
    x = _pool(x)
    x = _activation(self.conv_b5(x))
    x = _activation(self.conv_b6(x))
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_trans_b1(x))
    if debug:
      print("x", x.shape)
    x = torch.cat([x_2, x], dim=1)
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_c1(x))
    x = _activation(self.conv_c2(x))
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_trans_b2(x))
    if debug:
      print("x", x.shape)
    x = torch.cat([x_1, x], dim=1)
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_c3(x))
    x = _activation(self.conv_c4(x))
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_trans_b3(x))
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_c5(x))
    x = self.conv_c6(x)
    if debug:
      print("x", x.shape)
    # x = x.view(n_batch, n_max_images, img_size, img_size, _n_light_axix *_n_light_axix)
    # for i_batch in range(n_batch):
    #   for i_image in range(n_max_images):
    #     x.data[i_batch, i_image:, :, :, light_dirs[i_batch, i_image]] = 0.
    x = x.view(n_batch, n_max_images, img_size, img_size, _n_light_axix, _n_light_axix)
    if debug:
      print("x", x.shape)
    return x

class VNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_a1 = _weight_norm(nn.Conv2d(512, 512, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a2 = _weight_norm(nn.Conv2d(512, 512, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a3 = _weight_norm(nn.Conv2d(512, 512, 3, padding=1, padding_mode=_padding_mode))
    self.conv_a4 = _weight_norm(nn.Conv2d(512, 1, 3, padding=1, padding_mode=_padding_mode))

  def forward(self, images, debug=False):
    n_batch, n_max_images, img_channel, img_size, _ = images.shape
    if debug:
      print("images", images.shape)
    x = torch.zeros([n_batch, n_max_images, img_channel*2, img_size, img_size], device=images.device)
    if debug:
      print("x", x.shape)
    for i_image in range(n_max_images):
      x[:, i_image, :img_channel, :, :] = images[:, :(i_image+1), :, :, :].mean(dim=1)
      x[:, i_image, img_channel:, :, :] = images[:, :(i_image+1), :, :, :].max(dim=1)[0]
      x[:, i_image, 0, :, :] = 1.0 - 0.9 ** (i_image + 1)
    if debug:
      print("x", x.shape)
    x = x.view(n_batch * n_max_images, img_channel*2, img_size, img_size)
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_a1(x))
    x = _activation(self.conv_a2(x))
    x = _activation(self.conv_a3(x))
    x = self.conv_a4(x)
    if debug:
      print("x", x.shape)
    x = x.view(n_batch, n_max_images, img_size, img_size)
    if debug:
      print("x", x.shape)
    return x

class QNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.image_conv = ImageConv()
    self.anet = ANet()
    self.vnet = VNet()

  def forward(self, images, light_dirs, debug=False):
    images = self.image_conv(images)
    if debug:
      print("images", images.shape)
    x_a = self.anet(images, light_dirs)
    if debug:
      print("x_a", x_a.shape)
    x_v = self.vnet(images)
    if debug:
      print("x_v", x_v.shape)
    x = x_a - x_a.mean(dim=(-2, -1), keepdim=True) + x_v[..., None, None]
    return x

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc_a1 = nn.Linear(19 * (16 * 16 + 2), 32)
    self.fc_a2 = nn.Linear(32, 19 * 8 * 8 * 24 * 24)

  def forward(self, images_orig, light_dirs_orig, debug=False):
    n_batch = images_orig.shape[0]
    n_images = images_orig.shape[1]
    device=images_orig.device
    images_orig = images_orig.mean(dim=2)
    light_dirs_orig = torch.stack([light_dirs_orig // 24, light_dirs_orig % 24], dim=-1) / 23
    if debug:
      print("images_orig", images_orig.shape, "light_dirs_orig", light_dirs_orig.shape, light_dirs_orig.dtype)
    images = torch.zeros([n_batch, 19, 16, 16], device=device)
    images[:, :n_images] = images_orig[..., ::8, ::8]
    light_dirs = torch.zeros([n_batch, 19, 2], device=device)
    light_dirs[:, :n_images] = light_dirs_orig
    x = torch.cat([images.view(n_batch, -1), light_dirs.view(n_batch, -1)], dim=-1)
    if debug:
      print("x", x.shape)
    x = _activation(self.fc_a1(x))
    if debug:
      print("x", x.shape)
    x = self.fc_a2(x)
    if debug:
      print("x", x.shape)
    x = x.view(n_batch, 19, 8, 8, 24, 24)
    if debug:
      print("x", x.shape)
    return x

class QNetEnhance(nn.Module):
  def __init__(self):
    super().__init__()
    if os.environ.get("RLPS_ABLATION") == "4":
      self.qnet = MLP()
    else:
      self.qnet = QNet()
    self.grad_id = 0

  def transform_to(self, images, light_dirs, transform_id):
    light_dirs_x = light_dirs // (_n_light_axix // 2)
    light_dirs_y = light_dirs % (_n_light_axix // 2)
    if transform_id == 0:
      return images, light_dirs
    elif transform_id == 1:
      light_dirs_x, light_dirs_y = light_dirs_x, (_n_light_axix // 2) - 1 - light_dirs_y
      return images.flip(dims=[4]), light_dirs_x * (_n_light_axix // 2) + light_dirs_y
    elif transform_id == 2:
      light_dirs_x, light_dirs_y = (_n_light_axix // 2) - 1 - light_dirs_x, light_dirs_y
      return images.flip(dims=[3]), light_dirs_x * (_n_light_axix // 2) + light_dirs_y
    elif transform_id == 3:
      light_dirs_x, light_dirs_y = (_n_light_axix // 2) - 1 - light_dirs_x, (_n_light_axix // 2) - 1 - light_dirs_y
      return images.flip(dims=[3, 4]), light_dirs_x * (_n_light_axix // 2) + light_dirs_y
    elif transform_id >= 4 and transform_id < 8:
      light_dirs_x, light_dirs_y = light_dirs_y, light_dirs_x
      light_dirs = light_dirs_x * (_n_light_axix // 2) + light_dirs_y
      return self.transform_to(images.transpose(3, 4), light_dirs, transform_id - 4)

  def transform_from(self, q_vals, transform_id):
    if transform_id == 0:
      return q_vals
    elif transform_id == 1:
      return q_vals.flip(dims=[3, 5])
    elif transform_id == 2:
      return q_vals.flip(dims=[2, 4])
    elif transform_id == 3:
      return q_vals.flip(dims=[2, 3, 4, 5])
    elif transform_id >= 4 and transform_id < 8:
      q_vals = self.transform_from(q_vals, transform_id - 4)
      return q_vals.transpose(2, 3).transpose(4, 5)

  def forward(self, images_orig, light_dirs_orig, debug=False):
    n_batch = images_orig.shape[0]
    if debug:
      print("images_orig", images_orig.shape, "light_dirs_orig", light_dirs_orig.shape)
    with torch.no_grad():
      transform_range = 1 if os.environ.get("RLPS_ABLATION") == "4" else 8
      state = [self.transform_to(images_orig, light_dirs_orig, transform_id) for transform_id in range(transform_range)]
      images, light_dirs = [torch.cat(curr_state) for curr_state in zip(*state)]
      if debug:
        print("images", images.shape, "light_dirs", light_dirs.shape)
      q_vals = self.qnet(images, light_dirs)
      q_vals = q_vals.view([transform_range, n_batch] + list(q_vals.shape[1:]))
    state = [self.transform_to(images_orig, light_dirs_orig, self.grad_id)]
    images, light_dirs = [torch.cat(curr_state) for curr_state in zip(*state)]
    if debug:
      print("images", images.shape, "light_dirs", light_dirs.shape)
    q_vals = q_vals.clone()
    q_vals[self.grad_id] = self.qnet(images, light_dirs)
    self.grad_id = (self.grad_id + 1) % transform_range
    q_vals = torch.stack([self.transform_from(q_vals[i, ...], i) for i in range(transform_range)])
    q_vals = q_vals.mean(dim=0)
    if debug:
      print("q_vals", q_vals.shape)
    return q_vals
