import os
import sys
import glob
import math
import cProfile
import torch
import cv2 as cv
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from .light_dirs import light_dirs
from .rl_env import Env, mesh_action_ids
from .rl_model import QNetEnhance, _n_light_axix
from .benchmark import imload_downsample_from_ids

device = torch.device("cuda:0")
# device = torch.device("cpu")
batch_size = 1
down_sample = 2

class Buffer:
  def __init__(self, size=256):
    self.size = size
    self.items = 0
    self.input_file = [None for i in range(size)]
    self.reward = [None for i in range(size)]
    self.ids = [None for i in range(size)]

  def insert(self, input_file, ids, reward):
    pos = self.items % self.size
    self.items += 1
    self.input_file[pos] = input_file
    self.reward[pos] = reward
    self.ids[pos] = ids

  def state(self, pos):
    data_loader = np.load(self.input_file[pos], allow_pickle=True)
    ids = self.ids[pos]
    _, images, _ = imload_downsample_from_ids(data_loader, ids, down_sample=down_sample)
    images = images.transpose(0, 3, 1, 2)
    reward = self.reward[pos]
    light_dirs_mode = data_loader["meta_info"].item()["light_dirs_mode"]
    state_dirs = mesh_action_ids(light_dirs(light_dirs_mode), _n_light_axix//2)[ids]
    action_ids = mesh_action_ids(light_dirs(light_dirs_mode), _n_light_axix)
    return images[:-1], state_dirs[:-1], action_ids[ids[1:]], reward, images[1:], state_dirs[1:], action_ids

  def sample(self, size, rng):
    replace = self.items < size
    sample_ids = rng.choice(np.arange(min(self.items, self.size)), size=size, replace=replace)
    state = [self.state(pos) for pos in sample_ids]
    state = [np.stack(curr_state, axis=0) for curr_state in zip(*state)]
    return state

def main():
  n_max_images = int(sys.argv[2])
  algo = sys.argv[3]
  rng = np.random.default_rng(0)
  input_files = []
  for input_path in sys.argv[1].split("+"):
    for input_file in sorted(glob.glob(os.path.join(input_path, "*.npz"))):
      input_files.append(input_file)
  # input_files = input_files[:1]
  env = Env(input_files, rng, algo=algo, down_sample=down_sample)
  env.training = True
  state, info = env.reset()
  buffer = Buffer(size=16384)
  qnet = QNetEnhance().to(device)
  qnet_target = QNetEnhance().to(device)
  qnet_target.eval()
  for target_param, param in zip(qnet_target.parameters(), qnet.parameters()):
    target_param.data.copy_(param.data)
  optimizer = optim.AdamW(qnet.parameters(), lr=1e-5, weight_decay=1e-8)
  # optimizer = optim.Adam(qnet.parameters(), lr=1e-3)
  # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000, 10000])
  train_downsample = 3
  batch_start = batch_size * n_max_images ** 2 // train_downsample
  it = 0
  errors = []
  error_min = sys.float_info.max
  env.render()
  loss_bellman = 0.0
  loss_smooth = 0.0
  # cv.waitKey(1000)
  for it_total in range(200000):
  # for it_total in range(40):
    eps = min(1.0, 2000 / (it_total + 2000 - batch_start))
    # eps = max(0.1, min(1.0, 2000 / (it_total + 2000 - batch_start)))
    with torch.no_grad():
      qnet.eval()
      x_images = state[0]
      # x_images = np.pad(state[0], ((0, n_max_images - 1 - it), (0, 0), (0, 0), (0, 0)))
      x_images = torch.tensor(x_images, dtype=torch.float32, device=device)[None, ...]
      # print("x_images", x_images.shape)
      x_dirs = state[1]
      # x_dirs = np.pad(state[1], ((0, n_max_images - 1 - it),))
      x_dirs = env.mesh_action_ids(_n_light_axix//2)[x_dirs]
      x_dirs = torch.tensor(x_dirs, dtype=torch.long, device=device)[None, ...]
      # print("x_dirs", x_dirs.shape)
      q_vals = qnet(x_images, x_dirs)[0, -1, ...].cpu().numpy()
      action = np.argmax(q_vals.mean(axis=(0, 1)).flatten()[env.mesh_action_ids(_n_light_axix)])
    sel = rng.random()
    if sel < eps:
      if os.environ.get("RLPS_ABLATION") == "3":
        action = rng.integers(env.action_space())
      else:
        action = env.action_greedy()
    action = env.action_around(action)
    next_state, reward, info = env.step(action)
    image = (np.clip(1.0 + q_vals.mean(axis=(2, 3)) / 4, 0.0, 1.0) * 255).astype(np.uint8)
    image = image.repeat(20, axis=0).repeat(20, axis=1)
    cv.imshow("q_vals_img", image)
    image = (np.clip(1.0 + q_vals.mean(axis=(0, 1)) / 2, 0.0, 1.0) * 255).astype(np.uint8)
    image = image.repeat(10, axis=0).repeat(10, axis=1)
    cv.imshow("q_vals_dir", image)
    image = (np.clip(1.0 + reward / 24, 0.0, 1.0) * 255).astype(np.uint8)
    cv.imshow("reward", image)
    it += 1
    mask = info > 0.0
    print("action % 4d" % action, \
          "reward % 10.3f" % reward[mask].mean(), \
          "loss % 12.6f % 12.6f" % (loss_bellman, loss_smooth), \
          "q_vals % 8.3f" % q_vals.min(), \
          "it_total % 6d" % it_total, "it % 4d" % it, "eps % 8.5f" % eps, "info % 8.3f" % info[mask].mean())
    state = next_state
    env.render()
    if it == n_max_images - 1:
      errors.append(info[mask].mean())
      input_file, ids, reward = env.state_compressed()
      reward = reward.reshape(n_max_images - 1, 8, 16, 8, 16).mean(axis=(2, 4))
      buffer.insert(input_file, ids, reward)
      state, info = env.reset()
      it = 0
    if it_total < batch_start or it_total % train_downsample:
      continue
    batch_state_images, batch_state_dirs, batch_action, batch_reward, \
      batch_next_images, batch_next_dirs, batch_action_ids = buffer.sample(batch_size, rng)
    batch_state_images = torch.tensor(batch_state_images, dtype=torch.float32, device=device)
    batch_state_dirs = torch.tensor(batch_state_dirs, dtype=torch.long, device=device)
    batch_next_images = torch.tensor(batch_next_images, dtype=torch.float32, device=device)
    batch_next_dirs = torch.tensor(batch_next_dirs, dtype=torch.long, device=device)
    with torch.no_grad():
      q_vals = qnet(batch_next_images, batch_next_dirs)
      q_vals = q_vals.view(batch_size, q_vals.shape[1], q_vals.shape[2], q_vals.shape[3], -1)
      # print("q_vals", q_vals.shape, "target_index", target_index.shape)
      q_vals_target = qnet_target(batch_next_images, batch_next_dirs)
      q_vals_target = q_vals.view(batch_size, q_vals.shape[1], q_vals.shape[2], q_vals.shape[3], -1)
      # print("q_vals_target", q_vals_target.shape, "target", target.shape)
      target = torch.zeros([batch_size, n_max_images - 1, 8, 8], dtype=torch.float32, device=device)
      for i_batch in range(batch_size):
        for i_image in range(n_max_images - 1):
          action_ids = torch.tensor(batch_action_ids[i_batch], dtype=torch.long, device=device)
          # print("action_ids", action_ids, "q_vals[i_batch, i_image].mean(dim=(0, 1))", q_vals[i_batch, i_image].mean(dim=(0, 1)))
          target_index = torch.argmax(q_vals[i_batch, i_image].mean(dim=(0, 1))[action_ids])
          # print("target_index", target_index, "action_ids[target_index]", action_ids[target_index])
          target[i_batch, i_image] = q_vals_target[i_batch, i_image, :, :, action_ids[target_index]]
      batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device)
      # print("target", target.shape, "batch_reward", batch_reward.shape)
      target = 0.5 * target + batch_reward
    qnet.train()
    q_vals = qnet(batch_state_images, batch_state_dirs)
    loss_smooth = 1e-8 * batch_size * n_max_images * (
      (q_vals[..., :, :-1] - q_vals[..., :, :1]).abs().mean() +
      (q_vals[..., :-1, :] - q_vals[..., 1:, :]).abs().mean())
    q_vals = q_vals.view(batch_size, n_max_images - 1, 8, 8, -1)
    loss_bellman = 0.0
    for i_batch in range(batch_size):
      for i_image in range(n_max_images - 1):
        action_ids = torch.tensor(batch_action_ids[i_batch], dtype=torch.long, device=device)
        weight = (1.0 - 0.9 ** (i_image + 1))
        diff = q_vals[i_batch, i_image, :, :, batch_action[i_batch, i_image]] - target[i_batch, i_image]
        loss_bellman += (weight * diff).pow(2).mean()
    loss = loss_bellman + loss_smooth
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()
    cv.waitKey(10)
    if it_total // train_downsample % 1000 == 0:
      error_curr = np.mean(errors[-100:])
      if True or it_total > 20000 and error_min > error_curr:
        print("save model at iter % 4d, error: % 10.6f" % (it_total, error_curr))
        error_min = error_curr
        torch.save({"it_total": it_total, "qnet": qnet.state_dict(), "optimizer": optimizer.state_dict()}, sys.argv[4])
      else:
        print("skip model at iter % 4d, error: % 10.6f, error_min: % 10.6f" % (it_total, error_curr, error_min))
      for target_param, param in zip(qnet_target.parameters(), qnet.parameters()):
        target_param.data.copy_(param.data)

if __name__ == "__main__":
  # cProfile.run("main()")
  main()
  sys.exit(0)
