import os
import math
import cv2 as cv
import numpy as np
from .light_dirs import light_dirs
from .benchmark import imload_downsample_from_ids, benchmark_from_ids, action_greedy

def mesh_action_ids(all_light_dirs, mesh_size, dtype=np.float32):
  mesh_x = np.linspace(-1.0, 1.0, num=mesh_size, endpoint=True, dtype=dtype)
  mesh_y = np.linspace(-1.0, 1.0, num=mesh_size, endpoint=True, dtype=dtype)
  mesh_x, mesh_y = np.meshgrid(mesh_x, mesh_y, indexing="ij")
  meshgrid = np.stack([mesh_x.flatten(), mesh_y.flatten()], axis=1)
  dist = np.linalg.norm(all_light_dirs[:, None, :2] - meshgrid[None, :, :], ord=2, axis=2)
  ids = dist.argmin(axis=1)
  return ids

class Env:
  def __init__(self, input_files, rng, algo, down_sample=2):
    self.input_files = input_files
    self.down_sample = down_sample
    self.rng = rng
    self.algo = algo
    self.images = []
    self.normal = []
    self.reward = []
    self.training = False
    self.input_file = None
    self.data_loader = None
    self.all_light_dirs = None

  def action_space(self):
    return len(self.all_light_dirs)

  def mesh_action_ids(self, mesh_size=24, dtype=np.float32):
    return mesh_action_ids(self.all_light_dirs, mesh_size, dtype)

  def action_greedy(self):
    return action_greedy(self.data_loader, self.ids, start=self.rng.integers(7), step=7)

  def action_around(self, action):
    dist = np.sum(self.all_light_dirs * self.all_light_dirs[None, action, :], axis=1)
    p = np.power(dist.clip(min=0.0), 32)
    p/= p.sum()
    # print("dist", dist.min(), "p", p.max())
    return self.rng.choice(len(self.all_light_dirs), p=p)

  def benchmark(self):
    error, error_mean, normal = benchmark_from_ids(self.data_loader, self.ids, self.algo, self.down_sample)
    return error, error_mean, normal

  def state_compressed(self):
    # return self.input_file, self.ids, np.array(self.normal), np.array(self.reward)
    return self.input_file, self.ids, np.array(self.reward)

  def state(self, dtype=np.float32):
    images = self.images.astype(dtype).transpose(0, 3, 1, 2)
    # normal_pred = self.normal_pred.transpose(2, 0, 1)
    # normal_gt = self.normal_gt.transpose(2, 0, 1)
    # depth_gt = self.depth[np.newaxis, ...] * 0.1
    # normal = np.concatenate([normal_pred, normal_gt, depth_gt], axis=0)
    # normal = normal.reshape(normal.shape[0], res_x, self.down_sample, res_y, self.down_sample)
    # normal = np.mean(normal, axis=(2, 4))
    # return images, normal, self.ids
    return images, self.ids

  def reset(self, scene_id=None):
    if scene_id is None:
      self.input_file = self.rng.choice(self.input_files)
    else:
      self.input_file = self.input_files[scene_id]
    self.data_loader = np.load(self.input_file, allow_pickle=True)
    self.all_light_dirs = light_dirs(self.data_loader["meta_info"].item()["light_dirs_mode"])
    self.ids = [np.argmax(self.all_light_dirs[:, 2])]
    self.normal_gt, self.images, _ = imload_downsample_from_ids(self.data_loader, self.ids, self.down_sample)
    self.error, _, self.normal_pred = self.benchmark()
    _, self.error_all, _ = benchmark_from_ids(self.data_loader, range(len(self.all_light_dirs)), self.algo, self.down_sample)
    # self.error = np.clip(self.error, 0.0, 90.0)
    self.normal = [self.normal_pred]
    self.reward = []
    print("input_file", self.input_file, "error_all", self.error_all)
    return self.state(), self.error

  def step(self, action):
    self.ids.append(action)
    _, self.images, _ = imload_downsample_from_ids(self.data_loader, self.ids, self.down_sample)
    new_error, error_mean, self.normal_pred = self.benchmark()
    # new_error = np.clip(new_error, 0.0, 90.0)
    # reward = (self.error - new_error)

    # reward = np.zeros_like(new_error)
    # reward[new_error > 0.] = np.clip(10.0 / (new_error[new_error > 0.] + 10.0), 0.0, 1.0)
    # if action in self.ids[:-1]:
    #   reward *= 0.0

    weight = (1.0 - 0.9 ** (len(self.ids) - 1)) * 10.0 / (self.error_all + 10.0)
    reward = weight * (0.5 * self.error - new_error)
    if os.environ.get("RLPS_ABLATION") == "2":
      reward = -new_error if len(self.ids) == 20 else np.zeros_like(new_error)
    if os.environ.get("RLPS_ABLATION") == "5":
      reward = -new_error
    self.error = new_error
    self.normal.append(self.normal_pred)
    self.reward.append(reward)
    return self.state(), reward, error_mean

  def putText(image, text, center, color, resolution=256):
    fontScale = resolution * 0.0008
    thickness = int(resolution * 0.001)
    (width, height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
    center[0] -= width / 2
    center[1] += height / 2
    cv.putText(image, text, center, cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

  def render(self, resolution=256):
    cv.imshow("Last Image", self.images[-1])
    cv.imshow("Normal Ground Truth", self.normal_gt * 0.5 + 0.5)
    cv.imshow("Normal Predict", self.normal_pred * 0.5 + 0.5)
    image = np.full([resolution, resolution, 3], 255, dtype=np.uint8)
    nodeRadius = int(resolution * 0.015)
    n_light_axix = int(math.sqrt(self.action_space()))
    # print(self.action_space(), n_light_axix)
    for light_dir in self.all_light_dirs:
      light_dir = light_dir[[1, 0]]
      nodeCenter = (resolution * (0.5 + 0.45 * light_dir)).astype(int)
      cv.circle(image, nodeCenter, nodeRadius, (0, 0, 0), -1)
    for i, curr_id in enumerate(self.ids):
      color = (0, 0, 255) if i == len(self.ids) - 1 else (0, 255, 0)
      light_dir = self.all_light_dirs[curr_id, [1, 0]]
      nodeCenter = (resolution * (0.5 + 0.45 * light_dir)).astype(int)
      cv.circle(image, nodeCenter, nodeRadius, color, -1)
    cv.imshow("Selected Light", image)
    cv.waitKey(10)

def main():
  for scene_id in range(100):
    rng = np.random.default_rng(scene_id)
    env = Env(["data/blobs_render_test/%06d.npz" % scene_id], rng)
    state = env.reset()
    env.render()
    for i in range(30):
      action = rng.integers(env.action_space())
      state, reward, error = env.step(action)
      print("reward", reward.mean(), "error", error[error > 0].mean())
      env.render()

if __name__ == "__main__":
  main()
