import io
import os
import sys
import glob
import math
import time
import zipfile
from functools import partial
import torch
import cv2 as cv
import numpy as np
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "true"
import torch.optim as optim
import torch.nn.functional as F
from .rl_env import Env
from .benchmark import _all_algos, display_normal, display_normal_err
from .rl_model import QNetEnhance, _n_light_axix

# torch.use_deterministic_algorithms(True)
# device = torch.device("cpu")
device = torch.device("cuda:0")

def q_vals_dir_norm(q_vals, ids, up_sample=8):
  q_vals = q_vals.mean(axis=(0, 1))
  axis = np.linspace(-1, 1, _n_light_axix)
  q_vals_mask = np.hypot(*np.meshgrid(axis, axis)) < 0.99
  # print("q_vals_mask", q_vals_mask.shape, q_vals_mask.mean())
  q_vals[~q_vals_mask] = q_vals[q_vals_mask].min()
  q_vals_dir_norm = (q_vals - q_vals[q_vals_mask].mean()) / (q_vals - q_vals[q_vals_mask].mean()).max()
  dsize = (q_vals_dir_norm.shape[0] * up_sample, q_vals_dir_norm.shape[1] * up_sample)
  q_vals_dir_norm = np.clip(cv.resize(q_vals_dir_norm, dsize, interpolation=cv.INTER_NEAREST) * 0.5 + 0.5, 0.0, 1.0)
  # print("q_vals_dir_norm", q_vals_dir_norm.min(), q_vals_dir_norm.max())
  axis = np.linspace(-1, 1, _n_light_axix * up_sample)
  q_vals_mask = (np.hypot(*np.meshgrid(axis, axis)) < 1.0)
  q_vals_dir_norm[~q_vals_mask] = 0.0
  q_vals_dir_norm = cv.applyColorMap((q_vals_dir_norm * 255).astype(np.uint8), cv.COLORMAP_VIRIDIS)
  for i in range(len(ids)):
    curr_id = ids[i]
    center = np.array([curr_id % _n_light_axix, curr_id // _n_light_axix]) * up_sample
    color = (0, 0, 255) if i == len(ids) - 1 else (255, 255, 255)
    # q_vals_dir_norm = cv.circle(q_vals_dir_norm, tuple(center), up_sample, color, up_sample // 2)
    thickness = int(up_sample * 0.3)
    q_vals_dir_norm = cv.rectangle(q_vals_dir_norm, tuple(center), tuple(center + up_sample), color, thickness)

  # q_vals_dir_norm = np.stack([q_vals_dir_norm] * 3, axis=-1).reshape(-1, 3)
  # q_vals_dir_norm[:, 0] = np.clip((1.0 -q_vals_dir_norm[:, 0]) * 2, 0.0, 1.0)
  # q_vals_dir_norm[:, 2] = np.clip(q_vals_dir_norm[:, 2] * 2, 0.0, 1.0)
  # q_vals_dir_norm[:, 1] = np.minimum(q_vals_dir_norm[:, 0], q_vals_dir_norm[:, 2])
  # q_vals_dir_norm[ids, :] = 0.0
  # q_vals_dir_norm[ids[:-1], 1] = 1.0
  # q_vals_dir_norm[ids[-1], 2] = 1.0
  q_vals_dir_norm = q_vals_dir_norm.reshape(_n_light_axix * up_sample, _n_light_axix * up_sample, 3)
  return q_vals_dir_norm

def q_vals_stack(q_vals, init_image, normal_gt, normal_pred, up_sample=8, margin=2):
  n_q_vals_row = len(q_vals) // 3
  print("normal_err", normal_gt.shape, normal_pred.shape)
  normal_mask = normal_gt[:, :, 2] > 0.0
  normal_err = np.arccos((normal_gt[None, ...] * normal_pred).sum(axis=-1)) * 180 / math.pi
  normal_err[:, ~normal_mask] = 0.0
  normal_gt = display_normal(normal_gt, normal_mask).astype(np.uint16)
  normal_pred = list(map(partial(display_normal, normal_mask=normal_mask), normal_pred))
  mem_zip = io.BytesIO()
  zf = zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED)
  for i, (curr_q_vals, curr_normal_pred, curr_normal_err) in enumerate(zip(q_vals, normal_pred, normal_err)):
    zf.writestr(f"q_vals_{i:06d}.png", cv.imencode(".png", curr_q_vals)[1])
    zf.writestr(f"normal_pred_{i:06d}.png", cv.imencode(".png", curr_normal_pred)[1])
    curr_normal_err = display_normal_err(curr_normal_err, normal_mask)
    zf.writestr(f"normal_err_{i:06d}.png", cv.imencode(".png", curr_normal_err)[1])
  q_vals = np.stack(q_vals[:(n_q_vals_row * 3)], axis=0).astype(np.uint16)
  normal_pred = np.stack(normal_pred[:(n_q_vals_row * 3)], axis=0).astype(np.uint16)[:, :, :, :3]
  res_q_vals = q_vals.shape[2]
  res_normal_pred = normal_pred.shape[2]
  q_vals = q_vals.reshape(n_q_vals_row, 3, res_q_vals, res_q_vals, 3)
  normal_pred = normal_pred.reshape(n_q_vals_row, 3, res_normal_pred, res_normal_pred, 3)
  q_vals = np.pad(q_vals, ((0, 0), (0, 0), (margin, margin), (margin, margin), (0, 0)), constant_values=256)
  normal_pred = np.pad(normal_pred, ((0, 0), (0, 0), (margin, margin), (margin, margin), (0, 0)), constant_values=256)
  # print("q_vals", q_vals.shape)
  res_q_vals = q_vals.shape[2]
  res_normal_pred = normal_pred.shape[2]
  q_vals = q_vals.transpose(0, 2, 1, 3, 4).reshape(n_q_vals_row * res_q_vals, 3 * res_q_vals, 3)
  normal_pred = normal_pred.transpose(0, 2, 1, 3, 4).reshape(n_q_vals_row * res_normal_pred, 3 * res_normal_pred, 3)
  q_vals = np.pad(q_vals, ((0, margin), (margin, margin), (0, 0)), constant_values=256)
  normal_pred = np.pad(normal_pred, ((0, 0), (margin, margin), (0, 0)), constant_values=256)
  init_image = init_image.transpose(1, 0, 2)[::-1, :, :]
  init_image = np.clip(init_image ** (1 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint16)
  init_image = np.pad(init_image, ((margin, margin), (margin, margin), (0, 0)), constant_values=256)
  normal_gt = np.pad(normal_gt, ((margin, margin), (margin, margin), (0, 0)), constant_values=256)
  image = np.concatenate([init_image, normal_gt[:, :, :3]], axis=1)
  image = np.pad(image, ((margin, 0), (margin, margin), (0, 0)), constant_values=256)
  image = np.pad(image, ((0, 0), (0, q_vals.shape[1] - image.shape[1]), (0, 0)), constant_values=256)
  normal_pred = np.pad(normal_pred, ((0, 0), (0, q_vals.shape[1] - normal_pred.shape[1]), (0, 0)), constant_values=256)
  print("q_vals", q_vals.shape, "image", image.shape)
  image = np.concatenate([image, normal_pred, q_vals], axis=0)
  margin_mask = image[..., 0] == 256
  image = image.astype(np.uint8)
  image[margin_mask, 0] = 255
  zf.close()
  return image, mem_zip.getvalue()

def rl_test(qnet, input_files, n_max_images, algo, down_sample=2):
  rng = np.random.default_rng(0)
  env = Env(input_files, rng, algo=algo, down_sample=down_sample)
  results = []
  it_total = 0
  for scene_id in range(len(input_files)):
    state, info = env.reset(scene_id)
    results_iter = []
    results_q_vals = []
    results_normal_pred = []
    for it in range(n_max_images - 1):
      if it == 0:
        init_image = env.images[0]
        normal_gt = env.normal_gt
        # print("init_image", init_image.shape, "init_normal", init_normal.shape)
      with torch.no_grad():
        x_images = torch.tensor(state[0], dtype=torch.float32, device=device)[None, ...]
        x_dirs = env.mesh_action_ids(_n_light_axix//2)[state[1]]
        x_dirs = torch.tensor(x_dirs, dtype=torch.long, device=device)[None, ...]
        q_vals = qnet(x_images, x_dirs)[0, -1, ...].cpu().numpy()
        q_vals_clip = q_vals.mean(axis=(0, 1)).flatten()[env.mesh_action_ids(_n_light_axix)]
        q_vals_clip[env.ids] = -1e9
        action = np.argmax(q_vals_clip)
      # action = env.action_greedy()
      next_state, reward, info = env.step(action)
      image = (np.clip(1.0 + q_vals.mean(axis=(2, 3)) / 8, 0.0, 1.0) * 255).astype(np.uint8)
      cv.imshow("q_vals_img", image)
      image = (np.clip(1.0 + q_vals.mean(axis=(0, 1)) / 4, 0.0, 1.0) * 255).astype(np.uint8)
      cv.imshow("q_vals_dir", image)
      image = q_vals_dir_norm(q_vals, env.mesh_action_ids(_n_light_axix)[env.ids])
      cv.imshow("q_vals_dir_norm", image)
      results_q_vals.append(image)
      results_normal_pred.append(env.normal_pred)
      image = (np.clip(reward / 2, 0.0, 1.0) * 255).astype(np.uint8)
      cv.imshow("reward", image)
      print(f"{time.time():.3f}", \
            "action % 4d" % action, \
            "reward % 10.3f" % reward.mean(), \
            "loss % 10.3f" % 0.0, \
            "q_vals % 8.2f" % q_vals.max(), \
            "it_total % 6d" % it_total, "it % 4d" % (it + 1), "eps % 8.5f" % 0.0, "info % 8.3f" % info)
      it_total += 1
      state = next_state
      env.render()
      if it >= 1:
        results_iter.append(info)
      cv.waitKey(10)
    results.append(results_iter)
    print(f"Result {algo} ({env.input_file }): {results_iter}")
    image, mem_zip = q_vals_stack(np.array(results_q_vals), init_image, normal_gt, np.array(results_normal_pred))
    if len(sys.argv) >= 5:
      cv.imwrite(os.path.join(sys.argv[4], f"{algo}_{scene_id:06d}.jpg"), image)
      with open(os.path.join(sys.argv[4], f"{algo}_{scene_id:06d}.zip"), "wb") as f:
        f.write(mem_zip)
    cv.imshow("q_vals_stack", image)
    cv.waitKey(10)
  print(f"Result {algo} (mean): {np.mean(results, axis=0).tolist()}")

def main():
  n_max_images = int(sys.argv[2])
  input_files = []
  for input_path in sys.argv[1].split("+"):
    for input_file in sorted(glob.glob(os.path.join(input_path, "*.npz"))):
      input_files.append(input_file)
  qnet = QNetEnhance().to(device)
  for algo in _all_algos:
    checkpoint = torch.load(os.path.join(sys.argv[3], f"{algo.lower()}.bin"))
    qnet.load_state_dict(checkpoint["qnet"])
    qnet.eval()
    rl_test(qnet, input_files, n_max_images, algo)

if __name__ == "__main__":
  main()
