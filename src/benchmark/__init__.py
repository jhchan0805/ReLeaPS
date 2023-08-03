import os
import sys
import glob
import zipfile
import importlib
import itertools
from functools import partial
import cv2 as cv
import numpy as np
from scipy.ndimage import minimum_filter
from ..light_dirs import light_dirs
from .imload import imload_downsample_from_ids

# _all_algos = ["LS"]
# _algo_processes = [32]
# _all_algos = ["CNNPS"]
# _algo_processes = [12]
# _all_algos = ["PSFCN"]
# _algo_processes = [1]
# _all_algos = ["SPLINENET"]
# _algo_processes = [1]
_all_algos = ["LS", "CNNPS", "PSFCN", "SPLINENET"]
_algo_processes = [32, 12, 12, 1]

def display_normal(normal, normal_mask):
  normal_mask = minimum_filter(normal_mask, size=3, mode="constant") > 0.0
  normal_mask = normal_mask.transpose(1, 0)[::-1, :]
  normal_alpha = (normal_mask * 255).astype(np.uint8)
  normal = normal.transpose(1, 0, 2)[::-1, :, [2, 1, 0]]
  normal = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
  normal = np.concatenate([normal, normal_alpha[:, :, np.newaxis]], axis=-1)
  return normal

def display_normal_err(normal_err, normal_mask):
  normal_mask = minimum_filter(normal_mask, size=3, mode="constant") > 0.0
  normal_mask = normal_mask.transpose(1, 0)[::-1, :]
  normal_alpha = (normal_mask * 255).astype(np.uint8)
  normal_err = normal_err.transpose(1, 0)[::-1, :]
  normal_err = np.clip(normal_err, 0.0, 90) / 90
  normal_err = cv.applyColorMap((normal_err * 255).astype(np.uint8), cv.COLORMAP_TURBO)
  normal_err = np.concatenate([normal_err, normal_alpha[:, :, np.newaxis]], axis=-1)
  return normal_err

def benchmark(images, light_dirs, object_mask, normal_gt, algo, dtype=np.float32, debug=False):
  images = images.astype(dtype).copy()
  light_dirs = light_dirs.astype(dtype).copy()
  normal_gt = normal_gt.astype(dtype).copy()
  if debug:
    print("images", images.shape, images.dtype)
    print("light_dirs", light_dirs.shape, light_dirs.dtype)
    print("normal_gt", normal_gt.shape, normal_gt.dtype)
  normal_gt /= (np.linalg.norm(normal_gt, ord=2, axis=2, keepdims=True) + 1e-6)
  normal_gt[~object_mask, :] = 0.0
  benchmark_algo = importlib.import_module(f".{algo.lower()}", __package__).benchmark
  normal_pred = benchmark_algo(images, light_dirs, object_mask, normal_gt, dtype)
  normal_pred /= (np.linalg.norm(normal_pred, ord=2, axis=2, keepdims=True) + 1e-6)
  normal_pred[~object_mask, :] = 0.0
  error = 1.0 - (normal_pred * normal_gt).sum(axis=-1)
  error[~object_mask] = 0.0
  if debug:
    print("error", error.shape, error.dtype, error.min(), error.max())
  error_degree = np.arccos(1.0 - error) / np.pi * 180
  return error_degree, normal_pred

def benchmark_from_ids(data_loader, ids, algo, down_sample=2):
  ids = list(set(ids))
  normal_gt, selected_images, selected_light_dirs = imload_downsample_from_ids(data_loader, ids, down_sample)
  object_mask = minimum_filter(normal_gt[..., -1], size=3, mode="constant") > 0.0
  error, N = benchmark(selected_images, selected_light_dirs, object_mask, normal_gt, algo)
  error_mean = error[object_mask].mean()
  return error, error_mean, N

def benchmark_random_file(test_id, input_file, n_lights, algo, down_sample=2):
  print("benchmark_random_file", test_id, input_file, algo)
  rng = np.random.default_rng(test_id)
  data_loader = np.load(input_file, allow_pickle=True)
  all_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])
  num_light_dirs = len(all_light_dirs)
  ids = rng.choice(np.arange(num_light_dirs), size=n_lights, replace=False)
  ids[0] = np.argmax(all_light_dirs[:, 2])
  results = []
  for j in range(2, n_lights):
    _, result, _ = benchmark_from_ids(data_loader, ids[:(j+1)], algo, down_sample=down_sample)
    results.append(result)
  return results

def benchmark_random_pool(pool, algo, input_path, n_lights, n_samples, rng):
  input_files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
  input_files = [input_files[i % len(input_files)] for i in range(n_samples)]
  benchmark_random_file_partial = partial(benchmark_random_file, n_lights=n_lights, algo=algo)
  results = pool.starmap(benchmark_random_file_partial, enumerate(input_files))
  for input_file, result in zip(input_files, results):
    print(f"Result {algo} ({input_file}): {result}")
  print(f"Result {algo} (mean): {np.mean(results, axis=0).tolist()}")

def benchmark_all_file(test_id, input_file, algo, down_sample=2):
  print("benchmark_all_file", test_id, input_file, algo)
  data_loader = np.load(input_file, allow_pickle=True)
  num_light_dirs = len(light_dirs(data_loader["meta_info"].item()["light_dirs_mode"]))
  _, result, _ = benchmark_from_ids(data_loader, range(num_light_dirs), algo, down_sample=down_sample)
  return result

def benchmark_all_pool(pool, algo, input_path):
  input_files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
  benchmark_all_file_partial = partial(benchmark_all_file, algo=algo)
  results = pool.starmap(benchmark_all_file_partial, enumerate(input_files))
  for input_file, result in zip(input_files, results):
    print("Result " + algo + " (" + input_file + "):", result)
  print("Result " + algo + " (mean):", np.mean(results).item())

def action_greedy(data_loader, ids, start=0, step=1, down_sample=4):
  all_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])
  num_light_dirs = len(all_light_dirs)
  ids = ids + [0]
  id_best = 0
  error_best = sys.float_info.max
  for k in range(start, num_light_dirs, step):
    if k in ids[:-1]:
      continue
    ids[-1] = k
    _, result, _ = benchmark_from_ids(data_loader, ids, algo="LS", down_sample=down_sample)
    if result < error_best:
      error_best = result
      id_best = k
  return id_best

