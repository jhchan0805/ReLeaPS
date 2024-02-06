import math
import numpy as np
from scipy.ndimage import minimum_filter
from ..light_dirs import light_dirs
from .imload import imload_downsample_from_ids

THRESHOLD_QUANTILE = 0.1

def N_trace(images, light_dirs, object_mask, normal_gt=None, dtype=np.float32, debug=False):
  images = images.astype(dtype).copy().mean(axis=-1).transpose(1, 2, 0)
  light_dirs = light_dirs.astype(dtype).copy()
  if normal_gt is not None and debug:
    normal_gt = normal_gt.astype(dtype).copy()
  else:
    normal_gt = np.zeros([images.shape[0], images.shape[1], 3], dtype=dtype)
    normal_gt[:, :, 2] = 1.0
  if debug:
    print("images", images.shape, images.dtype)
    print("light_dirs", light_dirs.shape, light_dirs.dtype)
    print("normal_gt", normal_gt.shape, normal_gt.dtype)
    print("object_mask", object_mask.shape, object_mask.dtype)
  intensity_low = np.quantile(images[object_mask, :], dtype(THRESHOLD_QUANTILE)).astype(dtype)
  intensity_high = np.quantile(images[object_mask, :], dtype(1.0)).astype(dtype)
  if debug:
    print("intensity_low", intensity_low, "intensity_high", intensity_high)
  light_num = light_dirs.shape[0]
  light_dirs = light_dirs[np.newaxis, :, :]
  light_dirs /= np.linalg.norm(light_dirs, ord=2, axis=2, keepdims=True)
  I = images[object_mask, :, np.newaxis]
  weight = (I > intensity_low).astype(dtype)
  L = np.broadcast_to(light_dirs, [object_mask.sum(), light_num, 3]) * weight
  I = I * weight
  LTI = np.matmul(L.transpose(0, 2, 1), I)
  LTL = np.matmul(L.transpose(0, 2, 1), L) + 1e-6 * np.eye(3, dtype=dtype)[None, ...]
  inv_LTL = np.linalg.inv(LTL)
  inv_LTL_LTI = np.matmul(inv_LTL, LTI)
  N = np.zeros([images.shape[0], images.shape[1], 3], dtype=dtype)
  N[object_mask, :] = inv_LTL_LTI[..., 0]
  N /= (np.linalg.norm(N, ord=2, axis=2, keepdims=True) + 1e-6)
  N[~object_mask, :] = 0.0
  N_tr = np.zeros([images.shape[0], images.shape[1]], dtype=dtype)
  trace_inv_LTL = np.trace(inv_LTL, axis1=1, axis2=2)
  N_tr[object_mask] = trace_inv_LTL
  normal_gt /= (np.linalg.norm(normal_gt, ord=2, axis=2, keepdims=True) + 1e-6)
  normal_gt[~object_mask, :] = 0.0
  if debug:
    print("L.T", L.transpose(0, 2, 1).shape)
    print("L.T I", LTI.shape, LTI.min(), LTI.max(), LTI.dtype)
    print("L.T L", LTL.shape, LTL.min(), LTL.max(), LTL.dtype)
    print("inv(L.T L)", inv_LTL.shape, inv_LTL.min(), inv_LTL.max())
    print("inv(L.T L)LTI", inv_LTL_LTI.shape, inv_LTL_LTI.min(), inv_LTL_LTI.max())
    print("N_tr", N_tr.shape, N_tr.min(), N_tr.max())
    import cv2 as cv
    cv.imshow("Normal Ground Truth", normal_gt * 0.5 + 0.5)
    cv.imshow("Normal Reconstruct", N * 0.5 + 0.5)
    I_reconstruct = (light_dirs[np.newaxis, ...] * N[:, :, np.newaxis, :]).sum(-1)
    I_reconstruct = np.clip(I_reconstruct, 0.0, 1.0)
    I_GT = (light_dirs[np.newaxis, ...] * normal_gt[:, :, np.newaxis, :]).sum(-1)
    I_GT = np.clip(I_GT, 0.0, 1.0)
    cv.imshow("Image Input", images[..., -1])
    cv.imshow("Image Ground Truth", I_GT[..., -1])
    cv.imshow("Image Reconstruct", I_reconstruct[..., -1])
    cv.imshow("Image Ratio", images[..., -1] / (I_GT[..., -1] + 1e-6))
    cv.waitKey(0)
    print("N", N.shape, N.dtype, N.min(), N.max())
  return N, N_tr

def C_vis(pixels, threshold, selected_light_dirs, all_light_dirs, dtype=np.float32, debug=False):
  if debug:
    import cv2 as cv
    print("pixels", pixels.shape, pixels.dtype)
    print("threshold", threshold)
    print("selected_light_dirs", selected_light_dirs.shape, selected_light_dirs.dtype)
    print("all_light_dirs", all_light_dirs.shape, all_light_dirs.dtype)
  w2 = 1.0 / pixels.shape[0]
  C_v = np.exp(-np.power(all_light_dirs[:, :] - np.array([[0, 0, 1]], dtype=dtype), 2).sum(axis=-1) / (2 * w2))
  for curr_pixel, curr_light_dir in zip(pixels, selected_light_dirs):
    V_p = -1.0 if curr_pixel.mean() <= threshold else 1.0
    C_v += V_p * np.exp(-np.power(all_light_dirs[:, :] - curr_light_dir[None, :], 2).sum(axis=-1) / (2 * w2))
  C_v = C_v.clip(-1.0, 1.0) / (2 * math.pi * w2)
  if debug:
    print("C_v", C_v.shape, C_v.dtype)
    resolution=256
    image = np.full([resolution, resolution, 3], 255, dtype=np.uint8)
    nodeRadius = int(resolution * 0.03)
    for curr_C_v, curr_light_dir in zip(C_v, all_light_dirs):
      color = (0, max(0, int(curr_C_v * 255)), max(0, int(-curr_C_v * 255)))
      nodeCenter = tuple((resolution * (0.5 + 0.45 * curr_light_dir[[1, 0]])).astype(int))
      cv.circle(image, nodeCenter, nodeRadius, color, -1)
    nodeRadius = int(resolution * 0.015)
    for curr_pixel, curr_light_dir in zip(pixels, selected_light_dirs):
      color = (255, 0, 255) if curr_pixel.mean() <= threshold else (255, 255, 0)
      nodeCenter = tuple((resolution * (0.5 + 0.45 * curr_light_dir[[1, 0]])).astype(int))
      cv.circle(image, nodeCenter, nodeRadius, color, -1)
    cv.imshow("C_vis", image)
    cv.waitKey(0)
  return C_v

def C_lin(pixels, threshold, selected_light_dirs, all_light_dirs, dtype=np.float32, debug=False):
  if debug:
    import cv2 as cv
    print("pixels", pixels.shape, pixels.dtype)
    print("threshold", threshold)
    print("selected_light_dirs", selected_light_dirs.shape, selected_light_dirs.dtype)
    print("all_light_dirs", all_light_dirs.shape, all_light_dirs.dtype)
  weight = pixels.mean(axis=-1) > threshold
  L = selected_light_dirs * weight[:, None]
  LTL = np.matmul(L.T, L) + 1e-6 * np.eye(3, dtype=dtype)
  v, w = np.linalg.eig(LTL)
  S_min = w[:, v.argmin()]
  C_l = (all_light_dirs * S_min[None, :]).sum(axis=-1)
  if debug:
    print("L.T L", LTL.shape)
    print("v", v, "w", w, "S_min", S_min)
    print("C_l", C_l.shape, C_l.dtype)
    resolution=256
    image = np.full([resolution, resolution, 3], 255, dtype=np.uint8)
    nodeRadius = int(resolution * 0.03)
    for curr_C_l, curr_light_dir in zip(C_l, all_light_dirs):
      color = (0, max(0, int(curr_C_l * 255)), max(0, int(-curr_C_l * 255)))
      nodeCenter = tuple((resolution * (0.5 + 0.45 * curr_light_dir[[1, 0]])).astype(int))
      cv.circle(image, nodeCenter, nodeRadius, color, -1)
    cv.imshow("C_lin", image)
    cv.waitKey(0)
  return np.abs(C_l)

def okabe_light_ids(data_loader, n_lights, rng, debug=False):
  all_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])
  ids = [np.argmax(all_light_dirs[:, 2])]
  for i in range(n_lights - 1):
    normal_gt, selected_images, selected_light_dirs = imload_downsample_from_ids(data_loader, ids)
    object_mask = minimum_filter(normal_gt[..., -1], 3) > 0.0
    N, N_tr = N_trace(selected_images, selected_light_dirs, object_mask, normal_gt, debug=debug)
    N_tr = N_tr.clip(max=100) + 1e-2 * rng.random(N_tr.shape, dtype=np.float32)
    N_tr_max = np.argmax(N_tr)
    resolution = N_tr.shape[0]
    if debug:
      import cv2 as cv
      image = np.stack([N_tr * 0.01] * 3, axis=-1)
      nodeRadius = int(resolution * 0.03)
      nodeCenter = np.unravel_index(N_tr_max, N_tr.shape)[::-1]
      cv.circle(image, nodeCenter, nodeRadius, (0, 0, 1), -1)
      # image.reshape(-1, 3)[N_tr_max, :] = [0.0, 0.0, 1.0]
      print("N_tr_max", N_tr_max, "N_tr", image.shape, image.dtype)
      cv.imshow("Normal Trace", image)
      cv.waitKey(0)
    pixels = selected_images[:, N_tr_max // resolution, N_tr_max % resolution, :]
    threshold = np.quantile(selected_images[:, object_mask].mean(axis=-1),
      np.float32(THRESHOLD_QUANTILE)).astype(np.float32)
    C_v = C_vis(pixels, threshold, selected_light_dirs, all_light_dirs, debug=debug)
    C_l = C_lin(pixels, threshold, selected_light_dirs, all_light_dirs, debug=debug)
    # ids.append(np.random.randint(len(all_light_dirs)))
    ids.append((C_v * C_l).argmax())
  return ids

if __name__ == "__main__":
  for i in range(500):
    data_loader = np.load(f"data/blobs_test/{i:06d}.npz", allow_pickle=True)
    ids = okabe_light_ids(data_loader, 10, np.random.default_rng(i), debug=True)
    print(ids)
