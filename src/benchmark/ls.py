import numpy as np

def benchmark(images, light_dirs, object_mask, normal_gt=None, dtype=np.float32, debug=False):
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
  intensity_low = np.quantile(images[object_mask, :], dtype(0.0)).astype(dtype)
  intensity_high = np.quantile(images[object_mask, :], dtype(1.0)).astype(dtype)
  if debug:
    print("intensity_low", intensity_low, "intensity_high", intensity_high)
  images = np.append(images, np.ones([images.shape[0], images.shape[1], 1], dtype=dtype), axis=2)
  light_num = light_dirs.shape[0] + 1
  light_dirs = np.append(light_dirs, [np.array([0.0, 0.0, 1.0], dtype=dtype)], axis=0)
  light_dirs = light_dirs[np.newaxis, :, :]
  light_dirs /= np.linalg.norm(light_dirs, ord=2, axis=2, keepdims=True)
  I = images[object_mask, :, np.newaxis]
  weight = np.maximum(0.0, np.minimum(I - intensity_low, intensity_high - I))
  weight[:, -1, :] = 1e-3
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
  normal_gt /= (np.linalg.norm(normal_gt, ord=2, axis=2, keepdims=True) + 1e-6)
  normal_gt[~object_mask, :] = 0.0
  if debug:
    print("L.T", L.transpose(0, 2, 1).shape)
    print("L.T I", LTI.shape, LTI.min(), LTI.max(), LTI.dtype)
    print("L.T L", LTL.shape, LTL.min(), LTL.max(), LTL.dtype)
    print("inv(L.T L)", inv_LTL.shape, inv_LTL.min(), inv_LTL.max())
    print("inv(L.T L)LTI", inv_LTL_LTI.shape, inv_LTL_LTI.min(), inv_LTL_LTI.max())
    import cv2 as cv
    cv.imshow("Normal Ground Truth", normal_gt * 0.5 + 0.5)
    cv.imshow("Normal Reconstruct", N * 0.5 + 0.5)
    I_reconstruct = (light_dirs[np.newaxis, ...] * N[:, :, np.newaxis, :]).sum(-1)
    I_reconstruct = np.clip(I_reconstruct, 0.0, 1.0)
    I_GT = (light_dirs[np.newaxis, ...] * normal_gt[:, :, np.newaxis, :]).sum(-1)
    I_GT = np.clip(I_GT, 0.0, 1.0)
    for i in range(light_num - 1):
      cv.imshow("Image Input", images[..., i])
      cv.imshow("Image Ground Truth", I_GT[..., i])
      cv.imshow("Image Reconstruct", I_reconstruct[..., i])
      cv.imshow("Image Ratio", images[..., i] / (I_GT[..., i] + 1e-6))
      cv.waitKey(0)
    print("N", N.shape, N.dtype, N.min(), N.max())
  return N

