import os
import sys
import importlib
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0")
normalize = importlib.import_module(".PS-FCN.datasets.pms_transforms", __package__).normalize
PS_FCN = importlib.import_module(".PS-FCN.models.PS_FCN_run", __package__).PS_FCN
net = PS_FCN(c_in=6).to(device)
checkpoint_path = os.path.join("PS-FCN", "data", "models", "PS-FCN_B_S_32_normalize.pth.tar")
checkpoint = torch.load(os.path.join(os.path.dirname(__file__), checkpoint_path))
net.load_state_dict(checkpoint['state_dict'])

def benchmark(images, light_dirs, object_mask, normal_gt=None, dtype=np.float32, debug=False):
  images = images.astype(dtype).copy()[:, :, :, ::-1]
  n_images = images.shape[0]
  images = np.concatenate(normalize(images), axis=2)
  images *= np.sqrt(n_images / 32)
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
  images = images.transpose(2, 0, 1)[np.newaxis, ...]
  # Rotate images to DiLiGenT format
  images = images.transpose(0, 1, 3, 2)[:, :, ::-1, :]
  object_mask = object_mask.transpose(1, 0)[::-1, :]
  images[:, :, ~object_mask] = 0.0
  images = torch.tensor(images.copy(), dtype=torch.float32, device=device)
  light_dirs = torch.tensor(light_dirs.copy(), dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
  light_dirs = light_dirs.repeat(1, 1, images.shape[2], images.shape[3])
  if debug:
    print("images", images.shape, images.dtype)
    import cv2 as cv
    cv.imshow("image", images.detach()[0, :3].permute(1, 2, 0).cpu().numpy())
    print("light_dirs", light_dirs.shape, light_dirs.dtype)
    print(light_dirs[0, :3, 0, 0])
  with torch.no_grad():
    normal_pred = net([images, light_dirs])[0, ...].permute(1, 2, 0).cpu().numpy()
  del images
  del light_dirs
  torch.cuda.empty_cache()
  normal_pred[~object_mask, :] = 0.0
  normal_pred = normal_pred.transpose(1, 0, 2)[:, ::-1, :]
  if debug:
    print("normal_pred", normal_pred.shape, normal_pred.dtype)
    import cv2 as cv
    cv.imshow("Normal Ground Truth", normal_gt * 0.5 + 0.5)
    cv.imshow("Normal Predicted", normal_pred * 0.5 + 0.5)
    cv.waitKey(0)
  return normal_pred
