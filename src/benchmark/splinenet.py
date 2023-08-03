import os
import sys
import importlib
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
import torch
import numpy as np
import torch.nn.functional as F

_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_old_path = sys.path.copy()
sys.path.append(os.path.join(os.path.dirname(__file__), "SPLINE-Net"))
Solver = importlib.import_module(".SPLINE-Net.solver", __package__).Solver
sys.path = _old_path

# Simplified Config for https://github.com/q-zh/SPLINE-Net
class Config(object):
  def __init__(self):
    # Model configuration.
    self.image_size = 32 # image resolution
    self.g_conv_dim = 64 # number of conv filters in the first layer of G
    self.d_conv_dim = 64 # number of conv filters in the first layer of D
    self.g_repeat_num = 6 # number of residual blocks in G
    self.d_repeat_num = 5 # number of strided conv layers in D
    self.lambda_L1 = 180/np.pi # weight for L1 loss
    self.lambda_Light = 180/np.pi # weight for Light loss
    self.lambda_Iso = 0.02*180/np.pi # weight for Iso loss
    self.lambda_Sparse = 2e-5*180/np.pi # weight for Sparse loss
    self.lambda_Conti = 1e-3*180/np.pi # weight for Continuity loss
    # Training configuration.
    self.dataset = "Pixels" # name of the dataset
    self.batch_size = 128 # mini-batch size
    self.num_iters = 200000 # number of total iterations for training D
    self.num_iters_decay = 100000 # number of iterations for decaying lr
    self.g_lr = 0.0001 # learning rate for G
    self.d_lr = 0.0001 # learning rate for D
    self.n_critic = 5 # number of D updates per each G update
    self.beta1 = 0.5 # beta1 for Adam optimizer
    self.beta2 = 0.999 # beta2 for Adam optimizer
    self.resume_iters = None # resume training from this step
    # Test configuration.
    self.test_iters = 200000 # test model from this step
    # Miscellaneous.
    self.num_workers = 1
    self.mode = "test"
    self.use_tensorboard = False
    # Directories.
    self.train_image_dir = "data/train"
    self.test_image_dir = "data/test"
    self.shape = None
    self.log_dir = "photometric/logs"
    self.model_save_dir = os.path.join(_data_path, "SPLINE-NET")
    self.result_dir = "photometric/results"
    self.result_ind = None
    # Step size.
    self.log_step = 10
    self.model_save_step = 10000
    self.lr_update_step = 50000

_config = Config()
_solver = Solver(None, _config)
_solver.restore_model(_solver.test_iters)

def benchmark(images, light_dirs, object_mask, normal_gt=None, dtype=np.float32, debug=False):
  if debug:
    print("images", images.shape, images.dtype)
    print("light_dirs", light_dirs.shape, light_dirs.dtype)
    print("normal_gt", normal_gt.shape, normal_gt.dtype)
    print("object_mask", object_mask.shape, object_mask.dtype)
  # images = images[:, ::4, ::4]
  # normal_gt = normal_gt[::4, ::4]
  # object_mask = object_mask[::4, ::4]
  images = images.astype(dtype).copy()[:, :, :, ::-1]
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
  images = images.transpose(0, 3, 1, 2)
  images = images.transpose(0, 1, 3, 2)[:, :, ::-1, :]
  object_mask = object_mask.transpose(1, 0)[::-1, :]
  images[:, :, ~object_mask] = 0.0
  images = torch.tensor(images.copy(), dtype=torch.float32, device=device)
  if debug:
    print("images", images.shape, images.dtype)
    import cv2 as cv
    cv.imshow("image", images.detach()[0, :3].permute(1, 2, 0).cpu().numpy())
    print("light_dirs", light_dirs.shape, light_dirs.dtype)
    print(light_dirs[0, :3])
  images = images.mean(axis=1)
  dataset = images.reshape(images.shape[0], images.shape[1] * images.shape[2]).transpose(1, 0)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)
  normal_pred = []
  with torch.no_grad():
    for x in data_loader:
      obs_map = torch.zeros(x.shape[0], 1, 32, 32, dtype=torch.float32, device=device)
      for i_image in range(x.shape[1]):
        (px, py) = np.round((light_dirs[i_image, :2] * 0.5 + 0.5) * 31)
        obs_map[:, 0, int(px), int(py)] = x[:, i_image]
      x = _solver.G(obs_map)
      x = torch.cat([x, obs_map],dim=1)
      x = _solver.D(x)[:, (1, 0, 2)].cpu().numpy()
      normal_pred.append(x)
  normal_pred = np.concatenate(normal_pred, axis=0).reshape(images.shape[1], images.shape[2], 3)
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
