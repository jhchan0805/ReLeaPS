import os
import time
import threading
import cv2 as cv
import numpy as np
from ..light_dirs import light_dirs

_image_cache = None

def _image_cache_clean():
  global _image_cache
  while True:
    time.sleep(10)
    # old_len = len(_image_cache)
    _image_cache = dict(filter(lambda val: val[1][1], _image_cache.items()))
    _image_cache = dict(map(lambda val: [val[0], [val[1][0], False]], _image_cache.items()))
    # print("_image_cache_clean: before", old_len, "after", len(_image_cache))

def imload_downsample_from_ids(data_loader, ids, down_sample=2):
  global _image_cache
  if _image_cache is None:
    _image_cache = {}
    _image_cache_clean_thread = threading.Thread(target=_image_cache_clean, daemon=True)
    _image_cache_clean_thread.start()
  normal_key = (data_loader.zip.filename, down_sample)
  if normal_key in _image_cache:
    _image_cache[normal_key][1] = True
    normal = _image_cache[normal_key][0]
  else:
    normal = data_loader["normal"]
    res_x = normal.shape[0] // down_sample
    res_y = normal.shape[1] // down_sample
    normal = normal.reshape(res_x, down_sample, res_y, down_sample, 3)
    normal = np.mean(normal, axis=(1, 3))
    _image_cache[normal_key] = [normal, True]
  selected_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])[ids, :]
  selected_images = []
  for curr_id, curr_light_dir in zip(ids, selected_light_dirs):
    curr_image_key = (data_loader.zip.filename, curr_id, down_sample)
    if curr_image_key in _image_cache:
      _image_cache[curr_image_key][1] = True
      curr_image = _image_cache[curr_image_key][0]
    else:
      if os.environ.get("RLPS_ABLATION") == "1":
        curr_image = (normal * curr_light_dir[np.newaxis, np.newaxis, :]).sum(axis=-1, keepdims=True)
        curr_image = curr_image.clip(0.0, 1.0).repeat(3, -1) * 0.9
        curr_image += np.random.rand(*curr_image.shape) * 0.1
        # print("curr_image", curr_image.shape, curr_image.min(), curr_image.max())
      else:
        curr_image = cv.imdecode(data_loader["image_%06d" % curr_id], cv.IMREAD_UNCHANGED).astype(np.float32)
        curr_image = np.power(curr_image / np.float32(255.0), np.float32(2.2))
        res_x = curr_image.shape[0] // down_sample
        res_y = curr_image.shape[1] // down_sample
        curr_image = curr_image.reshape(res_x, down_sample, res_y, down_sample, 3)
        curr_image = np.mean(curr_image, axis=(1, 3))
      _image_cache[curr_image_key] = [curr_image, True]
    # print("curr_image", curr_image.shape, curr_image.dtype)
    selected_images.append(curr_image)
  selected_images = np.stack(selected_images, axis=0)
  return normal, selected_images, selected_light_dirs
