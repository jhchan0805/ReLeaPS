import numpy as np

def light_dirs_grid(n_light_axix=11, dtype=np.float32):
  light_x_array = np.linspace(-1.0, 1.0, num=n_light_axix, endpoint=True)
  light_x_array = np.sign(light_x_array) * np.power(np.abs(light_x_array), 2) * 2.0
  light_x_array = light_x_array[:, np.newaxis]
  light_y_array = np.linspace(-1.0, 1.0, num=n_light_axix, endpoint=True)
  light_y_array = np.sign(light_y_array) * np.power(np.abs(light_y_array), 2) * 2.0
  light_y_array = light_y_array[np.newaxis, :]
  light_dirs = np.stack(np.broadcast_arrays(light_x_array, light_y_array, 1.0), axis=-1)
  light_dirs = light_dirs.reshape(-1, 3)
  light_dirs /= np.linalg.norm(light_dirs, ord=2, axis=1, keepdims=True)
  return light_dirs.astype(dtype)

def light_dirs_diligent(dtype=np.float32):
  light_dirs = np.loadtxt("data/light_dirs/diligent_ball.txt", dtype=dtype)
  light_dirs /= np.linalg.norm(light_dirs, ord=2, axis=1, keepdims=True)
  return light_dirs

def light_dirs_diligent100(dtype=np.float32):
  light_dirs = np.loadtxt("data/light_dirs/diligent100_theory.csv", dtype=dtype, delimiter=",")
  light_dirs /= np.linalg.norm(light_dirs, ord=2, axis=1, keepdims=True)
  return light_dirs

def light_dirs_new(dtype=np.float32):
  light_dirs = np.loadtxt("data/light_dirs/new.csv", dtype=dtype, delimiter=",")
  light_dirs /= np.linalg.norm(light_dirs, ord=2, axis=1, keepdims=True)
  return light_dirs

_light_dirs_grid = light_dirs_grid()
_light_dirs_diligent = light_dirs_diligent()
_light_dirs_diligent100 = light_dirs_diligent100()
_light_dirs_new = light_dirs_new()
_light_dirs = {
  "grid": _light_dirs_grid,
  "diligent": _light_dirs_diligent,
  "diligent100": _light_dirs_diligent100,
  "new": _light_dirs_new,
}

def light_dirs(mode):
  if isinstance(mode, np.ndarray):
    return mode / np.linalg.norm(mode, ord=2, axis=1, keepdims=True)
  return _light_dirs[mode].copy()
