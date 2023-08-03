import io
import os
import sys
import glob
import zipfile
import numpy as np
import cv2 as cv
from scipy.io import loadmat

def get_crop_slice(mask):
  up = 0
  while np.all(mask[up, :] == 1):
    up += 1
  down = mask.shape[0] - 1
  while np.all(mask[down, :] == 1):
    down -= 1
  left = 0
  while np.all(mask[:, left] == 1):
    left += 1
  right = mask.shape[1] - 1
  while np.all(mask[:, right] == 1):
    right -= 1
  while right - left < down - up:
    if left > 0:
      left -= 1
    if right - left < down - up and right < mask.shape[1]:
      right += 1
  while down - up < right - left:
    if up > 0:
      up -= 1
    if down - up < right - left and down < mask.shape[0]:
      down += 1
  print("up", up, "down", down, "height", down - up + 1, "left", left, "right", right, "width", right - left + 1)
  return slice(up, down + 1), slice(left, right + 1)


def process(input_dir, file_output):
  print(f"input_dir {input_dir}")
  light_intensities = input_dir.joinpath("light_intensities.txt").read_bytes().decode('utf-8').strip().split("\n")
  light_intensities = np.loadtxt(light_intensities)
  light_dirs = input_dir.joinpath("light_directions.txt").read_bytes().decode('utf-8').strip().split("\n")
  light_dirs = np.loadtxt(light_dirs)
  images_orig = []
  mask = np.frombuffer(input_dir.joinpath("mask.png").read_bytes(), dtype=np.uint8)
  mask = cv.imdecode(mask, cv.IMREAD_UNCHANGED) == 0
  print("mask", mask.shape, mask.dtype)
  if len(mask.shape) == 3:
    mask = mask[:, :, 0]
  # mask = normal[:, :, 2] <= 0.0
  slice_x, slice_y = get_crop_slice(mask)
  if input_dir.joinpath("normal.txt").exists():
    normal = input_dir.joinpath("normal.txt").read_bytes().decode('utf-8').strip().split("\n")
    normal = np.loadtxt(normal).reshape(mask.shape[0], mask.shape[1], 3)
    if "harvestPNG" in input_dir.name:
      normal = normal[::-1, :, :]
    cv.imshow("normal_orig", normal * 0.5 + 0.5)
  elif input_dir.joinpath("Normal_gt.mat").exists():
    with input_dir.joinpath("Normal_gt.mat").open("rb") as f:
      normal = loadmat(f)["Normal_gt"]
  # elif "TURBINE" in input_dir.name:
  #   normal = loadmat("data/TURBINE.mat")["Normal_gt"]
  elif "DiLiGenT10^2" in str(input_dir):
    normal_path = os.path.join("data", "pmsData", input_dir.name, "Normal_gt.mat")
    normal = loadmat(normal_path)["Normal_gt"]
  else:
    normal = np.zeros([mask.shape[0], mask.shape[1], 3])
    normal[:, :, 2] = 1.0
  normal = normal[:, :, [1, 0, 2]]
  normal[:, :, 0] *= -1
  normal = normal[slice_x, slice_y, :]
  mask = mask[slice_x, slice_y]
  filenames = input_dir.joinpath("filenames.txt").read_bytes().decode('utf-8').strip().split("\n")
  # print(filenames)
  for filename in filenames:
    image_file = np.frombuffer(input_dir.joinpath(filename.strip()).read_bytes(), dtype=np.uint8)
    image = cv.imdecode(image_file, cv.IMREAD_UNCHANGED)
    images_orig.append(image)
  images_orig = np.array(images_orig, dtype=np.float32)
  images_orig /= images_orig.max()
  print("images_orig", images_orig.shape, images_orig.dtype)
  cv.imshow("images_orig", images_orig[0])
  cv.waitKey(10)
  images = images_orig[:, slice_x, slice_y, :] / light_intensities[:, None, None, :]
  images = np.power(np.clip(images / np.quantile(images, 0.99), 0.0, 1.0), 1 / 2.2)
  images[:, mask, :] = 0
  normal[mask, :] = 0.0
  images = np.array([cv.resize(image, (256, 256), cv.INTER_AREA) for image in images], dtype=np.float32)
  images = np.array(np.clip(images * 255, 0, 255), dtype=np.uint8)
  normal = cv.resize(normal, (256, 256), cv.INTER_AREA)
  images = images.transpose(0, 2, 1, 3)[:, :, ::-1, :]
  normal = normal.transpose(1, 0, 2)[:, ::-1, :]
  normal = normal[:, :, [1, 0, 2]]
  normal[:, :, 1] *= -1
  for i in range(1, 2):
    print("images", images.shape, images.dtype)
    cv.imshow("mask", mask * 1.0)
    cv.imshow("images", images[i])
    cv.imshow("normal", normal * 0.5 + 0.5)
    cv.waitKey(10)
  if "bearPNG" in input_dir.name:
    images = images[20:]
    light_dirs = light_dirs[20:]
  meta_info = {
    "filename": input_dir.name,
    "light_dirs_mode": light_dirs,
    "crop": (slice_x, slice_y),
  }
  images = {"image_%06d" % i: cv.imencode(".webp", images[i])[1] for i, image in enumerate(images)}
  print("normal", normal.shape, normal.dtype)
  np.savez_compressed(file_output, normal=normal, meta_info=meta_info, **images)
  cv.waitKey(10)

def main():
  input_file = zipfile.ZipFile(sys.argv[1])
  zip_path = zipfile.Path(input_file).joinpath(sys.argv[2])
  input_dirs = sorted(map(lambda x: x.name, zip_path.iterdir()))
  # if "DiLiGenT10^2" in sys.argv[1]:
  #   input_dirs = filter(lambda input_dir: "TURBINE" in input_dir, input_dirs)
  if "Junhong" in sys.argv[1]:
    input_dirs = filter(lambda input_dir: "_" not in input_dir, input_dirs)
  for i, input_dir in enumerate(input_dirs):
    print("Processing", input_dir)
    process(zip_path.joinpath(input_dir), os.path.join(sys.argv[3], "%06d.npz" % i))

if __name__ == "__main__":
  main()
