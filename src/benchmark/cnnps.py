import os
import gc
import sys
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, concatenate

def get_densenet_2d_channel_last_2dense(rows, cols):
  inputs1 = Input((rows, cols, 1))
  x0 = inputs1
  x1 = Conv2D(16, (3, 3), padding='same', name='conv1')(x0)
  # 1st Denseblock
  x1a = Activation('relu')(x1)
  x2 = Conv2D(16, (3, 3), padding='same', name='conv2')(x1a)
  x2 = Dropout(0.2)(x2)
  xc1 = concatenate([x2, x1], axis=3)
  xc1a = Activation('relu')(xc1)
  x3 = Conv2D(16, (3, 3), padding='same', name='conv3')(xc1a)
  x3 = Dropout(0.2)(x3)
  xc2 = concatenate([x3,x2,x1], axis=3)
  # Transition
  xc2a = Activation('relu')(xc2)
  x4 = Conv2D(48, (1, 1), padding='same', name='conv4')(xc2a)
  x4 = Dropout(0.2)(x4)
  x1 = AveragePooling2D((2, 2), strides=(2, 2))(x4)
  # 2nd Dense block
  x1a = Activation('relu')(x1)
  x2 = Conv2D(16, (3, 3), padding='same', name='conv5')(x1a)
  x2 = Dropout(0.2)(x2)
  xc1 = concatenate([x2, x1], axis=3)
  xc1a = Activation('relu')(xc1)
  x3 = Conv2D(16, (3, 3), padding='same', name='conv6')(xc1a)
  x3 = Dropout(0.2)(x3)
  xc2 = concatenate([x3,x2,x1], axis=3)
  xc2a = Activation('relu')(xc2)
  x4 = Conv2D(80, (1, 1), padding='same', name='conv7')(xc2a)
  x = Flatten()(x4)
  x = Dense(128,activation='relu',name='dense1b')(x)
  x = Dense(3, name='dense2')(x)
  normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1))
  x = normalize(x)
  outputs = x
  model = Model(inputs = inputs1, outputs = outputs)
  return model

_w = 32 # size of observation map
_k = 1 # the number of different rotations for the rotational pseudo-invariance
_model = get_densenet_2d_channel_last_2dense(_w, _w)
_model.compile()
_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_model.load_weights(os.path.join(_data_path, "weight_and_model.hdf5"))

def light_embedding_2d_rot_invariant(I, imax, L, N):
  m = I.shape[0]
  embed_rot = []
  normal_rot = []
  rot = []
  count = 0
  anglemask = np.zeros((I.shape[0], I.shape[1]), dtype=np.float32)
  for k in range(I.shape[0]):
    angle1 = 180 * np.arccos(L[:, 2]) / np.pi
    tgt = np.where(angle1 < 90)
    anglemask[k, tgt] = 1
  for k in range(_k):
    theta = k * 360 / _k
    count = count + 1
    theta = np.pi * theta / 180
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    p = 0.5 * (L[:, 0] + 1) * (_w - 1) #x 0:w-1
    q = 0.5 * (L[:, 1] + 1) * (_w - 1) #y 0:w-1
    x = [p - 0.5 * (_w - 1), q - 0.5 * (_w - 1)]
    x_ = np.dot(rotmat, x)
    p = x_[0, :] + 0.5 * (_w - 1)
    q = x_[1, :] + 0.5 * (_w - 1)
    p = np.int32(p)
    q = np.int32(q)
    light_idx = q * _w + p # 0:w*w-1
    x = [N[:, 0], N[:, 1]]
    x_ = np.dot(rotmat, x)
    pn = x_[0, :]
    qn = x_[1, :]
    normal = [np.transpose(pn), np.transpose(qn), N[:,2]]
    normal = np.transpose(normal)
    temp = I * anglemask / np.transpose(imax)
    embed = np.zeros((m, _w * _w), dtype=np.float32)
    embed[:, light_idx] = temp
    embed = np.reshape(embed, (m, _w, _w))
    embed_rot.append(embed.copy())
    normal_rot.append(normal.copy())
    rot.append(rotmat.copy())
    del embed, temp, normal
  embed_rot = np.array(embed_rot).transpose(1, 0, 2, 3)
  normal_rot = np.array(normal_rot).transpose(1, 0, 2)
  return np.array(embed_rot), np.array(normal_rot), np.array(rot)

def test_network(Sv, Nv, Rv, IDv, Szv):
  height = Szv[0]
  width  = Szv[1]
  rotdiv = Sv.shape[1]
  NestList = []
  for r in range(rotdiv):
    embed_div = Sv[:, r, :, :]
    embed_div = np.reshape(embed_div, (embed_div.shape[0], embed_div.shape[1], embed_div.shape[2], 1))
    # predict
    outputs = _model.predict(embed_div, verbose=0)
    Nest = np.zeros((height * width, 3), dtype=np.float32)
    error = 0
    Err = np.zeros((height * width, 3), dtype=np.float32)
    rot = Rv[r, :, :]
    # N = np.zeros()
    for k in range(len(IDv)):
      # n = outputs[k,:];
      n = np.zeros((2,1),np.float32)
      n[0] = outputs[k,0]
      n[1] = outputs[k,1]
      n = np.dot(np.linalg.inv(rot),n)
      n = [n[0,0],n[1,0],outputs[k,2]]
      n = n/np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
      nt = Nv[k,0,:];
      Nest[IDv[k],:] = n
      for l in range(3):
        Err[IDv[k],l] = 180*np.arccos(min(1,abs(n.dot(np.transpose(nt)))))/np.pi
      error = error + 180*np.arccos(min(1,abs(n.dot(np.transpose(nt)))))/np.pi
    # print("[Angle %d] Ave.Error = %.2f " % (r,(error/len(IDv))))
    NestList.append(Nest.copy())
  NestMean = np.mean(NestList,axis=0)
  Nest = np.zeros((height*width,3), np.float32)
  error = 0
  Err = np.zeros((height*width,3), np.float32)
  for k in range(len(IDv)):
    # n = outputs[k,:];
    n = NestMean[IDv[k],:]
    n = n/np.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
    nt = Nv[k,0,:]
    Nest[IDv[k],:] = n
    for l in range(3):
      Err[IDv[k],l] = 180*np.arccos(min(1,abs(n.dot(np.transpose(nt)))))/np.pi
    error = error + 180*np.arccos(min(1,abs(n.dot(np.transpose(nt)))))/np.pi
  # if rotdiv >= 2:
  #   print("[Mean] Ave.Error = %.2f" % (error/len(IDv)))
  Err = np.reshape(Err,(height,width,3))
  Nest = np.reshape(Nest, (height,width,3))
  return Nest

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
  Iv = images[object_mask, :]
  Nv = normal_gt[object_mask, :]
  object_mask_id = np.flatnonzero(object_mask)
  imax = np.amax(Iv, axis=1) + 1e-3
  if debug:
    print("Iv", Iv.shape, Iv.dtype)
    print("Nv", Nv.shape, Nv.dtype)
    print("object_mask_id", object_mask_id.shape, object_mask_id.dtype)
    print("imax", imax.shape, imax.dtype)
  Sv, Nv, Rv = light_embedding_2d_rot_invariant(Iv, [imax], light_dirs, Nv)
  IDv = object_mask_id
  Szv = object_mask.shape[:2]
  normal_pred = test_network(Sv, Nv, Rv, IDv, Szv)
  K.clear_session()
  gc.collect()
  if debug:
    import cv2 as cv
    cv.imshow("images", images[:, :, 0])
    cv.imshow("Normal Ground Truth", normal_gt * 0.5 + 0.5)
    cv.imshow("Normal Predicted", normal_pred * 0.5 + 0.5)
    cv.waitKey(0)
  return normal_pred
