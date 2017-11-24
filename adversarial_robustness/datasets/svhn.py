from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import scipy.io
import random
from adversarial_robustness.dataset import *

class SVHN(Dataset):
  def __init__(self, include_train=True, data_dir=default_data_dir, **kwargs):
    self.X, self.y, self.Xv, self.yv, self.Xt, self.yt = load_svhn(
        include_train=include_train, data_dir=data_dir)
    self.feature_names = [str(i) for i in range(32*32)]
    self.label_names = [str(i) for i in range(10)]
    self.image_shape = (32, 32)

def load_svhn(include_train=True, data_dir=default_data_dir):
  f1 = data_dir + '/SVHN.pickle'
  f2 = data_dir + '/SVHN1.pickle'
  f3 = data_dir + '/SVHN2.pickle'
  f4 = data_dir + '/SVHN3.pickle'

  if not os.path.exists(f1):
    print('Dataset not found, downloading and preprocessing...')
    download_and_preprocess_svhn(data_dir)
  classes = np.array([0,1,2,3,4,5,6,7,8,9])
  limit = 200000
  with open(f1, 'rb') as f:
    save = pickle.load(f)
    train_labels = save['train_labels'][:limit]
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
  def reformat(dataset, labels):
    dataset = dataset.reshape(
      (-1, 32, 32, 1)).astype(np.float32)
    labels = labels.astype(np.int32)
    return dataset, labels
  Xv, yv = reformat(valid_dataset, valid_labels)
  Xt, yt = reformat(test_dataset, test_labels)
  if include_train:
    with open(f2, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset1'][:limit]
      del save
    X, y = reformat(train_dataset, train_labels)
  else:
    X, y = Xv, yv
  return X, y, Xv, yv, Xt, yt

def download_and_preprocess_svhn(data_dir):
  """
  Adapted from https://github.com/hangyao/street_view_house_numbers/blob/master/1_preprocess_single.ipynb
  """
  f1 = data_dir + '/SVHN.pickle'
  f2 = data_dir + '/SVHN1.pickle'
  f3 = data_dir + '/SVHN2.pickle'
  f4 = data_dir + '/SVHN3.pickle'

  url = 'http://ufldl.stanford.edu/housenumbers/'

  def maybe_download(filename, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
      print('Attempting to download:', filename)
      filename, _ = urlretrieve(url + filename, filename)
      print('\nDownload Complete!')
    statinfo = os.stat(filename)
    return filename

  train_matfile = maybe_download('svhn/train_32x32.mat')
  test_matfile = maybe_download('svhn/test_32x32.mat')
  extra_matfile = maybe_download('svhn/extra_32x32.mat')

  train_data = scipy.io.loadmat('svhn/train_32x32.mat', variable_names='X').get('X')
  train_labels = scipy.io.loadmat('svhn/train_32x32.mat', variable_names='y').get('y')
  test_data = scipy.io.loadmat('svhn/test_32x32.mat', variable_names='X').get('X')
  test_labels = scipy.io.loadmat('svhn/test_32x32.mat', variable_names='y').get('y')
  extra_data = scipy.io.loadmat('svhn/extra_32x32.mat', variable_names='X').get('X')
  extra_labels = scipy.io.loadmat('svhn/extra_32x32.mat', variable_names='y').get('y')

  print(train_data.shape, train_labels.shape)
  print(test_data.shape, test_labels.shape)
  print(extra_data.shape, extra_labels.shape)

  train_labels[train_labels == 10] = 0
  test_labels[test_labels == 10] = 0
  extra_labels[extra_labels == 10] = 0

  random.seed()

  n_labels = 10
  valid_index = []
  valid_index2 = []
  train_index = []
  train_index2 = []
  for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,0] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,0] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,0] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,0] == (i))[0][200:].tolist())

  random.shuffle(valid_index)
  random.shuffle(train_index)
  random.shuffle(valid_index2)
  random.shuffle(train_index2)

  valid_data = np.concatenate((extra_data[:,:,:,valid_index2], train_data[:,:,:,valid_index]), axis=3).transpose((3,0,1,2))
  valid_labels = np.concatenate((extra_labels[valid_index2,:], train_labels[valid_index,:]), axis=0)[:,0]
  train_data_t = np.concatenate((extra_data[:,:,:,train_index2], train_data[:,:,:,train_index]), axis=3).transpose((3,0,1,2))
  train_labels_t = np.concatenate((extra_labels[train_index2,:], train_labels[train_index,:]), axis=0)[:,0]
  test_data = test_data.transpose((3,0,1,2))
  test_labels = test_labels[:,0]

  print(train_data_t.shape, train_labels_t.shape)
  print(test_data.shape, test_labels.shape)
  print(valid_data.shape, valid_labels.shape)

  image_size = 32  # Pixel width and height.
  pixel_depth = 255.0  # Number of levels per pixel.

  def im2gray(image):
    '''Normalize images'''
    image = image.astype(float)
    # Use the Conversion Method in This Paper:
    # [http://www.eyemaginary.com/Rendering/TurnColorsGray.pdf]
    image_gray = np.dot(image, [[0.2989],[0.5870],[0.1140]])
    return image_gray

  train_data_c = im2gray(train_data_t)[:,:,:,0]
  test_data_c = im2gray(test_data)[:,:,:,0]
  valid_data_c = im2gray(valid_data)[:,:,:,0]

  print(train_data_c.shape, train_labels_t.shape)
  print(test_data_c.shape, test_labels.shape)
  print(valid_data_c.shape, valid_labels.shape)

  def GCN(image, min_divisor=1e-4):
    """Global Contrast Normalization"""

    imsize = image.shape[0]
    mean = np.mean(image, axis=(1,2), dtype=float)
    std = np.std(image, axis=(1,2), dtype=float, ddof=1)
    std[std < min_divisor] = 1.
    image_GCN = np.zeros(image.shape, dtype=float)

    for i in np.arange(imsize):
      image_GCN[i,:,:] = (image[i,:,:] - mean[i]) / std[i]

    return image_GCN

  train_data_GCN = GCN(train_data_c)
  test_data_GCN = GCN(test_data_c)
  valid_data_GCN = GCN(valid_data_c)

  print(train_data_GCN.shape, train_labels_t.shape)
  print(test_data_GCN.shape, test_labels.shape)
  print(valid_data_GCN.shape, valid_labels.shape)

  pickle_file = f1

  try:
    f = open(pickle_file, 'wb')
    save = {
      #'train_dataset': train_data_GCN,
      'train_labels': train_labels_t,
      'valid_dataset': valid_data_GCN,
      'valid_labels': valid_labels,
      'test_dataset': test_data_GCN,
      'test_labels': test_labels,
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)

  pickle_file = f2

  try:
    f = open(pickle_file, 'wb')
    save = { 'train_dataset1': train_data_GCN[:200000], }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)

  pickle_file = f3

  try:
    f = open(pickle_file, 'wb')
    save = { 'train_dataset2': train_data_GCN[200000:400000], }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)

  pickle_file = f4

  try:
    f = open(pickle_file, 'wb')
    save = { 'train_dataset3': train_data_GCN[400000:], }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)

if __name__ == '__main__':
  import pdb
  dataset = SVHN()
  pdb.set_trace()
  pass
