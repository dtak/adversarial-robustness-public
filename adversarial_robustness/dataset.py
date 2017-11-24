import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from adversarial_robustness.utils import *

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

"""
Wrapper class around datasets with training/test splits and basic plotting helpers.
"""
class Dataset(object):
  def __repr__(self):
    lenXv = len(self.Xv) if hasattr(self, 'Xv') else 0
    return '{} Dataset, D={}, K={}, N={}/{}/{} train/valid/test'.format(
        self.__class__.__name__, self.num_features, self.num_classes,
        len(self.X), lenXv, len(self.Xt))

  @property
  def name(self):
    return self.__class__.__name__

  @cachedproperty
  def num_features(self):
    return self.X.shape[1]

  @cachedproperty
  def num_classes(self):
    return len(np.unique(self.y))

  @cachedproperty
  def onehot_y(self):
    y = onehot(self.y)
    assert(y.shape[1] == self.onehot_yt.shape[1])
    return y

  @cachedproperty
  def onehot_yt(self):
    return onehot(self.yt)

  @cachedproperty
  def onehot_yv(self):
    return onehot(self.yv)

  @property
  def Xf(self):
    if hasattr(self, 'Xv'):
      return np.vstack((self.X, self.Xv))
    else:
      return self.X

  @property
  def yf(self):
    if hasattr(self, 'Xv'):
      return np.hstack((self.y, self.yv))
    else:
      return self.y

  @property
  def onehot_yf(self):
    return onehot(self.yf)

  def imshow_example(self, x, **kwargs):
    assert(hasattr(self, 'image_shape'))
    image = x.reshape(self.image_shape)
    imshow_kw = { 'interpolation': 'none' }
    if len(self.image_shape) == 2:
      imshow_kw['cmap'] = 'gray'
    else:
      image = -image
    imshow_kw.update(kwargs)
    plt.xticks([])
    plt.yticks([])
    return plt.imshow(image, **imshow_kw)

  def imshow_gradient(self, grad, percentile=99, **kwargs):
    assert(hasattr(self, 'image_shape'))
    image = grad.reshape(self.image_shape)
    if len(self.image_shape) == 3:
      # Convert RGB gradient to diverging BW gradient (ensuring the span isn't thrown off by outliers).
      # copied from https://github.com/PAIR-code/saliency/blob/master/saliency/visualization.py
      image = np.sum(image, axis=2)
      span = abs(np.percentile(image, percentile))
      vmin = -span
      vmax = span
      image = np.clip((image - vmin) / (vmax - vmin), -1, 1) * span
    imshow_kw = { 'cmap': 'gray', 'interpolation': 'none' }
    imshow_kw.update(kwargs)
    plt.xticks([])
    plt.yticks([])
    return plt.imshow(image, **imshow_kw)
