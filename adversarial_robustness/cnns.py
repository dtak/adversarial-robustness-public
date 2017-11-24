import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from adversarial_robustness.neural_network import *
from adversarial_robustness.lecun_lcn import *

class MNIST_CNN(NeuralNetwork):
  @property
  def x_shape(self): return [None, 28*28]

  @property
  def y_shape(self): return [None, 10]

  def rebuild_model(self, X, reuse=None):
    batch_norm_params = { 'is_training': self.is_train, 'decay': 0.9, 'updates_collections': None }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
      L0 = tf.reshape(X, [-1, 28, 28, 1])
      L1 = slim.conv2d(L0, 32, [5, 5], reuse=reuse, scope=self.name+'/L1')
      L2 = slim.max_pool2d(L1, [2, 2])
      L3 = slim.conv2d(L2, 64, [5, 5], reuse=reuse, scope=self.name+'/L3')
      L4 = slim.max_pool2d(L3, [2, 2])
      L4 = slim.flatten(L4)
      L5 = slim.fully_connected(L4, 1024, reuse=reuse, scope=self.name+'/L5')
      L5 = slim.dropout(L5, is_training=self.is_train)
      L6 = slim.fully_connected(L5, 10,
          activation_fn=None, normalizer_fn=None, reuse=reuse, scope=self.name+'/L6')
      return [L1, L2, L3, L4, L5, L6]

class SVHN_CNN(NeuralNetwork):
  @property
  def x_shape(self): return [None, 32, 32, 1]

  @property
  def y_shape(self): return [None, 10]

  def rebuild_model(self, X, reuse=None):
    batch_norm_params = {'is_training': self.is_train, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
      L0 = tf.reshape(X, [-1, 32, 32, 1])
      L0 = LecunLCN(L0, [-1, 32, 32, 1])
      L1 = slim.conv2d(L0, 32, [5, 5], reuse=reuse, scope=self.name+'/L1')
      L2 = slim.max_pool2d(L1, [2, 2])
      L3 = slim.conv2d(L2, 64, [5, 5], reuse=reuse, scope=self.name+'/L3')
      L4 = slim.max_pool2d(L3, [2, 2])
      L4 = slim.flatten(L4)
      L5 = slim.fully_connected(L4, 1024, reuse=reuse, scope=self.name+'/L5')
      L5 = slim.dropout(L5, is_training=self.is_train)
      L6 = slim.fully_connected(L5, 10,
          activation_fn=None, normalizer_fn=None, reuse=reuse, scope=self.name+'/L6')
      return [L1, L2, L3, L4, L5, L6]
