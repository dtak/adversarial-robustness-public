import pickle
import numpy as np
import tensorflow as tf
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(root_dir)
from adversarial_robustness.cnns import *
from adversarial_robustness.neural_network import *
from adversarial_robustness.datasets.svhn import SVHN
from adversarial_robustness.datasets.notmnist import notMNIST
from adversarial_robustness.datasets.mnist import MNIST
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path", type=str,
    help="Place where model is saved")
parser.add_argument(
    "--model-label", type=str,
    help="What to call model in plots")
parser.add_argument(
    "--example-path", type=str,
    help="Place to save JSMA examples")
parser.add_argument(
    "--dataset", type=str,
    help="which dataset")
parser.add_argument(
    "--eps", type=float,
    help="how much to move")
FLAGS = parser.parse_args()

if FLAGS.dataset == 'mnist':
  dataset = MNIST()
  CNN = MNIST_CNN
elif FLAGS.dataset == 'notmnist':
  dataset = notMNIST()
  CNN = MNIST_CNN
elif FLAGS.dataset == 'svhn':
  dataset = SVHN(include_train=False)
  CNN = SVHN_CNN

tf.reset_default_graph()
cnn = CNN()
cnn.load(FLAGS.model_path)

adv_X_dir = FLAGS.example_path
os.system('mkdir -p {}'.format(adv_X_dir))
template = adv_X_dir + '/jsma-target{}.npy'

Xt = dataset.Xt
yt = dataset.yt
kwargs = { 'clip_min': Xt.min(), 'clip_max': Xt.max() }
n_per_class = 5
Xta = np.concatenate([Xt[np.argwhere(yt==j)[:n_per_class,0]] for j in range(10)])

with tf.Session() as sess:
  cnn.init(sess)
  for target in range(10):
    print(target)
    advX = cnn.generate_jsma_examples(sess, Xta, None, target)
    filename = template.format(target)
    np.save(filename, advX)
