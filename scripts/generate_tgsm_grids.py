import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(root_dir)
from adversarial_robustness.cnns import *
from adversarial_robustness.plot_helpers import *
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
    help="Place to save TGSM examples")
parser.add_argument(
    "--dataset", type=str,
    help="which dataset")
parser.add_argument(
    "--eps", type=float,
    help="how much to move")
parser.add_argument(
    "--epochs", type=int, default=26,
    help="how many steps")
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
template = adv_X_dir + '/tgsm-target{}-eps{}-epoch{}.npy'

Xt = dataset.Xt
yt = dataset.yt
eps = FLAGS.eps
kwargs = { 'clip_min': Xt.min(), 'clip_max': Xt.max(), 'eps': -eps }
n_per_class = 5
epochs = FLAGS.epochs

Xta = np.concatenate([Xt[np.argwhere(yt==j)[:n_per_class,0]] for j in range(10)])

def gridshow(epoch, offset=0):
  fig = plt.figure(figsize=(10,10))
  title = r'TGSM, '+str(epoch)+r' iteration(s) at $\epsilon='+str(eps)+'$, '+FLAGS.model_label
  plt.gcf().suptitle(title, y=0.925, fontsize=16)
  for targ in range(10):
    if epoch > 0:
      filename = template.format(targ, str(eps).replace('.','pt'), epoch)
      advX = np.load(filename)
    for orig in range(10):
      ax = plt.subplot(10, 10, orig*10 + targ + 1)
      if orig == 9: plt.xlabel(dataset.label_names[targ], fontsize=9)
      if targ == 0: plt.ylabel(dataset.label_names[orig], fontsize=9)
      if orig == targ:
        dataset.imshow_example(Xta[orig*5+offset])
        outline(ax)
      elif epoch == 0:
        dataset.imshow_example(Xta[orig*5+offset])
      else:
        dataset.imshow_example(advX[orig*5+offset])
  plt.figtext(0.075, 0.5, r'Original Label', rotation='vertical', va='center', fontsize=14)
  plt.figtext(0.5, 0.075, r'Target Label', va='top', ha='center', fontsize=14)
  filename = template.format('grid', str(eps).replace('.','pt'), epoch)
  fig.savefig(filename.replace('.npy', '.png'), bbox_inches='tight', pad_inches=0.01)
  plt.close(fig)

if not os.path.exists(template.format(9, str(eps).replace('.','pt'), epochs-1)):
  with tf.Session() as sess:
    cnn.init(sess)
    for target in range(10):
      advX = Xta
      advY = onehot([target]*len(Xta), 10)
      for epoch in range(0, epochs):
        filename = template.format(target, str(eps).replace('.','pt'), epoch)
        if os.path.exists(filename):
          advX = np.load(filename)
        elif epoch > 0:
          advX = cnn.gradient_sign_attack(sess, advX, advY, **kwargs)
          np.save(filename, advX)
        if not os.path.exists(filename.replace('.npy', '-preds.npy')):
          preds = cnn.predict_proba_(sess, advX)
          np.save(filename.replace('.npy', '-preds.npy'), preds)

if not os.path.exists(template.format('grid', str(eps).replace('.','pt'), epochs-1).replace('.npy', '.png')):
  for epoch in range(epochs):
    gridshow(epoch)
