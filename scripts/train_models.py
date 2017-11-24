import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
import sys
sys.path.append(root_dir)
from adversarial_robustness.cnns import *
from adversarial_robustness.datasets.svhn import SVHN
from adversarial_robustness.datasets.notmnist import notMNIST
from adversarial_robustness.datasets.mnist import MNIST
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--savedir", type=str,
    help="Place to save model")
parser.add_argument(
    "--name", type=str, default="",
    help="Model name")
parser.add_argument(
    "--dataset", type=str, default="",
    help="Dataset")
parser.add_argument(
    "--l2cs", type=float, default=0.0,
    help="L2 certainty sensitivity penalty")
parser.add_argument(
    "--l2dbl", type=float, default=0.0,
    help="L2 double backprop penalty")
parser.add_argument(
    "--lr", type=float, default=0.0002,
    help="learning rate")
parser.add_argument(
    "--adameps", type=float, default=1e-04,
    help="adam epsilon")
parser.add_argument(
    "--advtraineps", type=float, default=0.0,
    help="adversarial training epsilon")
parser.add_argument(
    "--distilltemp", type=float, default=1.0,
    help="temperature for distillation")
parser.add_argument(
    "--batchsize", type=int, default=256,
    help="batch size")
parser.add_argument(
    "--nbatches", type=int, default=15000,
    help="number of batches")
FLAGS = parser.parse_args()

name = FLAGS.name
model_dir = FLAGS.savedir
adv_X_dir = root_dir + '/cached/fgsm'

if FLAGS.dataset == 'mnist':
  dataset = MNIST()
  CNN = MNIST_CNN
  fgsm_file = adv_X_dir + '/mnist-normal-fgsm-perturbation.npy'
elif FLAGS.dataset == 'notmnist':
  dataset = notMNIST()
  CNN = MNIST_CNN
  fgsm_file = adv_X_dir + '/notmnist-normal-fgsm-perturbation.npy'
elif FLAGS.dataset == 'svhn':
  dataset = SVHN()
  CNN = SVHN_CNN
  fgsm_file = adv_X_dir + '/svhn-normal-fgsm-perturbation.npy'

X = dataset.X
y = dataset.onehot_y
Xt = dataset.Xt[:1024]
yt = dataset.onehot_yt[:1024]
clip_min = dataset.X.min()
clip_max = dataset.X.max()

dX = np.sign(np.load(fgsm_file))[:1024]

def _fgsm(eps):
  return np.clip(Xt[:len(dX)] + eps * dX, clip_min, clip_max)

fgsm = { 0.1: _fgsm(0.1), 0.2: _fgsm(0.2), 0.3: _fgsm(0.3) }
epses = [0.1, 0.2, 0.3]

scores = {}
train_curves = {}
train_curves['batch_number'] = []
train_curves['batch_accuracy'] = []
train_curves['cross_entropy'] = []
train_curves['l2_grad_logp_true'] = []
train_curves['l2_grad_logp_rest'] = []
train_curves['l2_grad_logp_all'] = []
train_curves['l2_param_grads'] = []
train_curves['adv_accuracy'] = []
train_curves['test_accuracy'] = []

batch_size = FLAGS.batchsize
num_batches = FLAGS.nbatches
num_epochs = int(np.ceil(num_batches / (len(X) / batch_size)))
print(num_epochs)

if FLAGS.distilltemp > 1.01:
  print('distillation')
  num_batches2 = min(FLAGS.nbatches, 10000)
  num_epochs2 = int(np.ceil(num_batches2 / (len(X) / batch_size)))
  cnn2 = CNN()
  cnn2.fit(X, y, softmax_temperature=FLAGS.distilltemp, learning_rate=FLAGS.lr, epsilon=FLAGS.adameps, num_epochs=num_epochs2, batch_size=batch_size)
  yhat = tf.nn.softmax(cnn2.logits/FLAGS.distilltemp)
  with tf.Session() as sess:
    cnn2.init(sess)
    ysmooth = yhat.eval(feed_dict={ cnn2.X: X[:1000] })
    for i in range(1000, len(X), 1000):
      ysmooth = np.vstack((ysmooth, yhat.eval(feed_dict={ cnn2.X: X[i:i+1000] })))
  y = ysmooth

tf.reset_default_graph()
cnn = CNN()
cnn.l2_grad_logp_all = tf.nn.l2_loss(tf.gradients(cnn.logps, cnn.X)[0])
cnn.l2_grad_logp_true = tf.nn.l2_loss(tf.gradients(cnn.logps * cnn.y, cnn.X)[0])
cnn.l2_grad_logp_rest = tf.nn.l2_loss(tf.gradients(cnn.logps * (1-cnn.y), cnn.X)[0])

optimizer = tf.train.AdamOptimizer(
    learning_rate=FLAGS.lr,
    epsilon=FLAGS.adameps)

loss_fn = cnn.loss_function(
    softmax_temperature=FLAGS.distilltemp,
    l2_certainty_sensitivity=FLAGS.l2cs,
    l2_double_backprop=FLAGS.l2dbl)

if FLAGS.advtraineps > 1e-06:
  print('adversarial training')
  adv_loss = cnn.adversarial_training_loss(FLAGS.advtraineps, clip_min, clip_max)
  loss_fn = (loss_fn + adv_loss) / 2.0

gradients, variables = zip(*optimizer.compute_gradients(loss_fn))
cnn.l2_param_grads = tf.add_n([tf.nn.l2_loss(g) for g in gradients])
cnn.train_op = optimizer.apply_gradients(zip(gradients, variables))

batches = cnn.minibatches({ 'X': X, 'y': y }, batch_size=batch_size, num_epochs=num_epochs)
t = time.time()
i = 0
checkpoint_interval = 2500
print_interval = 500
curve_interval = 100
filenames = []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for batch in batches:
    batch[cnn.is_train] = True
    _, loss = sess.run([cnn.train_op, loss_fn], feed_dict=batch)
    if i % checkpoint_interval == 0:
      cnn.vals = [v.eval() for v in cnn.vars]
      filename = model_dir+'/{}-batch{}-cnn.pkl'.format(name, i)
      cnn.save(filename)
      filenames.append(filename)
      with open(model_dir+'/{}-batch{}-train-curves.pkl'.format(name,i), 'wb') as f:
        pickle.dump(train_curves, f)
    if i % print_interval == 0:
      print('Batch {}, loss {}, {}s'.format(i, loss, time.time() - t))
    if i % curve_interval == 0:
      values = sess.run([
        cnn.accuracy,
        cnn.l2_grad_logp_true,
        cnn.l2_grad_logp_rest,
        cnn.l2_grad_logp_all,
        cnn.l2_param_grads,
        cnn.cross_entropy,
      ], feed_dict=batch)
      train_curves['batch_number'].append(i)
      train_curves['batch_accuracy'].append(values[0])
      train_curves['l2_grad_logp_true'].append(values[1])
      train_curves['l2_grad_logp_rest'].append(values[2])
      train_curves['l2_grad_logp_all'].append(values[3])
      train_curves['l2_param_grads'].append(values[4])
      train_curves['cross_entropy'].append(values[5])
      train_curves['adv_accuracy'].append(sess.run(cnn.accuracy, feed_dict={ cnn.X: fgsm[epses[1]][:512], cnn.y: yt[:512] }))
      train_curves['test_accuracy'].append(sess.run(cnn.accuracy, feed_dict={ cnn.X: Xt[:512], cnn.y: yt[:512] }))
    i += 1
  cnn.vals = [v.eval() for v in cnn.vars]

filename = model_dir+'/{}-cnn.pkl'.format(name)
cnn.save(filename)
filenames.append(filename)

for filename in filenames:
  cnn2 = CNN()
  cnn2.load(filename)
  cnn2.save(filename)

with open(model_dir+'/{}-train-curves.pkl'.format(name), 'wb') as f:
  pickle.dump(train_curves, f)

for key, values in train_curves.items():
  if key == 'batch_number':
    continue
  fig = plt.figure()
  plt.plot(train_curves['batch_number'], values, marker='o', lw=2)
  plt.title(key)
  plt.xlabel('Minibatch')
  plt.ylabel(key)
  if 'grad' in key:
    plt.yscale('log')
  plt.savefig(model_dir+'/{}-traincurves-{}.png'.format(name,key))
  plt.close(fig)

scores[(name, 'norm')] = cnn.score(Xt, yt).accuracy
for eps in epses:
  scores[(name, eps)] = cnn.score(fgsm[eps], yt[:len(fgsm[eps])]).accuracy
print(scores)

with open(model_dir+'/{}-scores.pkl'.format(name), 'wb') as f:
  pickle.dump(scores, f)

with open(model_dir+'/{}-flags.pkl'.format(name), 'wb') as f:
  pickle.dump(vars(FLAGS), f)
