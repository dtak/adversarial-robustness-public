import os
from six.moves.urllib.request import urlretrieve
from tensorflow.examples.tutorials.mnist import input_data
from adversarial_robustness.dataset import *

class MNIST(Dataset):
  def __init__(self, data_dir=default_data_dir, **kwargs):
    self.X, self.y, self.Xt, self.yt = load_mnist(data_dir)
    self.feature_names = [str(i) for i in range(728)]
    self.label_names = [str(i) for i in range(10)]
    self.image_shape = (28, 28)

def load_mnist(datadir=default_data_dir):
  if not os.path.exists(datadir):
    os.makedirs(datadir)
  base_url = 'http://yann.lecun.com/exdb/mnist/'
  for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
    if not os.path.exists(os.path.join(datadir, filename)):
      urlretrieve(base_url + filename, os.path.join(datadir, filename))
  mnist = input_data.read_data_sets(datadir)
  X, y = mnist.train.images, mnist.train.labels
  Xt, yt = mnist.test.images, mnist.test.labels
  return X, y, Xt, yt

if __name__ == '__main__':
  import pdb
  dataset = MNIST()
  pdb.set_trace()
  pass
