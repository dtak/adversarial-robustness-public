from collections import defaultdict
import numpy as np

class cachedproperty(object):
  """Simplified version of https://github.com/pydanny/cached-property"""
  def __init__(self, function):
    self.__doc__ = getattr(function, '__doc__')
    self.function = function

  def __get__(self, instance, klass):
    if instance is None: return self
    value = instance.__dict__[self.function.__name__] = self.function(instance)
    return value

def isint(x):
  return isinstance(x, (int, np.int32, np.int64))

def onehot(Y, K=None):
  if K is None:
    K = np.unique(Y)
  elif isint(K):
    K = list(range(K))
  data = np.array([[y == k for k in K] for y in Y]).astype(int)
  return data

def softmax(z):
  assert len(z.shape) == 2
  s = np.max(z, axis=1)
  s = s[:, np.newaxis]
  e_x = np.exp(z - s)
  div = np.sum(e_x, axis=1)
  div = div[:, np.newaxis]
  return e_x / div

def logits_to_logprobs(z):
  from scipy.misc import logsumexp
  return z - logsumexp(z, axis=1, keepdims=True)

class lazydict(defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret

def adversarial_grid(sess, model, adv_Xes, y, epses=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7], verbose=True):
  preds = {}
  scores = {}
  for eps in epses:
    preds[eps] = model.predict_proba_(sess, adv_Xes[eps])
    scores[eps] = np.mean(np.argmax(preds[eps], 1) == np.argmax(y, 1))
    if verbose:
      print('ε={} has accuracy {}'.format(eps, scores[eps]))
  return preds, scores
