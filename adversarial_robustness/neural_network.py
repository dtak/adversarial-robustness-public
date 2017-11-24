from __future__ import absolute_import
from __future__ import print_function
import uuid
import time
import os
import numpy as np
import tensorflow as tf
import six
import six.moves.cPickle as pickle
from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty

from adversarial_robustness.utils import *
from adversarial_robustness.score import Score

"""
Class attempting to make Tensorflow models more object-oriented
and similar to sklearn's fit/predict interface.
"""
@add_metaclass(ABCMeta)
class NeuralNetwork():
  def __init__(self, name=None, dtype=tf.float32, **kwargs):
    self.vals = None
    self.name = (name or str(uuid.uuid4()))
    self.dtype = dtype
    self.setup_model(**kwargs)
    assert(hasattr(self, 'X'))
    assert(hasattr(self, 'y'))
    assert(hasattr(self, 'logits'))

  def setup_model(self, X=None, y=None):
    with tf.name_scope(self.name):
      self.X = tf.placeholder(self.dtype, self.x_shape, name="X") if X is None else X
      self.y = tf.placeholder(self.dtype, self.y_shape, name="y") if y is None else y
      self.is_train = tf.placeholder_with_default(
          tf.constant(False, dtype=tf.bool), shape=(), name="is_train")
    self.model = self.rebuild_model(self.X)

  @property
  def logits(self):
    return self.model[-1]

  def rebuild_model(self, X, reuse=None):
    """Define all of your Tensorflow variables here, making sure to scope them
    under `self.name`, and also making sure to return a list/tuple whose final element
    is your network's logits. In subclasses, remember to call super!"""

  @abstractproperty
  def x_shape(self):
    """Specify the shape of X; for MNIST, this could be [None, 784]"""

  @abstractproperty
  def y_shape(self):
    """Specify the shape of y; for MNIST, this would be [None, 10]"""

  @property
  def num_features(self):
    """Helper to return the dimensionality of X (aka D)"""
    return np.product(self.x_shape[1:])

  @property
  def num_classes(self):
    """Helper to return the dimensionality of y (aka K)"""
    return np.product(self.y_shape[1:])

  def eval(self, tf_quantity, batches=20, batch_size=128, **kwargs):
    """Evaluate the average value of a symbolic quantity over several batches."""
    with tf.Session() as sess:
      self.init(sess)
      batch_iterator = self.minibatches(kwargs, batch_size=batch_size)
      quantity = 0
      n_batches = 0
      for i, feed_dict in enumerate(batch_iterator):
        if i >= batches:
          break
        quantity += sess.run(tf_quantity, feed_dict=feed_dict)
        n_batches += 1
    return quantity / float(n_batches)

  ###################################################
  # Useful (cached) functions of our network outputs
  #
  def cross_entropy_with(self, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y))

  @cachedproperty
  def preds(self):
    """Symbolic TF variable returning an Nx1 vector of predictions"""
    return tf.argmax(self.logits, axis=1)

  @cachedproperty
  def probs(self):
    """Symbolic TF variable returning an NxK vector of probabilities"""
    return tf.nn.softmax(self.logits)

  @cachedproperty
  def logps(self):
    """Symbolic TF variable returning an Nx1 vector of log-probabilities"""
    return self.logits - tf.reduce_logsumexp(self.logits, 1, keep_dims=True)

  @cachedproperty
  def certainty_sensitivity(self):
    """Symbolic TF variable returning the gradients (wrt each input component)
    of the sum of log probabilities, also interpretable as the sensitivity of
    the model's certainty to changes in `X` or the model's score function"""
    crossent_w_1 = self.cross_entropy_with(tf.ones_like(self.y) / self.num_classes)
    return tf.gradients(crossent_w_1, self.X)[0]

  @cachedproperty
  def grad_sum_logps(self):
    """Same as `certainty_sensitivity`, but scaled differently (since we're not taking
    any averages), without requiring you to pass a placeholder for y.
    The reason why we don't just always use this version is so we can have similar
    scalings for double backprop / certainty insensitivity / cross entropy
    (they all use tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat,y))."""
    return tf.gradients(self.logps, self.X)[0]

  @cachedproperty
  def l1_certainty_sensitivity(self):
    """Cache the L1 loss of that product"""
    return tf.reduce_mean(tf.abs(self.certainty_sensitivity))

  @cachedproperty
  def l2_certainty_sensitivity(self):
    """Cache the L2 loss of that product"""
    return tf.nn.l2_loss(self.certainty_sensitivity)

  @cachedproperty
  def l1_weights(self):
    """L1 loss for the weights of the network"""
    return tf.add_n([tf.reduce_sum(tf.abs(v)) for v in tf.trainable_variables() if v in self.vars])

  @cachedproperty
  def l2_weights(self):
    """L2 loss for the weights of the network"""
    return tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v in self.vars])

  @cachedproperty
  def cross_entropy(self):
    """Symbolic TF variable returning information distance between the model's
    predictions and the true labels y"""
    return self.cross_entropy_with(self.y)

  @cachedproperty
  def cross_entropy_grads(self):
    """Symbolic TF variable returning the input gradients of the cross entropy.
    Note that if you pass in y=(1^{NxK})/K, this returns the same value as
    certainty_sensitivity."""
    return tf.gradients(self.cross_entropy, self.X)[0]

  @cachedproperty
  def l1_double_backprop(self):
    """L1 loss of the sensitivity of the cross entropy"""
    return tf.reduce_sum(tf.abs(self.cross_entropy_grads))

  @cachedproperty
  def l2_double_backprop(self):
    """L2 loss of the sensitivity of the cross entropy"""
    return tf.nn.l2_loss(self.cross_entropy_grads)

  @cachedproperty
  def binary_logits(self):
    assert(self.num_classes == 2)
    return self.logps[:,1] - self.logps[:,0]

  @cachedproperty
  def binary_logit_grads(self):
    return tf.gradients(self.binary_logits, self.X)[0]

  @cachedproperty
  def accuracy(self):
    return tf.reduce_mean(tf.cast(tf.equal(
      self.preds, tf.argmax(self.y, 1)), dtype=tf.float32))

  #############################################
  # Predicting
  #
  def score(self, X, y):
    """Function that takes numpy arrays `X` (in NxD) and `y` (in either NxK or
    Nx1) and returns the model's predictive accuracy"""
    with tf.Session() as sess:
      self.init(sess)
      return self.score_(sess, X, y)

  def predict(self, X, n=1000):
    """Function that takes numpy arrays `X` (in NxD) and returns Nx1 predictions (batched)"""
    with tf.Session() as sess:
      self.init(sess)
      preds = self.predict_(sess, X[:n])
      for i in range(n, len(X), n):
        preds = np.hstack((preds, self.predict_(sess, X[i:i+n])))
    return preds

  def predict_proba(self, X):
    """Function that takes numpy arrays `X` (in NxD) and returns NxK predicted
    probabilities"""
    with tf.Session() as sess:
      self.init(sess)
      return self.predict_proba_(sess, X)

  # private functions that implement the above methods without starting a new
  # Tensorflow session
  def predict_proba_(self, sess, X):
      return sess.run(self.probs, feed_dict={ self.X: X })

  def predict_(self, sess, X):
    return sess.run(self.preds, feed_dict={ self.X: X })

  def score_(self, sess, X, y, n=1000):
    if len(y.shape) > 1: y = np.argmax(y, axis=1)
    with tf.Session() as sess:
      self.init(sess)
      logits = sess.run(self.logits, feed_dict={ self.X: X[:n] })
      for i in range(n, len(X), n):
        logits = np.vstack((
          logits, sess.run(self.logits, feed_dict={ self.X: X[i:i+n] })))
    return Score(y, logits)

  ###################################################
  # Explaining
  #
  def input_gradients(self, X, y=None, **kw):
    """Function that takes numpy arrays `X` (in NxD) and optionally some set of
    NxK labels `y` and returns the model's cross-entropy input gradients.
    Useful local linear approximation of what the model is doing locally."""
    with tf.Session() as sess:
      self.init(sess)
      return self.input_gradients_(sess, X, y)

  def batch_input_gradients(self, X, y, n=1000, **kw):
    """Batched version of input gradients"""
    with tf.Session() as sess:
      self.init(sess)
      grads = self.input_gradients_(sess, X[:n], y[:n], **kw)
      for i in range(n, len(X), n):
        grads = np.vstack((grads,
          self.input_gradients_(sess, X[i:i+n], y[i:i+n], **kw)))
    return grads

  def input_gradients_(self, sess, X, y=None, logits=False):
    if y is None:
      return sess.run(self.grad_sum_logps, feed_dict={ self.X: X })
    elif logits and self.num_classes == 2:
      return sess.run(self.binary_logit_grads, feed_dict={ self.X: X })
    elif isint(y):
      y = onehot(np.array([y]*len(X)), self.num_classes)
    return sess.run(self.cross_entropy_grads, feed_dict={ self.X: X, self.y: y })

  #############################################
  # Training
  #
  def minibatches(self, kwargs, batch_size=128, num_epochs=32):
    """Helper to generate minibatches of the training set (called by `fit`).
    Currently this just iterates sequentially through `kwargs['X']` for
    `num_epochs`, taking `batch_size` examples per iteration. If you need
    fancier behavior, you can override this function or provide your own batch
    generator to pass to `fit_batches`. """
    assert('X' in kwargs or self.X in kwargs)
    X = kwargs.get('X', kwargs.get(self.X, None))
    n = int(np.ceil(len(X) / batch_size))
    tensors = self.parse_placeholders(kwargs)
    for i in range(int(num_epochs * n)):
      idx = slice((i%n)*batch_size, ((i%n)+1)*batch_size)
      feed = {}
      for var, value in six.iteritems(tensors):
        feed[var] = value[idx]
      yield feed

  def loss_function(self,
      softmax_temperature=1.,
      l1_certainty_sensitivity=0., l2_certainty_sensitivity=0.,
      l1_weights=0., l2_weights=0.,
      l1_double_backprop=0., l2_double_backprop=0.):
    """By default, still just use the cross entropy as the loss, but allow
    users to penalize the L1 or L2 norm of the input sensitivity by passing
    the given params below to `fit`."""
    if softmax_temperature == 1:
      log_likelihood = self.cross_entropy
    else:
      log_likelihood = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits/softmax_temperature, labels=self.y))
    log_prior = 0
    for reg in [
        'l1_certainty_sensitivity', 'l2_certainty_sensitivity',
        'l1_double_backprop', 'l2_double_backprop',
        'l1_weights', 'l2_weights']:
      if eval(reg) > 0:
        log_prior += eval(reg) * getattr(self, reg)
    return log_likelihood + log_prior

  def equalizing(self, prop, num_batches=10, **kwargs):
    """Estimate the ratio between the cross entropy and some other penalty
    term from random initialization. This is helpful if you want to
    express the magnitude of a regularization term in a more intutive way."""
    ratio_sum = 0
    ratio_count = 0
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i, feed in enumerate(self.minibatches(kwargs)):
        if i >= num_batches: break
        feed[self.is_train] = True
        g1 = getattr(self, prop).eval(feed_dict=feed)
        g2 = self.cross_entropy.eval(feed_dict=feed)
        ratio_sum += g2 / g1
        ratio_count += 1
    return ratio_sum / ratio_count

  def adversarial_training_loss(self, fgsm_eps, clip_min, clip_max,
      l2_double_backprop=0.0):
    """
    Symbolic loss function term for our loss on FGSM examples, based on code from
    https://github.com/tensorflow/cleverhans/blob/80e57f67dc134f9cf954ad23784904d38af07577/cleverhans_tutorials/mnist_tutorial_tf.py#L139-L151.
    Importantly, we stop gradients to ensure that our model doesn't minimize this term by making its adversarial examples easier.
    """
    preds_max = tf.reduce_max(self.logits, 1, keep_dims=True)
    y_symb = tf.stop_gradient(tf.to_float(tf.equal(self.logits, preds_max)))
    y_symb = y_symb / tf.reduce_sum(y_symb, 1, keep_dims=True)
    adv_loss = self.cross_entropy_with(y_symb)
    loss_grad, = tf.gradients(adv_loss, self.X)
    normalized_grad = tf.stop_gradient(tf.sign(loss_grad))
    X_adv = tf.stop_gradient(tf.clip_by_value(
      self.X + fgsm_eps * normalized_grad,
      clip_min,
      clip_max))
    model2 = self.rebuild_model(X_adv, reuse=True)
    logits2 = model2[-1]
    cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y))
    loss = cross_entropy2
    if l2_double_backprop > 0:
      xent2_grads = tf.gradients(cross_entropy2, X_adv)[0]
      loss += l2_double_backprop * tf.nn.l2_loss(xent2_grads)
    return loss

  def fit(self, X, y, batch_size=128, num_epochs=32, **kwargs):
    """Trains the model for the specified duration. See `minibatches` and
    `fit_batches` for option definitions"""
    batch_kw = {}
    batch_kw.update(kwargs)
    batch_kw['X'] = X
    batch_kw['y'] = y
    data = self.minibatches(batch_kw, batch_size=batch_size, num_epochs=num_epochs)
    return self.fit_batches(data, **kwargs)

  def fit_batches(self, batches,
      optimizer=None, loss_fn=None, print_every=None,
      init=False, learning_rate=0.001, epsilon=1e-8,
      call_every=None, callback=None, capper=None, **kwargs):
    """
    Actually fit the model using the `batches` iterator, which should yield
    successive feed_dicts containing new examples. This is designed to be
    flexible so you can either iterate through a giant array in memory or
    have a queue loading files and doing preprocessing (though if you're
    really doing serious training, you're probably better off making all
    of the preprocessing symbolic and writing a clunky megascript).

    You can pass a custom optimizer, loss function, gradient capping function,
    or callback. You can also choose to reinitialize the model with its current param
    values before starting.
    """
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    if loss_fn is None:
      loss_fn = self.loss_function(**kwargs)

    grads_and_vars = optimizer.compute_gradients(loss_fn)

    if capper is not None:
      grads_and_vars = capper(grads_and_vars)

    train_op = optimizer.apply_gradients(grads_and_vars)
    t = time.time()

    with tf.Session() as sess:
      # Init
      sess.run(tf.global_variables_initializer())
      if init: self.init(sess)

      # Train
      for i, batch in enumerate(batches):
        batch[self.is_train] = True
        _, loss = sess.run([train_op, loss_fn], feed_dict=batch)
        if print_every and i % print_every == 0:
          print('Batch {}, loss {}, {}s'.format(i, loss, time.time() - t))
        if call_every and i % call_every == 0:
          callback(sess, self, batch, i)

      # Save
      self.vals = [v.eval() for v in self.vars]

  @classmethod
  def distill(cls, X, y, T=10., **kwargs):
    """Implement defensive distillation -- train the same model twice
    using a softmax temperature, initially on one-hot labels
    and subsequently on its own predictions."""
    net1 = cls()
    net1.fit(X, y, softmax_temperature=T, **kwargs)
    yhat = tf.nn.softmax(net1.logits/T)

    with tf.Session() as sess:
      net1.init(sess)
      ysmooth = yhat.eval(feed_dict={ net1.X: X[:1000] })
      for i in range(1000, len(X), 1000):
        ysmooth = np.vstack((ysmooth, yhat.eval(feed_dict={ net1.X: X[i:i+1000] })))

    net2 = cls()
    net2.fit(X, ysmooth, softmax_temperature=T, **kwargs)
    return net2

  #############################################
  # Adversarial attacks
  #
  def gradient_sign_attack(self, sess, X, y, eps=0.1,
      clip_min=None, clip_max=None, epochs=1, feed_dict={}):
    """Applies the Fast Gradient Sign method of generating adversarial examples
    for a variable number of `epochs` with a perturbation strength `eps`,
    clipping the values so they don't go crazy."""
    if clip_min is None: clip_min = np.min(X)
    if clip_max is None: clip_max = np.max(X)
    adv_X = X
    while epochs > 0:
      feed = { self.X: adv_X, self.y: y }
      feed.update(feed_dict)
      grads = sess.run(self.cross_entropy_grads, feed_dict=feed)
      adv_X = np.clip(adv_X + np.sign(grads)*eps, clip_min, clip_max)
      epochs -= 1
    return adv_X

  def generate_fgsm_examples(self, sess, X, y, eps=0.1, **kwargs):
    """Wrapper for running the FGSM that makes sure it's moving away from the
    true labels `y`."""
    assert(eps > 0)
    return self.gradient_sign_attack(sess, X, y, eps=eps, **kwargs)

  def generate_tgsm_examples(self, sess, X, y, targets=None, eps=0.1, **kwargs):
    """Wrapper for running the TGSM that makes sure it's moving _towards_ the
    adversarial `targets`."""
    assert(eps > 0)
    if targets is None:
      targets = onehot((np.argmax(y, 1) + 1) % self.num_classes, self.num_classes)
    elif isint(targets):
      targets = onehot([targets]*len(X), self.num_classes)
    return self.gradient_sign_attack(sess, X, targets, eps=-eps, **kwargs)

  def generate_jsma_examples(self, sess, X, y, targets=None,
      clip_min=None, clip_max=None, theta=1., gamma=0.25):
    """Wrapper around Cleverhans' underlying JSMA generation code"""
    from cleverhans import attacks_tf
    if clip_min is None: clip_min = np.min(X)
    if clip_max is None: clip_max = np.max(X)
    if targets is None:
      targets = onehot((np.argmax(y, 1) + 1) % self.num_classes, self.num_classes)
    elif isint(targets):
      targets = onehot([targets]*len(X), self.num_classes)
    jacobian = attacks_tf.jacobian_graph(self.logits, self.X, self.num_classes)
    return attacks_tf.jsma_batch(sess, self.X, self.logits, jacobian, X,
      theta=theta, gamma=gamma, clip_min=clip_min, clip_max=clip_max,
      nb_classes=self.num_classes, y_target=targets)

  def adversarial_examples(self, method, X, y=None, **kwargs):
    """Wrapper for generating adversarial examples"""
    with tf.Session() as sess:
      self.init(sess)
      if y is None:
        y = onehot(self.predict_(sess, X), self.num_classes)
      method = getattr(self, 'generate_'+method+'_examples')
      return method(sess, X, y, **kwargs)

  #############################################
  # Persisting/loading variables
  #
  @property
  def vars(self):
    """Find all of the (trainable) variables that belong to this model"""
    return tf.get_default_graph().get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

  def init(self, sess):
    """Assign all of the stored `vals` to our Tensorflow `vars`. Run this after
    you start a new session."""
    if self.vals is None:
      sess.run(tf.global_variables_initializer())
    else:
      for var, val in zip(self.vars, self.vals):
        sess.run(var.assign(val))

  def save(self, filename):
    """Save our variables as lists of numpy arrays, because they're a bit easier to work
    with than giant tensorflow directories."""
    with open(filename, 'wb') as f:
      pickle.dump(self.vals, f)

  def load(self, filename):
    """Load saved variables"""
    with open(filename, 'rb') as f:
      self.vals = pickle.load(f)

  def parse_placeholders(self, kwargs):
    """
    Figure out which elements of a dictionary are either tf.placeholders
    or strings referencing attributes that are tf.placeholders, then ensure
    we populate the feed dict with actual placeholders for easy feeding later.
    """
    feed = {}
    for dictionary in [kwargs, kwargs.get('feed_dict', {})]:
      for key, val in six.iteritems(dictionary):
        attr = getattr(self, key) if isinstance(key, str) and hasattr(self, key) else key
        if type(attr) == type(self.X):
          if len(attr.shape) > 1:
            if attr.shape[0].value is None:
              feed[attr] = val
    return feed
