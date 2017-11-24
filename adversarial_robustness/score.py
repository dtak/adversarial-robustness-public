import numpy as np
from functools import total_ordering
from adversarial_robustness.utils import *
import sklearn.metrics as skm

@total_ordering
class Score():
  def __init__(self, y_true, logits):
    self.y_true = y_true
    self.logits = logits

  @property
  def y_pred(self):
    return np.argmax(self.logits, 1)

  @cachedproperty
  def probs(self):
    return softmax(self.logits)

  @cachedproperty
  def logps(self):
    return logits_to_logprobs(self.logits)

  @cachedproperty
  def is_binary(self):
    return self.logits.shape[1] == 2

  @cachedproperty
  def avgtype(self):
    if self.is_binary:
      return 'binary'
    else:
      return 'macro'

  @cachedproperty
  def accuracy(self):
    return skm.accuracy_score(self.y_true, self.y_pred)

  @cachedproperty
  def f1(self):
    return skm.f1_score(self.y_true, self.y_pred, average=self.avgtype)

  @cachedproperty
  def roc_auc(self):
    return skm.roc_auc_score(self.y_true, self.probs[:,1])

  @property
  def auc(self): return self.roc_auc

  @property
  def auroc(self): return self.roc_auc

  @cachedproperty
  def confusion_matrix(self):
    return skm.confusion_matrix(self.y_true, self.y_pred)

  @cachedproperty
  def best_objective(self):
    if self.is_binary:
      return self.roc_auc
    else:
      return self.f1

  def __lt__(self, accuracy):
    return self.accuracy < accuracy

  def __eq__(self, accuracy):
    return self.accuracy == accuracy

  def __repr__(self):
    s = 'Accuracy: {:.2%}\nF1 Score: {:.4f}'.format(self.accuracy, self.f1)
    if self.is_binary:
      s += '\nROC AUC: {:.4f}'.format(self.roc_auc)
    return s
