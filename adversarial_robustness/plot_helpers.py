import matplotlib.pyplot as plt
import numpy as np
from pylab import Rectangle

class figure_grid():
  def next_subplot(self, **kwargs):
    if self.subplots:
      self.after_each()
    self.subplots += 1
    return self.fig.add_subplot(self.rows, self.cols, self.subplots, **kwargs)

  def each_subplot(self):
    for _ in range(self.rows * self.cols):
      yield self.next_subplot()

  def title(self, title, y=1.05, fontsize=14, **kwargs):
    self.fig.suptitle(title, y=y, fontsize=fontsize, **kwargs)

  def __init__(self, rows, cols, rowheight=3, rowwidth=12, after_each=lambda: None, after_all=lambda fig: None):
    self.rows = rows
    self.cols = cols
    self.fig = plt.figure(figsize=(rowwidth, rowheight*self.rows))
    self.subplots = 0
    if after_each == 'legend':
      after_each = lambda: plt.legend(loc='best')
    self.after_each = after_each
    self.after_all = after_all

  def __enter__(self):
    return self

  def __exit__(self, _type, _value, _traceback):
    self.after_each()
    plt.tight_layout()
    self.after_all(self.fig)
    plt.show()

  next = next_subplot

def outline(ax, **kw):
  a = ax.axis()
  rec = Rectangle((a[0]-0.7,a[2]-0.2),(a[1]-a[0])+1,(a[3]-a[2])+0.4,fill=False,lw=2, **kw)
  rec = ax.add_patch(rec)
  rec.set_clip_on(False)

def sidetext(text, fontsize=8):
  left, width = 0, .5
  bottom, height = .25, .5
  right = left + width
  top = bottom + height
  ax = plt.gca()
  ax.text(left, 0.5*(bottom+top), text,
      ha='right', va='center', rotation='vertical',
      transform=ax.transAxes, fontsize=fontsize)

model_labels = {
    'normal': 'Normal',
    'distilled': 'Distilled',
    'doubleback': r'Grad Reg ($y$)',
    'insensitive': r'Grad Reg ($\frac{1}{K}$)',
    'advtrain': 'Adv. Trained',
    'double-advtrain': 'Both Defenses',
    'double_advtrain': 'Both Defenses', }

model_colors = {
  'normal': 'blue',
  'insensitive': 'lawngreen',
  'doubleback': 'green',
  'distilled': 'skyblue',
  'advtrain': 'darkred',
  'double-advtrain': 'darkgreen',
  'double_advtrain': 'darkgreen', }

model_markers = {
    'normal':'o',
    'distilled': 'p',
    'doubleback': 'v',
    'insensitive': '^',
    'advtrain': 'H',
    'double-advtrain': 'D',
    'double_advtrain': 'D', }

def plot_grads(dataset, models, model_names, show=None, titley=9.5, **kw):
  if show is None: show = list(range(len(models)))
  fig = plt.figure(figsize=(len(show),len(models)+1))
  for i in range(len(show)):
    plt.subplot(len(models)+1,len(show),i+1)
    dataset.imshow_example(dataset.Xt[[show[i]]])
    if i == 0: sidetext('Images')
  for j in range(len(models)):
    grads = models[j].input_gradients(dataset.Xt[:25])
    for i in range(len(show)):
      plt.subplot(len(models)+1,len(show),(j+1)*len(show)+i+1)
      dataset.imshow_gradient(grads[show[i]], **kw)
      if i == 0: sidetext(model_labels[model_names[j]])
  plt.subplots_adjust(hspace=0.05, wspace=0.05)
  return fig

def grad_size_boxplot(grads, model_names):
  for i, m in enumerate(model_names):
    plt.axvspan(i+0.5, i+1.5, color=model_colors[m], alpha=0.1)
  l2_norms = [np.array([max(np.linalg.norm(g), 1e-20) for g in gs]) for gs in grads]
  plt.boxplot(l2_norms, showmeans=True, flierprops={'alpha': 0.25}, meanprops={'zorder': 3})
  plt.yscale('log')
  plt.gca().set_xticklabels([model_labels[m] for m in model_names], fontsize=12)

def compare_misclassification_overlaps(scores, yt, model_names, eps=0.4):
  if len(model_names) == 2:
    from matplotlib_venn import venn2 as venn
  elif len(model_names) == 3:
    from matplotlib_venn import venn3 as venn
  else:
    raise ValueError('can only compare 2 or 3 models')
  plt.figure(figsize=(12,3))
  for i, m1 in enumerate(model_names):
    plt.subplot(1, len(model_names), i+1)
    plt.title(model_labels[m1] + ' Examples')
    modz = list(reversed(model_names))
    sets = [set(np.argwhere(~np.equal(
              np.argmax(scores[(m2,m1)][0][eps], 1),
              np.argmax(yt[:len(scores[(m2,m1)][0][eps])], 1)
            ))[:,0]) for m2 in modz]
    labels = [model_labels[m2] for m2 in modz]
    colors = [model_colors[m2] for m2 in modz]
    venn(sets, labels, colors)
