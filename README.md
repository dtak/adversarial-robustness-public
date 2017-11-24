# Adversarial Robustness (and Interpretability) via Gradient Regularization

This repository contains Python code and iPython notebooks used to run the experiments in [Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients](TODO).

## Main Idea

If you add an imperceptibly small amount of carefully crafted noise to an image which a neural network classifies correctly, you can usually cause it to make an incorrect prediction. This type of noise addition is called "adversarial perturbation," and the perturbed images are called adversarial examples. Unfortunately, it turns out that it's [pretty](https://arxiv.org/pdf/1412.6572) [easy](https://arxiv.org/pdf/1602.02697) to generate adversarial examples which (1) [fool almost any model](https://arxiv.org/pdf/1605.07277) trained on the same dataset, and (2) [continue to fool models](https://arxiv.org/pdf/1707.07397) even when printed out or viewed at different perspectives and scales. As neural networks start being used for things like face recognition and self-driving cars, this vulnerability poses an increasingly pressing problem.

In this repository, we try to tackle this problem directly, by training neural networks with a type of regularization that penalizes how sensitive their predictions are to infinitesimal changes in their inputs.
This type of regularization moves examples further away from the decision boundary in input-space, and has the side-effect of making [gradient-based explanations](http://www.jmlr.org/papers/volume11/baehrens10a/baehrens10a.pdf) of the model -- as well as the adversarial perturbations themselves -- more human-interpretable. Check out the experiments below or the [paper](TODO) for more details!

## Repository Structure

- `notebooks/` contains iPython notebooks replicating the main experiments from the paper:
    - [MNIST](./notebooks/MNIST.ipynb) compares robustness to two adversarial attack methods (the [FGSM](https://arxiv.org/pdf/1412.6572) and [TGSM](https://arxiv.org/pdf/1607.02533.pdf)) when CNNs are trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) with with various forms of regularization: defensive distillation, adversarial training, and two forms of input gradient regularization. This is a good one to look at first, since it's got both the results and some textual explanation of what's going on.
    - [notMNIST](./notebooks/notMNIST.ipynb) does the same accuracy comparisons, but for the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html). We omit the textual explanations since it would be redundant with what's in the MNIST notebook.
    - [SVHN](./notebooks/SVHN.ipynb) does the same for the [Street View House Numbers dataset](http://ufldl.stanford.edu/housenumbers/).
- `scripts/` contains code used to train models and generate / animate adversarial examples.
- `cached/` contains data files with trained model parameters and adversarial examples. The actual data is gitignored, but you can download it (see instructions below).
- `adversarial_robustness/` contains code modeling Python code for representing neural networks, datasets, and training / explanation / visualization / adversarial perturbation. Some of the code is strongly influenced by [cleverhans](https://github.com/tensorflow/cleverhans) and [tensorflow-adversarial](https://github.com/gongzhitaao/tensorflow-adversarial), but we've modified everything to be more object-oriented.

## Replication

To immediately run the notebooks using models and adversarial examples used to generate figures in the paper, you can download [this zipped directory](https://s3.amazonaws.com/adversarial-robustness-cached-models/cached.zip), which should replace the `cached/` subdirectory of this folder.

To fully replicate all experiments, you can use the files in the [scripts](./scripts) directory to retrain models and regenerate adversarial examples.

This code was tested with Python 3.5 and Tensorflow >= 1.2.1. Most files should also work with Python 2.7, but training may not work with earlier versions of Tensorflow, which lack second-derivative support for many CNN operations.

## Citation

You can cite
```
TODO
```
