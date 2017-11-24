import os
import numpy as np

rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
savedir = '/n/regal/doshi-velez_lab/aross/adversarial_robustness_models'
os.system('mkdir -p {}/jobs'.format(savedir))

template = """#!/bin/bash
#SBATCH -t 0-06:00
#SBATCH --mem=6000
#SBATCH --constraint=cuda-7.5
#SBATCH --gres=gpu:1
#SBATCH -p gpu_requeue
#SBATCH -o {}/jobs/%x.out
#SBATCH -e {}/jobs/%x.err

cd {}
module load gcc/4.8.2-fasrc01 cuda/7.5-fasrc02 tensorflow/1.3.0-fasrc01
python -u scripts/train_models.py {}
"""

def run_job(model, name, **kwargs):
  name = '{}-{}'.format(model, name)

  args = {
    'name': name,
    'dataset': model
    'batchsize': 256,
    'nbatches': 15000,
    'lr': 0.0002,
    'adameps': 1e-04,
    'savedir': savedir
  }
  args.update(kwargs)

  argz = ""
  for key, val in args.items():
    argz += " --{}={}".format(key,val)

  text = template.format(savedir, savedir, rootdir, argz)

  filename = '{}/jobs/{}.slurm'.format(savedir, name)
  with open(filename, 'w') as f:
    f.write(text)

  os.system('sbatch {}'.format(filename))

for model in ['mnist', 'notmnist', 'svhn']:
  run_job(model, 'normal')
  run_job(model, 'distilled', distilltemp=50.0)
  run_job(model, 'advtrain', advtraineps=0.3)
  for l2 in np.logspace(2, 5, 13):
    run_job(model, 'doubleback-{}'.format(l2), l2dbl=l2)
    run_job(model, 'double-advtrain-{}'.format(l2), l2dbl=l2, advtraineps=0.3)
  for l2 in np.logspace(0, 3, 13):
    run_job(model, 'insensitive-{}'.format(l2), l2dbl=l2)
