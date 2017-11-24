import os
import sys
import glob
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(root_dir)
from adversarial_robustness.plot_helpers import *

#models = ['doubleback','distilled','insensitive','advtrain','normal']
models = ['insensitive'] #normal']
datasets = ['svhn'] #mnist','notmnist','svhn']

cmd = "python scripts/generate_tgsm_grids.py --model-path={} --model-label='{}' --example-path={} --dataset={} --eps=0.1"

for m in models:
  for d in datasets:
    model_path = glob.glob(root_dir + '/cached/models/' + d + '-' + m + '-*cnn.pkl')[0]
    example_path = root_dir + '/cached/tgsm/{}/{}'.format(d, m)
    actual_cmd = cmd.format(model_path, model_labels[m], example_path, d)
    print(actual_cmd)
    os.system(actual_cmd)

