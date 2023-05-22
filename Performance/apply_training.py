import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import *

def apply_training(dataset_path, model_path, output_file, vars, batch_size):
  model = tf.keras.models.load_model(model_path)
  dataset = tf.data.Dataset.load(dataset_path, compression='GZIP')
  n_taus = dataset.cardinality().numpy()
  output_dir = os.path.dirname(output_file)
  if len(output_dir) > 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if os.path.exists(output_file):
    os.remove(output_file)
  with tqdm(total=n_taus) as pbar:
    for x, y, w, meta in dataset.batch(batch_size):
      values = {
        'nn_score': model((x[:,:,:,2:4], x[:, 0, 0, :2])).numpy().flatten(),
      }
      for var in vars:
        values[var] = meta[:, get_index(var)].numpy().flatten()
      pd_dataset = pd.DataFrame(values)
      pd_dataset.to_hdf(output_file, key='taus', append=True, complevel=1, complib='zlib')
      pbar.update(values['nn_score'].shape[0])


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--vars', required=False, type=str, default=','.join([
    'nPV', 'L1Tau_type', 'L1Tau_gen_pt', 'L1Tau_gen_eta', 'L1Tau_pt', 'L1Tau_eta', 'L1Tau_hwIso',
  ]))
  parser.add_argument('--batch-size', required=False, type=int, default=3000)
  args = parser.parse_args()

  apply_training(args.dataset, args.model, args.output, args.vars.split(','), args.batch_size)
