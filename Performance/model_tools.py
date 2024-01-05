import json
import numpy as np
import os
import sys

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'
  os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

def save_model_summary(model, summary_path, verbose=0):
  lines = []
  def print_fn(line):
    lines.append(line)
    if verbose > 0:
      print(line)
  model.summary(print_fn=print_fn)
  with open(summary_path, 'w') as f:
    for line in lines:
      f.write(line + '\n')

def save_model_stat(model, stat_path, verbose=0):
  n_nonzero = 0
  weights = []
  layers = {}
  other_layers = []
  max_name_len = 0
  comp_names = [ 'kernel', 'bias' ]
  for layer in model.layers:
      name = layer.name
      available_components = [ comp for comp in comp_names if hasattr(layer, comp) ]
      if len(available_components) > 0:
        layers[name] = {}
        for comp in available_components:
          layers[name][comp] = {}
          x = getattr(layer, comp).numpy().flatten()
          x_nz = np.count_nonzero(x)
          n_nonzero += x_nz
          layers[name][comp]['n_weights'] = len(x)
          layers[name][comp]['n_weights_nonzero'] = x_nz
          max_name_len = max(max_name_len, len(name) + len(comp) + 3)
          weights.append(x)
      else:
        other_layers.append(name)
  w = np.concatenate(weights)
  unique_weights = np.unique(w)
  w_min = np.min(np.abs(w))
  w_max = np.max(np.abs(w))
  if verbose > 0:
    for layer_name, components in layers.items():
      for comp_name, comp in components.items():
        n_weights = comp['n_weights']
        n_wegihts_nonzero = comp['n_weights_nonzero']
        name_str = f'{layer_name}.{comp_name}:'
        print(f'{name_str:<{max_name_len}}n_weights={n_weights}, n_nonzero_weights={n_wegihts_nonzero}')
    print(f'total nonzero weights: {n_nonzero}')
    print(f'number of unique weights: {len(unique_weights)}')
  data = {
    'layers': layers,
    'other_layers': other_layers,
    'n_nonzero_weights': n_nonzero,
    'n_weights': len(w),
    'n_unique_weights': len(unique_weights),
    'min_weight': float(w_min),
    'max_weight': float(w_max),
  }
  with open(stat_path, 'w') as f:
    json.dump(data, f, indent=2)

def load_model(model_path):
  def fn(y_true, y_pred):
    pass
  custom_objects = [ 'l1tau_loss', 'id_loss', 'pt_loss', 'id_acc' ]
  custom_objects = { name: fn for name in custom_objects }
  return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--stat', required=False, type=str, default=None)
  parser.add_argument('--summary', required=False, type=str, default=None)
  args = parser.parse_args()

  model = load_model(args.model)
  if args.summary is not None:
    save_model_summary(model, args.summary, verbose=1)
  if args.stat is not None:
    save_model_stat(model, args.stat, verbose=1)