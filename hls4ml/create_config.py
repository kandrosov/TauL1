import copy
import json
import os
import sys
import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import hls4ml

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from TauL1.Performance.model_tools import load_model

def apply_customizations(config, customizations):
  def apply(cfg, updates, path=''):
    for key, value in updates.items():
      if isinstance(value, dict):
        apply(cfg[key], value, path=f'{path}/{key}')
      else:
        if key in cfg:
          print(f'Overriding {path}/{key}: {cfg[key]} -> {value}')
        else:
          print(f'Setting {path}/{key}: {value}')
        cfg[key] = value
  new_config = copy.deepcopy(config)
  apply(new_config, customizations)
  return new_config

def create_hls4ml_config(model_path, output, customizations=None, verbose=0):
  output_dir = os.path.dirname(output)
  os.makedirs(output_dir, exist_ok=True)
  model = load_model(model_path)
  config = hls4ml.utils.config_from_keras_model(model, granularity='name')
  if customizations is not None:
    print(f'Applying customizations...')
    config = apply_customizations(config, customizations)
  config_yaml = yaml.dump(config, default_flow_style=False)
  if verbose > 0:
    print(config_yaml)
  with open(output, 'w') as f:
    f.write(config_yaml)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--customizations', required=False, type=str)
  args = parser.parse_args()

  customizations = None
  if args.customizations is not None:
    customizations = json.loads(args.customizations)

  create_hls4ml_config(args.model, args.output, customizations=customizations, verbose=1)
