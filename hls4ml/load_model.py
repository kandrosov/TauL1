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

def load_hls4ml_model(model_path, config_path, fpga_part=None, output_path=None, compile=False):
  model = load_model(model_path)
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_path,
                                                     part=fpga_part, backend='Vivado')
  if compile:
    model.compile()
  return model

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--config', required=True, type=str)
  args = parser.parse_args()

  load_hls4ml_model(args.model, args.config)
