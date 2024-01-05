import contextlib
import hls4ml
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
#os.environ['PATH'] = '/usr/local/Xilinx/Vitis_HLS/2022.2/bin:' + os.environ['PATH']

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from TauL1.hls4ml.load_model import load_hls4ml_model

def convert_model(keras_model_path, hls_path, config_path, fpga_part):
  model_path = os.path.join(hls_path, 'model')
  if os.path.exists(model_path):
    shutil.rmtree(model_path)
  if os.path.exists(model_path + '.tar.gz'):
    os.remove(model_path + '.tar.gz')
  os.makedirs(model_path)
  model_pdf_path = os.path.join(hls_path, 'model.pdf')
  vivado_report_path = os.path.join(hls_path, 'vivado_report.txt')
  vivado_report_tmp_path = vivado_report_path + '.tmp'

  hls_model = load_hls4ml_model(keras_model_path, config_path, fpga_part=fpga_part, output_path=model_path)
  hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=model_pdf_path)
  hls_model.compile()
  hls_model.build(csim=False)
  with open(vivado_report_tmp_path, 'w') as f:
    with contextlib.redirect_stdout(f):
      hls4ml.report.read_vivado_report(model_path)
  shutil.move(vivado_report_tmp_path, vivado_report_path)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--config', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--part', required=False, type=str, default='xc7vx690t-ffg1927-2')
  args = parser.parse_args()

  convert_model(args.model, args.output, args.config, args.part)
