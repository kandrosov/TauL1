import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import hls4ml
import plotting
import shutil

os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
#os.environ['PATH'] = '/usr/local/Xilinx/Vitis_HLS/2022.2/bin:' + os.environ['PATH']

def convert_model(model_path, fpga_part, compile, build, print_report):
  model_path_hls = model_path + '_hls'

  if compile:
    if os.path.exists(model_path_hls):
      shutil.rmtree(model_path_hls)
    if os.path.exists(model_path_hls + '.tar.gz'):
      os.remove(model_path_hls + '.tar.gz')

    os.makedirs(model_path_hls)

    model = tf.keras.models.load_model(model_path)

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['LayerName']['sigmoid']['exp_table_t'] = 'ap_fixed<18,8>'
    config['LayerName']['sigmoid']['inv_table_t'] = 'ap_fixed<18,4>'

    plotting.print_dict(config)

    hls_model = hls4ml.converters.convert_from_keras_model(
      model, hls_config=config, output_dir=model_path_hls,
      part=fpga_part, backend='Vivado'
    )

    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=model_path_hls + '/model.pdf')

    hls_model.compile()

    n_nonzero = 0
    weights = []
    for layer in model.layers:
        name = layer.name
        if (name.startswith('conv') or name.startswith('dense')) and '_' not in name:
            print(layer.name)
            x = layer.kernel.numpy().flatten()
            x_nz = np.count_nonzero(x)
            n_nonzero += x_nz
            weights.append(x)
            print(x_nz)
            x = layer.bias.numpy().flatten()
            x_nz = np.count_nonzero(x)
            n_nonzero += x_nz
            weights.append(x)
            print(x_nz)
            #print(layer.kernel.numpy().)
            #print(layer.bias.numpy())
    w = np.concatenate(weights)
    print(f'total nonzero {n_nonzero}')
    print(len(np.unique(w)))

  if build:
    hls_model.build(csim=False)

  if print_report:
    hls4ml.report.read_vivado_report(model_path_hls)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--part', required=False, type=str, default='xc7vx690t-ffg1927-2')
  parser.add_argument('--compile', required=False, action='store_true')
  parser.add_argument('--build', required=False, action='store_true')
  parser.add_argument('--print-report', required=False, action='store_true')
  args = parser.parse_args()

  convert_model(args.model, args.part, args.compile, args.build, args.print_report)
