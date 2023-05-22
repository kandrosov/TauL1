import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import hls4ml
import plotting

os.environ['PATH'] = '/tools/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
#os.environ['PATH'] = '/usr/local/Xilinx/Vitis_HLS/2022.2/bin:' + os.environ['PATH']

model_name = 'model_v11'
model = tf.keras.models.load_model(f'/home/kandroso/workspace/TauL1/Training/models/{model_name}')

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
config['LayerName']['sigmoid']['exp_table_t'] = 'ap_fixed<18,8>'
config['LayerName']['sigmoid']['inv_table_t'] = 'ap_fixed<18,4>'

plotting.print_dict(config)

hls_model = hls4ml.converters.convert_from_keras_model(
  model, hls_config=config,
  output_dir=f'/home/kandroso/workspace/TauL1/Training/models/{model_name}_hls',
  part='xc7vx690t-ffg1927-2',
  backend='Vivado'
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

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

hls_model.build(csim=False)
hls4ml.report.read_vivado_report(f'/home/kandroso/workspace/TauL1/Training/models/{model_name}_hls')