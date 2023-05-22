import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Input, Dense, Conv2D, Flatten, Concatenate
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.quantizers import quantized_bits, quantized_relu, quantized_tanh
from tensorflow.keras.regularizers import l1
import os
import math
import sys
import yaml
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity, strip_pruning, UpdatePruningStep

gpu_cfg = {
  'gpu_index': 1,
  'gpu_mem': 7 # GB
}

def setup_gpu(gpu_cfg):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_cfg['gpu_index']], 'GPU')
            # tf.config.experimental.set_virtual_device_configuration(gpus[gpu_cfg['gpu_index']],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_cfg['gpu_mem']*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

setup_gpu(gpu_cfg)

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import *

def make_model(cfg):
  input_calo = Input(shape=(6, 9, 2), name='calo_grid')
  input_tau = Input(shape=(2,), name='tau_pt_eta')

  qbits = cfg['setup']['qbits']
  l1reg = cfg['setup']['l1reg']
  init = cfg['setup']['init']
  conv_cnt = 0
  dense_cnt = 0
  relu_cnt = 0
  x = input_calo
  for layer_cfg in cfg['layers']:
    if layer_cfg['type'] == 'conv':
      conv_cnt += 1
      relu_cnt += 1
      x = QConv2D(layer_cfg['filters'], layer_cfg['kernel_size'], name=f'conv{conv_cnt}',
                  strides=layer_cfg['strides'],
                  kernel_quantizer=quantized_bits(qbits, 0, alpha=1),
                  bias_quantizer=quantized_bits(qbits, 0, alpha=1),
                  kernel_initializer=init,
                  bias_initializer=init,
                  kernel_regularizer=l1(l1reg),
                  bias_regularizer=l1(l1reg)
                  )(x)
      x = QActivation(activation=quantized_relu(qbits), name=f'conv_relu{relu_cnt}')(x)
    elif layer_cfg['type'] == 'concat':
      x = Flatten(name='flatten')(x)
      x = Concatenate(name='concat')([x, input_tau])
    elif layer_cfg['type'] == 'dense':
      dense_cnt += 1

      layer = QDense(layer_cfg['units'], name=f'dense{dense_cnt}',
                 kernel_quantizer=quantized_bits(qbits, 0, alpha=1),
                 bias_quantizer=quantized_bits(qbits, 0, alpha=1),
                 kernel_initializer=init,
                 bias_initializer=init,
                 kernel_regularizer=l1(l1reg),
                 bias_regularizer=l1(l1reg)
                 )
      prune = layer_cfg.get('prune', 0.)
      if prune > 0.:
        layer = prune_low_magnitude(layer, ConstantSparsity(prune, layer_cfg['prune_begin'], layer_cfg['prune_freq']))
      x = layer(x)
      if layer_cfg['units'] > 1:
        relu_cnt += 1
        x = QActivation(activation=quantized_relu(qbits), name=f'dense_relu{relu_cnt}')(x)
    else:
      raise Exception("Unknown layer type: {}".format(layer_cfg['type']))
  output = Activation('sigmoid', name='sigmoid')(x)
  return keras.Model(inputs=[input_calo, input_tau], outputs=output, name='TauL1Model')

with open('Training/model.yaml') as f:
  cfg = yaml.safe_load(f)

model = make_model(cfg)
model.summary()
#raise Exception("Stop here")

#from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
#from tensorflow_model_optimization.sparsity.keras import strip_pruning

#pruning_params = {"pruning_schedule": pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)}
#model = prune.prune_low_magnitude(model, **pruning_params)

model.compile(optimizer='adam', loss='binary_crossentropy', weighted_metrics=['accuracy'])



input_idx = 0
dataset = tf.data.Dataset.load(f'output/skim_v1_tf_v1/taus_{input_idx}', compression='GZIP')

def to_train(x, y, w, meta):
  a = 0.5
  b = 1
  #k = 100
  gen_pt0 = 100
  gen_pt = meta[:, get_index('L1Tau_gen_pt')]
  is_tau = meta[:, get_index('L1Tau_type')] == TauType.tau
  #is_jet = meta[:, get_index('L1Tau_type')] == TauType.jet
  #k = tf.math.log(meta[:, get_index("L1Tau_gen_pt")]/20.)
  #w = w *( a * (meta[:,get_index('L1Tau_gen_pt')] - gen_pt0 ) + b)
  w = tf.where(is_tau & (gen_pt > gen_pt0), w[:,0]*( a * (gen_pt - gen_pt0) + b), w[:, 0])
  #w = tf.where(is_jet & (gen_pt > gen_pt0), w/( a * (gen_pt - gen_pt0) + b), w)
  #w = tf.where(is_tau & (gen_pt > gen_pt0), w[:, 0] * tf.exp((gen_pt - gen_pt0)/k), w[:, 0])
  return (x[:,:,:,2:4], x[:, 0, 0, :2]), y, w

ds_train_val = dataset.batch(300).map(to_train)
n_batches = ds_train_val.cardinality().numpy()
n_batches_train = int(n_batches * 0.8)
ds_train = ds_train_val.take(n_batches_train)
ds_val = ds_train_val.skip(n_batches_train)

k = 0
dirFile = f'Training/models/model_v{k}'
while(os.path.isdir(dirFile)):
  k+=1
  dirFile = f'Training/models/model_v{k}'
print(dirFile)

#callbacks = [ UpdatePruningStep() ]
callbacks = [
  tf.keras.callbacks.ModelCheckpoint(filepath=dirFile, save_weights_only=False, verbose=1, save_best_only=True),
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
]
model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, epochs=1000, verbose=1)
#model = strip_pruning(model)
#model.compile(optimizer='adam', loss='binary_crossentropy', weighted_metrics=['accuracy'])
#model.save(dirFile)
