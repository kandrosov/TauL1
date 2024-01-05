import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Input, Dense, Conv2D, Flatten, Concatenate, Multiply
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.quantizers import quantized_bits, quantized_relu
from tensorflow.keras.regularizers import l1
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity, strip_pruning, UpdatePruningStep

from ..CommonDef import *

pt_max = 255.


def make_model(cfg):
  input_calo = Input(shape=(6, 9, 2), name='calo_grid')
  input_tau = Input(shape=(2,), name='tau_pt_eta')
  input_center = Input(shape=(2, 3, 2), name='center_grid')

  qbits = cfg['setup']['qbits']
  l1reg = cfg['setup'].get('l1reg', 0)
  l1reg = l1(l1reg) if l1reg > 0 else None
  init = cfg['setup']['init']
  conv_cnt = 0
  center_cnt = 0
  dense_cnt = 0
  after_concat = False
  x_calo = input_calo
  x = None
  has_pruning = False
  for layer_cfg in cfg['layers']:
    if layer_cfg['type'] == 'conv':
      conv_cnt += 1
      name_conv = f'conv{conv_cnt}'
      x_calo = QConv2D(layer_cfg['filters'], layer_cfg['kernel_size'], name=name_conv,
                       strides=layer_cfg['strides'],
                       kernel_quantizer=quantized_bits(qbits, 0, alpha=1),
                       bias_quantizer=quantized_bits(qbits, 0, alpha=1),
                       kernel_initializer=init,
                       bias_initializer=init,
                       kernel_regularizer=l1reg,
                       bias_regularizer=l1reg
                      )(x_calo)
      x_calo = QActivation(activation=quantized_relu(qbits), name=name_conv+'_relu')(x_calo)
    elif layer_cfg['type'] == 'concat':
      x_calo = Flatten(name='flatten')(x_calo)
      if center_cnt > 0:
        x = Concatenate(name='concat')([x_calo, x])
      else:
        x = Concatenate(name='concat')([x_calo, input_tau])
      after_concat = True
    elif layer_cfg['type'] == 'dense':
      if after_concat:
        dense_cnt += 1
        name_dense = f'dense{dense_cnt}'
      else:
        if x is None:
          x = Flatten(name='flatten_center')(input_center)
          x = Concatenate(name='concat_center')([x, input_tau])
        center_cnt += 1
        name_dense = f'center{center_cnt}'

      layer = QDense(layer_cfg['units'], name=name_dense,
                 kernel_quantizer=quantized_bits(qbits, 0, alpha=1),
                 bias_quantizer=quantized_bits(qbits, 0, alpha=1),
                 kernel_initializer=init,
                 bias_initializer=init,
                 kernel_regularizer=l1reg,
                 bias_regularizer=l1reg
                 )
      prune = layer_cfg.get('prune', 0.)
      if prune > 0.:
        has_pruning = True
        layer = prune_low_magnitude(layer, ConstantSparsity(prune, layer_cfg['prune_begin'], layer_cfg['prune_freq']))
      x = layer(x)
      if not layer_cfg.get('is_output', False):
        x = QActivation(activation=quantized_relu(qbits), name=name_dense+'_relu')(x)
    else:
      raise Exception("Unknown layer type: {}".format(layer_cfg['type']))
  if cfg['setup']['regress_pt']:
    output_id = Activation('sigmoid', name='sigmoid_id')(x[:, 0:1])
    output_pt = Activation('hard_sigmoid', name='sigmoid_pt')(x[:, 1:2])
    pt_max_tf = tf.reshape(tf.constant(pt_max, dtype=tf.float32), (1, 1))
    output_pt = Multiply(name='scale_pt')([output_pt, pt_max_tf])
    output = Concatenate(name='concat_out')([output_id, output_pt])
  else:
    output = Activation('sigmoid', name='sigmoid')(x)
  model = keras.Model(inputs=[input_calo, input_tau, input_center], outputs=output, name='TauL1Model')
  return model, has_pruning

@tf.function
def binary_entropy(target, output):
  epsilon = tf.constant(1e-7, dtype=tf.float32)
  x = tf.clip_by_value(output, epsilon, 1 - epsilon)
  return - target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)

@tf.function
def accuracy(target, output):
  return tf.cast(tf.equal(target, tf.round(output)), tf.float32)

def id_loss(y_true, y_pred):
  return binary_entropy(y_true[:, 0], y_pred[:, 0]) * y_true[:, 2]

def pt_loss(y_true, y_pred):
  def _logcosh(x):
    return x + tf.math.softplus(-2.0 * x) - tf.cast(tf.math.log(2.0), x.dtype)
  rel_delta = (y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1]
  loss = tf.where(y_true[:, 0] == 1, _logcosh(rel_delta), tf.zeros_like(rel_delta))
  return loss * y_true[:, 2]

def id_acc(y_true, y_pred):
  return accuracy(y_true[:, 0], y_pred[:, 0]) * y_true[:, 3]

def l1tau_loss(y_true, y_pred):
  k = 20.
  return id_loss(y_true, y_pred) + k * pt_loss(y_true, y_pred)

def compile_model(model, cfg):
  opt = keras.optimizers.AdamW(learning_rate=cfg['setup']['learning_rate'], weight_decay=cfg['setup']['weight_decay'])
  metrics = [id_loss, id_acc]
  if cfg['setup']['regress_pt']:
    loss=l1tau_loss
    metrics.extend([pt_loss, l1tau_loss])
  else:
    loss=id_loss
  model.compile(optimizer=opt, loss=loss, metrics=metrics)

def make_save_model(has_pruning, cfg):
  def _save_model(model, path):
    if has_pruning:
      model = strip_pruning(model)
      compile_model(model, cfg)
    model.save(path)
  return _save_model

def to_train(x, y, w_orig, meta):
  is_tau = meta[:, get_index('L1Tau_type')] == TauType.tau
  input_tau = tf.stack([
    meta[:, get_index('L1Tau_hwPt')],
    meta[:, get_index('L1Tau_towerIEta')]
  ], axis=1)
  pnet_score = meta[:, get_index('Jet_PNet_probtauh')]
  pnet_score = tf.where(pnet_score > 0, pnet_score, tf.zeros_like(pnet_score))
  w = w_orig
  w = tf.where(is_tau, w[:,0] * pnet_score, w[:, 0] * (1-pnet_score))

  gen_pt = meta[:, get_index('L1Tau_gen_pt')]
  gen_pt_norm = tf.where(gen_pt < pt_max, gen_pt, tf.ones_like(gen_pt) * pt_max)
  y = tf.stack([y[:, 0], gen_pt_norm, w, w_orig[:, 0]], axis=1)
  input_calo = x
  input_center = input_calo[:, 2:4, 3:6, :]
  return (input_calo, input_tau, input_center), y

