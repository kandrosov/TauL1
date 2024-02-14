import os
import sys
import yaml
import datetime
import shutil

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.dirname(file_dir)
  base_base_dir = os.path.dirname(base_dir)
  if base_base_dir not in sys.path:
    sys.path.append(base_base_dir)
  __package__ = os.path.split(base_dir)[-1]

from .Training.model import make_model, make_input_fn, make_save_model, compile_model
from .Training.callbacks import ModelCheckpoint

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', required=False, default='Training/model.yaml', type=str)
  parser.add_argument('--output', required=False, default='data', type=str)
  parser.add_argument('--output-name', required=False, default=None, type=str)
  parser.add_argument('--gpu', required=False, default='1', type=str)
  parser.add_argument('--batch-size', required=False, type=int, default=2000)
  parser.add_argument('--patience', required=False, type=int, default=16)
  parser.add_argument('--n-epochs', required=False, type=int, default=10000)
  parser.add_argument('--dataset-train', required=False, default='/data_ssd/Run3_HLT/prod_v3_skim_v2-train', type=str)
  parser.add_argument('--dataset-val', required=False, default='/data_ssd/Run3_HLT/prod_v3_skim_v2-val', type=str)
  parser.add_argument('--summary-only', required=False, action='store_true')
  args = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  import tensorflow as tf
  from tensorflow_model_optimization.sparsity.keras import UpdatePruningStep

  with open(args.cfg) as f:
    cfg = yaml.safe_load(f)

  model, has_pruning = make_model(cfg)
  model.summary()
  if args.summary_only:
    sys.exit(0)

  to_train = make_input_fn(cfg['setup']['reduce_calo_precision'], cfg['setup']['reduce_center_precision'],
                           cfg['setup']['apply_avg_pool'], cfg['setup']['concat_input'], to_train=True)
  dataset_train = tf.data.Dataset.load(args.dataset_train, compression=None)
  ds_train = dataset_train.batch(args.batch_size).map(to_train).prefetch(tf.data.AUTOTUNE)

  dataset_val = tf.data.Dataset.load(args.dataset_val, compression=None)
  ds_val = dataset_val.batch(args.batch_size).map(to_train).prefetch(tf.data.AUTOTUNE)

  output_root = 'data'
  if args.output_name is None:
    output_name = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
  else:
    output_name = args.output_name
  dirFile = os.path.join(output_root, output_name)
  if os.path.exists(dirFile):
    raise RuntimeError(f'Output directory {dirFile} already exists')
  os.makedirs(dirFile)

  cfg_out = os.path.join(dirFile, 'model.yaml')
  shutil.copy(args.cfg, cfg_out)
  shutil.copy('Training/model.py', dirFile)

  dirFile = os.path.join(dirFile, 'model')
  print(dirFile)

  compile_model(model, cfg)
  callbacks = [
    ModelCheckpoint(dirFile, verbose=1, mode='min', min_rel_delta=1e-3, patience=args.patience,
                    save_callback=make_save_model(has_pruning, cfg)),
    tf.keras.callbacks.CSVLogger(os.path.join(dirFile, 'training_log.csv'), append=True),
  ]

  if has_pruning:
    callbacks.append(UpdatePruningStep())

  model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, epochs=args.n_epochs, verbose=1)


