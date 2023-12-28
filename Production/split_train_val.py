import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import shutil

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, type=str)
  parser.add_argument('--train-stop', required=True, type=int)
  parser.add_argument('--output-train', required=False, default=None, type=str)
  parser.add_argument('--output-val', required=False, default=None, type=str)
  parser.add_argument('--compression-in', required=False, default=None, type=str)
  parser.add_argument('--compression-out', required=False, default=None, type=str)
  args = parser.parse_args()

  output_train = args.input + '-train' if args.output_train is None else args.output_train
  if os.path.exists(output_train):
    shutil.rmtree(output_train)

  output_val = args.input + '-val' if args.output_val is None else args.output_val
  if os.path.exists(output_val):
    shutil.rmtree(output_val)

  dataset = tf.data.Dataset.load(args.input, compression=args.compression_in)
  print('Saving training dataset...')
  dataset.take(args.train_stop).save(output_train, compression=args.compression_out)
  print('Saving validation dataset...')
  dataset.skip(args.train_stop).save(output_val, compression=args.compression_out)
  print('Done')
