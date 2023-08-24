import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os
import sys

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'


from .RunKit.sh_tools import sh_call

def get_run_list(input):
  df = ROOT.RDataFrame('Events', input + '/nano_*.root')
  runs = df.AsNumpy(['run'])['run']
  unique_runs = np.stack(np.unique(runs, return_counts=True), axis=-1)
  return np.flip(unique_runs[unique_runs[:, 1].argsort()], axis=0)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, type=str)
  parser.add_argument('--output', required=False, default=None, type=str)
  parser.add_argument('--runs', required=False, default=None, type=str)
  args = parser.parse_args()

  if args.runs is None:
    print('Loading runs...', file=sys.stderr)
    run_list = get_run_list(args.input)
    print(f'{run_list.shape[0]} runs are avaliable', file=sys.stderr)
    print('run,n_events')
    for n in range(run_list.shape[0]):
      print(f'{run_list[n, 0]},{run_list[n, 1]}')
  else:
    if args.output is None:
      raise RuntimeError('Please specify --output')
    os.makedirs(args.output, exist_ok=True)
    runs = args.runs.split(',')
    for run in runs:
      sh_call([ 'python', 'RunKit/skim_tree.py' , '--input', args.input,
                '--output', os.path.join(args.output, f'run_{run}.root'),
                '--input-tree', 'Events', '--sel', f'run == {run}', '--verbose', '1' ],
              verbose=1)
