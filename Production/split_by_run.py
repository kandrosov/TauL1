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
  df = ROOT.RDataFrame('Events', input + '/*.root')
  runs = df.AsNumpy(['run'])['run']
  return np.sort(np.unique(runs))

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--input', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  args = parser.parse_args()

  print('Loading runs...')
  run_list = get_run_list(args.input)
  run_strs = [str(run) for run in run_list]
  print(f'{len(run_strs)} runs are avaliable: ' + ', '.join(run_strs))
  os.makedirs(args.output, exist_ok=True)
  for run in run_list:
    sh_call([ 'python', 'RunKit/skim_tree.py' , '--input', args.input,
              '--output', os.path.join(args.output, f'run_{run}.root'),
              '--input-tree', 'Events', '--sel', f'run == {run}', '--verbose', '1' ],
            verbose=1)
