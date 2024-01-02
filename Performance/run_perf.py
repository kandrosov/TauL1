import gc
import numpy as np
import os
import sys
import uproot
import yaml

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .RunKit.run_tools import ps_call
from .Performance.eval_tools import list_to_str
from .Performance.variable import VariableCollection
from .Performance.plot import PlotCollection
from .Performance.algorithm import Algorithm

class Dataset:
  def __init__(self, name, cfg, datasets_dir, scores_dir):
    self.name = name
    self.title = cfg['title']
    self.file = cfg['file']
    self.input_path = os.path.join(datasets_dir, self.file)
    self.scores_path = os.path.join(scores_dir, self.file)
    self.input = None
    self.scores = None
    self.columns = set()
    self.scores_columns = set([ 'L1Tau_NNtag', 'L1Tau_ptReg' ])

  def scores_ready(self):
    return os.path.exists(self.scores_path)

  def _load(self, input_file, columns, columns_name):
    if len(columns) == 0:
      raise RuntimeError(f'Dataset {self.name} has no {columns_name} columns')
    with uproot.open(input_file) as f:
      return f['Events'].arrays(columns)

  def get_size(self):
    df = self.get_input()
    column = list(self.columns - self.scores_columns)[0]
    return len(df[column])

  def get_input(self):
    if self.input is None:
      columns = list(self.columns - self.scores_columns)
      print(f'dataset {self.name}: loading input columns {list_to_str(columns)}')
      self.input = self._load(self.input_path, columns, 'input')
    return self.input

  def get_scores(self):
    if self.scores is None:
      if not self.scores_ready():
        raise RuntimeError(f'Dataset {self.name} has no scores')
      columns = list(self.scores_columns.intersection(self.columns))
      print(f'dataset {self.name}: loading scores columns {list_to_str(columns)}')
      self.scores = self._load(self.scores_path, columns, 'scores')
    return self.scores

  def __getitem__(self, key):
    if key not in self.columns:
      raise KeyError(f'Column {key} not found in dataset {self.name}')
    if key in self.scores_columns:
      return self.get_scores()[key]
    return self.get_input()[key]

class Setup:
  def __init__(self, cfg_path, base_dir):
    with open(cfg_path) as f:
      self.cfg = yaml.safe_load(f)

    self.base_dir = base_dir
    if not os.path.exists(base_dir):
      raise RuntimeError(f'Base directory {base_dir} does not exist')
    self.model_dir = os.path.join(base_dir, self.cfg['model_dir'])
    if not os.path.exists(self.model_dir):
      raise RuntimeError(f'Model directory {self.model_dir} does not exist')

    self.datasets = {}
    sources_dir = os.path.join(base_dir, self.cfg['scores_dir'])
    os.makedirs(sources_dir, exist_ok=True)
    for ds_name, ds_entry in self.cfg['datasets'].items():
      self.datasets[ds_name] = Dataset(ds_name, ds_entry, self.cfg['datasets_dir'], sources_dir)
    lut_bins = {}
    for name, entry in self.cfg['lut_bins'].items():
      lut_bins[name] = np.array(entry, dtype=np.float32)
    self.algos = {}
    algo_params_dir = os.path.join(base_dir, self.cfg['algo_params_dir'])
    os.makedirs(algo_params_dir, exist_ok=True)
    for algo_name, algo_cfg in self.cfg['algorithms'].items():
      self.algos[algo_name] = Algorithm(algo_name, algo_cfg, algo_params_dir, self.datasets, lut_bins)
    for algo in self.algos.values():
      algo.initialize_composite(self.algos)

    variables_dir = os.path.join(base_dir, self.cfg['variables_dir'])
    self.variables = VariableCollection(self.cfg['variables'], variables_dir, self.algos, self.datasets)
    for var_name, var in self.variables.items():
      var.dataset.columns.update(var.ds_columns)

    algo_columns = set()
    for algo in self.algos.values():
      algo.initialize_simple(self.algos, self.variables)
      algo_columns.update(algo.columns)
    for dataset in self.datasets.values():
      dataset.columns.update(algo_columns)

    plots_dir = os.path.join(base_dir, self.cfg['performance_dir'])
    self.plots = PlotCollection(plots_dir, self.cfg['plots'], self.variables)

    self.resolution_dir = os.path.join(base_dir, self.cfg['resolution']['output_dir'])
    resolution_ds_name = self.cfg['resolution']['ds_name']
    if resolution_ds_name not in self.datasets:
      raise RuntimeError(f'Resolution dataset {resolution_ds_name} not found in datasets')
    self.resolution_ds = self.datasets[resolution_ds_name]

    file_dir = os.path.dirname(os.path.abspath(__file__))
    for file in ['apply_training', 'eval_resolution']:
      file_path = os.path.join(file_dir, f'{file}.py')
      if not os.path.exists(file_path):
        raise RuntimeError(f'{file}.py does not exist in {file_dir}')
      setattr(self, f'{file}_py', file_path)


def run_perf(cfg_path, base_dir):
  setup = Setup(cfg_path, base_dir)

  for dataset in setup.datasets.values():
    if not dataset.scores_ready():
      cmd = [ 'python', setup.apply_training_py, '--dataset', dataset.input_path, '--model', setup.model_dir,
             '--output', dataset.scores_path, '--batch-size', str(setup.cfg['apply_training']['batch_size']) ]
      ps_call(cmd, verbose=1)

  if not os.path.exists(setup.resolution_dir):
    cmd = [ 'python', setup.eval_resolution_py, '--cfg', cfg_path, '--dataset', setup.resolution_ds.input_path,
           '--scores', setup.resolution_ds.scores_path, '--output', setup.resolution_dir ]
    ps_call(cmd, verbose=1)

  n_vars_not_ready = -1
  n_plots_not_ready = -1
  while n_vars_not_ready != 0:
    n_vars_ready, n_vars_evaluated, n_vars_not_ready = setup.variables.eval()
    print(f'Variables: {n_vars_ready} ready, {n_vars_evaluated} evaluated in this iteration,'
          f' {n_vars_not_ready} not ready')
    n_plots_ready, n_plots_evaluated, n_plots_not_ready = setup.plots.eval()
    print(f'Plots: {n_plots_ready} ready, {n_plots_evaluated} evaluated in this iteration,'
          f' {n_plots_not_ready} not ready')
    for algo_name, algo in setup.algos.items():
      if not algo.is_ready() and algo.is_ready_for_opt():
        print(f'Optimizing thresholds for {algo_name} ...')
        algo.optimize_thresholds()
        break
    gc.collect()

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', required=True, type=str)
  parser.add_argument('--model', required=True, type=str)
  args = parser.parse_args()

  run_perf(args.cfg, args.model)
