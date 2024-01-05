import copy
import gc
import json
import numpy as np
import os
import sys
import uproot
import yaml
import awkward as ak

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .RunKit.run_tools import ps_call
from .Performance.eval_tools import list_to_str, get_shortest_interval
from .Performance.variable import VariableCollection
from .Performance.plot import PlotCollection, PlotScoresDiff
from .Performance.algorithm import Algorithm

class Dataset:
  def __init__(self, name, cfg, datasets_dir, scores_dir, qscores_dir):
    self.name = name
    self.title = cfg['title']
    self.file = cfg['file']
    self.input_path = os.path.join(datasets_dir, self.file)
    self.scores_path = os.path.join(scores_dir, self.file)
    if qscores_dir is not None:
      self.qscores_path = os.path.join(qscores_dir, self.file)
    else:
      self.qscores_path = None
    self.input = None
    self.scores = None
    self.qscores = None
    self.columns = set()
    self.scores_columns = set([ 'L1Tau_NNtag', 'L1Tau_ptReg' ])
    self.qscores_columns = set([ 'L1Tau_NNtag_q', 'L1Tau_ptReg_q' ])
    self.columns.update(self.scores_columns)
    if self.qscores_path is not None:
      self.columns.update(self.qscores_columns)

  def scores_ready(self):
    return os.path.exists(self.scores_path)

  def qscores_ready(self):
    return self.qscores_path is not None and os.path.exists(self.qscores_path)

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
      columns = list(self.columns - self.scores_columns - self.qscores_columns)
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

  def get_qscores(self):
    if self.qscores is None:
      if not self.qscores_ready():
        raise RuntimeError(f'Dataset {self.name} has no quantized scores')
      columns = list(self.qscores_columns.intersection(self.columns))
      print(f'dataset {self.name}: loading quantized scores columns {list_to_str(columns)}')
      self.qscores = self._load(self.qscores_path, columns, 'qscores')
    return self.qscores

  def __getitem__(self, key):
    if key not in self.columns:
      raise KeyError(f'Column {key} not found in dataset {self.name}')
    if key in self.scores_columns:
      return self.get_scores()[key]
    if key in self.qscores_columns:
      return self.get_qscores()[key]
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

    self.scores_diff_dir = os.path.join(base_dir, self.cfg['scores_diff_dir'])

    self.eval_hls4ml = self.cfg.get('eval_hls4ml', False)
    if self.eval_hls4ml:
      self.hls4ml_dir = os.path.join(base_dir, self.cfg['hls4ml']['output_dir'])
      self.hls4ml_config_customizations = self.cfg['hls4ml'].get('config_customizations')
      if self.hls4ml_config_customizations is not None:
        self.hls4ml_config_customizations = json.dumps(self.hls4ml_config_customizations)
      self.hls4ml_config_path = os.path.join(self.hls4ml_dir, 'config.yaml')
      self.hls4ml_vivado_report_path = os.path.join(self.hls4ml_dir, 'vivado_report.txt')
      self.hls4ml_fpga_part = self.cfg['hls4ml']['fpga_part']
      self.hls4ml_model_path = os.path.join(self.hls4ml_dir, 'model')
      self.hls4ml_scores_dir = os.path.join(self.hls4ml_dir, self.cfg['scores_dir'])
    else:
      self.hls4ml_scores_dir = None

    self.datasets = {}
    sources_dir = os.path.join(base_dir, self.cfg['scores_dir'])
    os.makedirs(sources_dir, exist_ok=True)
    if self.hls4ml_scores_dir is not None:
      os.makedirs(self.hls4ml_scores_dir, exist_ok=True)
    for ds_name, ds_entry in self.cfg['datasets'].items():
      self.datasets[ds_name] = Dataset(ds_name, ds_entry, self.cfg['datasets_dir'], sources_dir, self.hls4ml_scores_dir)
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

    self.eval_resolution = 'resolution' in self.cfg
    if self.eval_resolution:
      self.resolution_dir = os.path.join(base_dir, self.cfg['resolution']['output_dir'])
      resolution_ds_name = self.cfg['resolution']['ds_name']
      if resolution_ds_name not in self.datasets:
        raise RuntimeError(f'Resolution dataset {resolution_ds_name} not found in datasets')
      self.resolution_ds = self.datasets[resolution_ds_name]


    self.model_summary_file = os.path.join(base_dir, self.cfg['model_summary_file'])
    self.model_stat_file = os.path.join(base_dir, self.cfg['model_stat_file'])

    file_dir = os.path.dirname(os.path.abspath(__file__))
    for file_entry in [ 'apply_training', 'eval_resolution', 'model_tools',
                        ('hls4ml', 'create_config'), ('hls4ml', 'convert_model') ]:
      if isinstance(file_entry, str):
        file = file_entry
        f_dir = file_dir
        attr_name = f'{file}_py'
      else:
        f_dir, file = file_entry
        attr_name = f'{f_dir}_{file}_py'
        f_dir = os.path.join(file_dir, '..', f_dir)
      file_path = os.path.join(f_dir, f'{file}.py')
      file_path = os.path.abspath(file_path)
      if not os.path.exists(file_path):
        raise RuntimeError(f'{file}.py does not exist in {file_dir}')
      setattr(self, attr_name, file_path)


def run_perf(cfg_path, base_dir):
  setup = Setup(cfg_path, base_dir)

  has_stat_file = os.path.exists(setup.model_stat_file)
  has_summary_file = os.path.exists(setup.model_summary_file)
  if not has_stat_file or not has_summary_file:
    cmd = [ 'python', setup.model_tools_py, '--model', setup.model_dir ]
    if not has_stat_file:
      cmd.extend(['--stat', setup.model_stat_file])
    if not has_summary_file:
      cmd.extend(['--summary', setup.model_summary_file])
    ps_call(cmd, verbose=1)

  if setup.eval_hls4ml:
    if not os.path.exists(setup.hls4ml_config_path):
      cmd = [ 'python', setup.hls4ml_create_config_py, '--model', setup.model_dir,
              '--output', setup.hls4ml_config_path ]
      if setup.hls4ml_config_customizations is not None:
        cmd.extend(['--customizations', setup.hls4ml_config_customizations])
      ps_call(cmd, verbose=1)
    if not os.path.exists(setup.hls4ml_vivado_report_path):
      cmd = [ 'python', setup.hls4ml_convert_model_py, '--model', setup.model_dir,
              '--config', setup.hls4ml_config_path, '--output', setup.hls4ml_dir,
              '--part', setup.hls4ml_fpga_part ]
      ps_call(cmd, verbose=1)

  for dataset in setup.datasets.values():
    cmd_base = [ 'python', setup.apply_training_py, '--dataset', dataset.input_path, '--model', setup.model_dir,
                 '--output', dataset.scores_path, '--batch-size', str(setup.cfg['apply_training']['batch_size']) ]
    if setup.cfg['apply_training']['regress_pt']:
      cmd_base.append('--has-pt-node')
    if not dataset.scores_ready():
      cmd = copy.deepcopy(cmd_base)
      cmd.extend(['--output', dataset.scores_path])
      ps_call(cmd, verbose=1)

    if setup.eval_hls4ml:
      if not dataset.qscores_ready():
        cmd = copy.deepcopy(cmd_base)
        cmd.extend([ '--output', dataset.qscores_path, '--use-hls4ml', '--hls4ml-model', setup.hls4ml_model_path,
                    '--hls4ml-config', setup.hls4ml_config_path, '--fpga-part', setup.hls4ml_fpga_part ])
        ps_call(cmd, verbose=1)

      scores_plot = PlotScoresDiff(f'scores_diff_{dataset.name}',
                                  os.path.join(setup.scores_diff_dir, f'{dataset.name}.pdf'))
      if not scores_plot.is_ready():
        scores_plot.delta_scores = ak.flatten(dataset['L1Tau_NNtag'] - dataset['L1Tau_NNtag_q'])
        scores_plot.cl = 0.95
        scores_plot.q_limits, scores_plot.q_values = get_shortest_interval(scores_plot.delta_scores,
                                                                           alpha=scores_plot.cl)
        scores_plot.eval()

  if setup.eval_resolution and not os.path.exists(setup.resolution_dir):
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
