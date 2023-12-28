import copy
import gc
import json
import numpy as np
import os
import sys
import uproot
import yaml
import awkward as ak

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import TauType
from .RunKit.run_tools import ps_call
from .Performance.eval_tools import get_pass_mask_default_or, get_pass_mask_nn, mk_sel_fn, pass_masks_to_count, \
                                    passed_to_rate, find_nn_thrs, get_tau_pass_mask_default, get_tau_pass_mask_nn, \
                                    get_differential_efficiency

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
      self.input = self._load(self.input_path, columns, 'input')
    return self.input

  def get_scores(self):
    if self.scores is None:
      if not self.scores_ready():
        raise RuntimeError(f'Dataset {self.name} has no scores')
      columns = list(self.scores_columns.intersection(self.columns))
      self.scores = self._load(self.scores_path, columns, 'scores')
    return self.scores

  def __getitem__(self, key):
    if key not in self.columns:
      raise KeyError(f'Column {key} not found in dataset {self.name}')
    if key in self.scores_columns:
      return self.get_scores()[key]
    return self.get_input()[key]

class EventMetric:
  def __init__(self, name, algo, dataset, metric_type, path):
    self.name = name
    self.algo = algo
    self.dataset = dataset
    self.metric_type = metric_type
    self.path = path
    self.value = None
    if os.path.exists(self.path):
      with open(self.path, 'r') as f:
        self.value = float(f.read())

  def is_ready(self):
    return self.value is not None

  def is_ready_for_eval(self):
    return self.algo.is_ready()

  def get_value(self):
    if not self.is_ready():
      raise RuntimeError(f'Event metric {self.name} is not ready')
    return self.value

  def set_value(self, value):
    self.value = float(value)
    with open(self.path, 'w') as f:
      f.write(str(value))

  def eval(self):
    if self.is_ready():
      raise RuntimeError(f'Event metric {self.name} is already evaluated')
    if not self.is_ready_for_eval():
      raise RuntimeError(f'Event metric {self.name} is not ready for evaluation')
    n_total = self.dataset.get_size()
    n_passed = self.algo.get_passed(self.dataset)
    if self.metric_type == 'rate':
      value = passed_to_rate(n_passed, n_total)
    elif self.metric_type == 'efficiency':
      value = float(n_passed) / n_total
    else:
      raise RuntimeError(f'Unknown metric type {self.metric_type}')
    self.set_value(value)

class CollectionBase:
  def eval(self):
    n_ready = 0
    n_evaluated = 0
    n_not_ready = 0
    for item_name, item in self.items():
      if item.is_ready():
        n_ready += 1
      elif item.is_ready_for_eval():
        print(f'Evaluating {item_name} ...')
        item.eval()
        n_evaluated += 1
        n_ready += 1
      else:
        n_not_ready += 1
    return n_ready, n_evaluated, n_not_ready

class EventMetricCollection(CollectionBase):
  def __init__(self, cfg, event_metrics_dir, algos, datasets):
    self.metrics = {}
    for base_name, entry in cfg.items():
      self.metrics[base_name] = {}
      dataset = datasets[entry['dataset']]
      metric_type = entry['type']
      base_path = os.path.join(event_metrics_dir, base_name)
      os.makedirs(base_path, exist_ok=True)
      for algo_name, algo in algos.items():
        full_name = f'{base_name}/{algo_name}'
        metric_path = os.path.join(base_path, f'{algo_name}.txt')
        self.metrics[base_name][algo_name] = EventMetric(full_name, algo, dataset, metric_type, metric_path)

  def get(self, base_name, algo_name):
    return self.metrics[base_name][algo_name]

  def items(self):
    for metrics in self.metrics.values():
      for metric in metrics.values():
        yield metric.name, metric

class Variable:
  def __init__(self, name, cfg, path, algo, dataset):
    self.name = name
    self.path = path
    self.column = self.name if 'column' not in cfg else cfg['column']
    self.tau_var = cfg.get('tau_var', True)
    self.algo = algo
    self.dataset = dataset
    self.bins = cfg['bins']
    self.major_ticks = cfg.get('major_ticks', None)
    self.minor_ticks = cfg.get('minor_ticks', None)
    self.xlabel = cfg['xlabel']
    self.xscale = cfg.get('xscale', 'linear')
    self.xlim = (self.bins[0], self.bins[-1])
    self.ylim = cfg.get('ylim', None)
    self.x = None
    self.x_up = None
    self.x_down = None
    self.y = None
    self.y_up = None
    self.y_down = None
    if os.path.exists(self.path):
      with open(self.path, 'r') as f:
        data = json.load(f)
      for entry in data:
        if entry['bins'] == self.bins:
          for attr in ['x', 'x_up', 'x_down', 'y', 'y_up', 'y_down']:
            setattr(self, attr, entry[attr])
          break

  def is_ready(self):
    return self.y is not None

  def is_ready_for_eval(self):
    return self.algo.is_ready()

  def set_efficiency(self, x, x_up, x_down, y, y_up, y_down):
    entry = { 'bins': self.bins }
    for attr, value in [ ('x', x), ('x_up', x_up), ('x_down', x_down), ('y', y), ('y_up', y_up), ('y_down', y_down) ]:
      if type(value) != list:
        value = value.tolist()
      setattr(self, attr, value)
      entry[attr] = value
    if os.path.exists(self.path):
      with open(self.path, 'r') as f:
        data = json.load(f)
    else:
      data = []
      os.makedirs(os.path.dirname(self.path), exist_ok=True)
    data.append(entry)
    with open(self.path, 'w') as f:
      json.dump(data, f, indent=2)

  def get_total_and_passed(self):
    total = self.dataset[self.column]
    if self.tau_var:
      mask_total = self.dataset['L1Tau_type'] == TauType.tau
      mask = self.algo.get_tau_pass_mask(self.dataset) & mask_total
      total = total[mask_total]
    else:
      mask = self.algo.get_pass_mask(self.dataset)
    passed = self.dataset[self.column][mask]
    if self.tau_var:
      total = ak.flatten(total)
      passed = ak.flatten(passed)
    return total, passed

  def eval(self):
    total, passed = self.get_total_and_passed()
    eff = get_differential_efficiency(self.bins, total=total, passed=passed)
    self.set_efficiency(eff['x'], eff['x_up'], eff['x_down'], eff['values']['passed']['y'],
                        eff['values']['passed']['y_up'], eff['values']['passed']['y_down'])

class VariableCollection(CollectionBase):
  def __init__(self, cfg, base_dir, algorithms, datasets):
    self.variables = {}
    for var_name, var_entry in cfg.items():
      tau_var = var_entry.get('tau_var', True)
      self.variables[var_name] = {}
      for ds_name in var_entry['datasets']:
        dataset = datasets[ds_name]
        self.variables[var_name][ds_name] = {}
        for algo_name, algo in algorithms.items():
          if tau_var and not algo.valid_per_tau: continue
          var_path = os.path.join(base_dir, var_name, ds_name, f'{algo_name}.json')
          self.variables[var_name][ds_name][algo_name] = Variable(var_name, var_entry, var_path, algo, dataset)

  def items(self):
    for var_name, var_entry in self.variables.items():
      for ds_name, ds_entry in var_entry.items():
        for algo_name, variable in ds_entry.items():
          yield f'{var_name}/{ds_name}/{algo_name}', variable

  def get(self, var_name, ds_name, algo_name):
    return self.variables[var_name][ds_name][algo_name]

class Plot:
  def __init__(self, name, output, entries):
    self.name = name
    self.output = output
    self.entries = entries

  def is_ready(self):
    return os.path.exists(self.output)

  def is_ready_for_eval(self):
    for entry in self.entries:
      if not entry['variable'].is_ready():
        return False
    return True

  def eval(self):
    if not self.is_ready_for_eval():
      raise RuntimeError(f'Plot {self.name} is not ready for evaluation')
    os.makedirs(os.path.dirname(self.output), exist_ok=True)
    with PdfPages(self.output) as pdf:
      fig, ax = plt.subplots(1, 1, figsize=(7, 7))
      legend_entries = []
      legend_names = []
      for entry in self.entries:
        var = entry['variable']
        plot_entry = ax.errorbar(var.x, var.y, xerr=(var.x_down, var.x_up),
                                 yerr=(var.y_down, var.y_up), fmt='.', color=entry['color'],
                                 markersize=8, linestyle='none')
        legend_entries.append(plot_entry)
        legend_names.append(entry['algo'])
      ax.legend(legend_entries, legend_names, loc='lower right')

      ax.set_xlabel(var.xlabel)
      ax.set_ylabel('Efficiency')
      ax.set_xscale(var.xscale)
      ax.set_xlim(*var.xlim)
      if var.ylim is not None:
        ax.set_ylim(*var.ylim)

      if var.major_ticks is not None:
        ax.set_xticks(var.major_ticks, minor=False)
      if var.minor_ticks is not None:
        ax.set_xticks(var.minor_ticks, minor=True)
      ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

      pdf.savefig(fig, bbox_inches='tight')

class PlotCollection(CollectionBase):
  def __init__(self, base_dir, cfg, variables):
    self.plots = {}
    for plot_name, plot_cfg in cfg.items():
      self.plots[plot_name] = {}
      for variant_name, variant_entries in plot_cfg['variants'].items():
        self.plots[plot_name][variant_name] = {}
        for variable_name, variable_entry in variables.variables.items():
          if not (plot_cfg['variables'] == 'all' or variable_name in plot_cfg['variables']):
            continue
          self.plots[plot_name][variant_name][variable_name] = {}
          for ds_name, ds_entry in variable_entry.items():
            if not (plot_cfg['datasets'] == 'all' or ds_name in plot_cfg['datasets']):
              continue
            plot_path = os.path.join(base_dir, plot_name, variant_name, ds_name, f'{variable_name}.pdf')
            entries = []
            for variant_entry in variant_entries:
              algo_name = variant_entry['algo']
              if algo_name not in ds_entry: continue
              entry = copy.deepcopy(variant_entry)
              entry['variable'] = ds_entry[entry['algo']]
              entries.append(entry)
            self.plots[plot_name][variant_name][variable_name][ds_name] = \
              Plot(plot_name, plot_path, entries)

  def items(self):
    for plot_name, plot_entry in self.plots.items():
      for variant_name, variant_entry in plot_entry.items():
        for variable_name, variable_entry in variant_entry.items():
          for ds_name, plot in variable_entry.items():
            yield f'{plot_name}/{variant_name}/{variable_name}/{ds_name}', plot

class Algorithm:
  def __init__(self, name, cfg, algo_params_dir, datasets, lut_bins):
    try:
      self.name = name
      self.algo = cfg['algo']
      self.columns = set()
      self.initialized = True
      if self.algo == 'composite':
        self.is_composite = True
        self.sub_algos = cfg['sub_algos']
        self.initialized = False
      else:
        self.is_composite = False
        self.valid_per_tau = cfg.get('valid_per_tau', True)
        if self.algo == 'l1_flag':
          self.l1_flag = cfg['l1_flag']
          self.columns.add(self.l1_flag)
        else:
          self.n_taus = int(cfg['n_taus'])
          if self.algo == 'default':
            self.pt_iso_thr = float(cfg.get('pt_iso_thr', -1))
            self.pt_noiso_thr = float(cfg.get('pt_noiso_thr', -1))
            self.columns.update(['L1Tau_pt', 'L1Tau_eta', 'L1Tau_hwIso'])
          elif self.algo == 'nn':
            self.lut_var = cfg['lut_var']
            self.lut_bins = lut_bins[cfg['lut_bins']]
            self.columns.update([ 'L1Tau_NNtag', 'L1Tau_ptReg', self.lut_var ])
            self.sel_fn = mk_sel_fn(self.lut_bins, self.lut_var)
            opt = cfg['thresholds_opt']
            self.thresholds_opt = {}
            self.thresholds_opt['dataset_eff'] = datasets[opt['dataset_eff']]
            self.thresholds_opt['dataset_rate'] = datasets[opt['dataset_rate']]
            self.thresholds_opt['extra_algos_eff'] = opt.get('extra_algos_eff', [])
            self.thresholds_opt['target_rate'] = opt['target_rate']
            self.thresholds_opt['initial_thresholds'] = opt.get('initial_thresholds', None)
            self.thresholds_opt['step'] = opt.get('step', 0.01)
            if 'tau_eff_var' in opt:
              self.thresholds_opt['tau_eff_var'] = (opt['tau_eff_var'], opt['tau_eff_dataset'], opt['tau_eff_algo'])
              self.thresholds_opt['tau_eff_scale'] = opt.get('tau_eff_scale', 1.)
            else:
              self.thresholds_opt['tau_eff_var'] = None
              self.thresholds_opt['tau_eff_scale'] = None
            self.initialized = False
            self.thresholds = None
            self.thresholds_path = os.path.join(algo_params_dir, f'{self.name}.json')
            if os.path.exists(self.thresholds_path):
              with open(self.thresholds_path, 'r') as f:
                params = json.load(f)
                entry = params[0]
                entry_bins = np.array(entry['lut_bins'], dtype=np.float32)
                if not np.array_equal(entry_bins, self.lut_bins):
                  raise RuntimeError(f'Algorithm {self.name}: stored bins {entry_bins} != {self.lut_bins}')
                self.thresholds = np.array(entry['thresholds'], dtype=np.float32)
          else:
            raise RuntimeError(f'Unknown algorithm {self.algo}')
    except KeyError as e:
      raise RuntimeError(f'algorithm {name}: missing key {e}')

  def initialize_composite(self, algos):
    if not self.initialized and self.is_composite:
      self.sub_algos = { algo_name: algos[algo_name] for algo_name in self.sub_algos }
      self.valid_per_tau = True
      for algo in self.sub_algos.values():
        self.columns.update(algo.columns)
        self.valid_per_tau = self.valid_per_tau and algo.valid_per_tau
      self.initialized = True

  def initialize_simple(self, algos, event_metrics, variables):
    if self.initialized or self.is_composite:
      return
    if self.algo == 'nn':
      self.thresholds_opt['target_rate'] = event_metrics.get('rate', self.thresholds_opt['target_rate'])
      self.thresholds_opt['extra_algos_eff'] = \
        { algo_name: algos[algo_name] for algo_name in self.thresholds_opt['extra_algos_eff'] }
      if self.thresholds_opt['initial_thresholds'] is not None:
        self.thresholds_opt['initial_thresholds'] = algos[self.thresholds_opt['initial_thresholds']]
      if self.thresholds_opt['tau_eff_var'] is not None:
        var_name, ds_name, algo_name = self.thresholds_opt['tau_eff_var']
        self.thresholds_opt['tau_eff_var'] = variables.get(var_name, ds_name, algo_name)
    self.initialized = True

  def set_thresholds(self, thresholds):
    if self.algo != 'nn':
      raise RuntimeError(f'Algorithm {self.name} is not a neural network')
    self.thresholds = np.array(thresholds, dtype=np.float32)
    with open(self.thresholds_path, 'w') as f:
      json.dump([ { 'lut_bins': self.lut_bins.tolist(), 'thresholds': thresholds.tolist() }], f)

  def is_ready(self):
    if not self.initialized:
      return False
    if self.is_composite:
      for sub_algo in self.sub_algos.values():
        if not sub_algo.is_ready():
          return False
    else:
      if self.algo == 'nn':
        if self.thresholds is None:
          return False
    return True

  def is_ready_for_opt(self):
    if self.algo != 'nn':
      return False
    if type(self.thresholds_opt['target_rate']) != float:
      if not self.thresholds_opt['target_rate'].is_ready():
        return False
      self.thresholds_opt['target_rate'] = self.thresholds_opt['target_rate'].get_value()
    if self.thresholds_opt['initial_thresholds'] is not None:
      if type(self.thresholds_opt['initial_thresholds']) != np.ndarray:
        if self.thresholds_opt['initial_thresholds'].thresholds is None:
          return False
        self.thresholds_opt['initial_thresholds'] = self.thresholds_opt['initial_thresholds'].thresholds
    if self.thresholds_opt['tau_eff_var'] is not None:
      if not self.thresholds_opt['tau_eff_var'].is_ready():
        return False
    for extra_algo in self.thresholds_opt['extra_algos_eff'].values():
      if not extra_algo.is_ready():
        return False
    return True

  def _get_combined_mask(self, dataset, mask_fn):
    pass_mask = None
    for sub_algo in self.sub_algos.values():
      pass_sub = getattr(sub_algo, mask_fn)(dataset)
      if pass_mask is None:
        pass_mask = pass_sub
      else:
        pass_mask = pass_mask | pass_sub
    return pass_mask

  def get_tau_pass_mask(self, dataset):
    if not self.valid_per_tau:
      raise RuntimeError(f'Algorithm {self.name} is not valid per tau')
    if not self.is_ready():
      raise RuntimeError(f'Algorithm {self.name} is not ready')
    if self.is_composite:
      return self._get_combined_mask(dataset, 'get_tau_pass_mask')
    else:
      if self.algo == 'default':
        df = dataset.get_input()
        if self.pt_iso_thr > 0:
          pt_thr = self.pt_iso_thr
          require_iso = True
        else:
          pt_thr = self.pt_noiso_thr
          require_iso = False
        return get_tau_pass_mask_default(df['L1Tau_pt'], df['L1Tau_eta'], df['L1Tau_hwIso'], pt_thr, require_iso)
      elif self.algo == 'nn':
        df = dataset.get_input()
        df_scores = dataset.get_scores()
        return get_tau_pass_mask_nn(df['L1Tau_pt'], df['L1Tau_eta'], df_scores['L1Tau_NNtag'], df_scores['L1Tau_ptReg'],
                                self.thresholds, self.sel_fn)
      else:
        raise RuntimeError(f'get_tau_pass_mask: algorithm {self.algo} not supported')

  def get_pass_mask(self, dataset):
    if not self.is_ready():
      raise RuntimeError(f'Algorithm {self.name} is not ready')
    if self.is_composite:
      return self._get_combined_mask(dataset, 'get_pass_mask')
    else:
      if self.algo == 'l1_flag':
        return dataset.get_input()[self.l1_flag]
      elif self.algo in 'default':
        df = dataset.get_input()
        return get_pass_mask_default_or(df['L1Tau_pt'], df['L1Tau_eta'], df['L1Tau_hwIso'], self.pt_iso_thr,
                                        self.pt_noiso_thr, self.n_taus)
      elif self.algo == 'nn':
        df = dataset.get_input()
        df_scores = dataset.get_scores()
        return get_pass_mask_nn(df['L1Tau_pt'], df['L1Tau_eta'], df_scores['L1Tau_NNtag'], df_scores['L1Tau_ptReg'],
                                self.thresholds, self.sel_fn, self.n_taus)
      else:
        raise RuntimeError(f'get_pass_mask: algorithm {self.algo} not supported')

  def get_passed(self, dataset):
    mask = self.get_pass_mask(dataset)
    return pass_masks_to_count(mask)

  def optimize_thresholds(self):
    if self.algo != 'nn':
      raise RuntimeError(f'optimize_thresholds: algorithm {self.name} is not a neural network')
    if not self.is_ready_for_opt():
      raise RuntimeError(f'optimize_thresholds: algorithm {self.name} is not ready for optimization')

    dataset_eff = self.thresholds_opt['dataset_eff']
    dataset_rate = self.thresholds_opt['dataset_rate']
    df_eff = dataset_eff.get_input()
    df_eff_scores = dataset_eff.get_scores()
    df_rate = dataset_rate.get_input()
    df_rate_scores = dataset_rate.get_scores()

    n_nn_thrs = len(self.lut_bins)

    extra_cond_eff=None
    for algo in self.thresholds_opt['extra_algos_eff'].values():
      algo_mask = algo.get_pass_mask(dataset_eff)
      if extra_cond_eff is None:
        extra_cond_eff = algo_mask
      else:
        extra_cond_eff = extra_cond_eff | algo_mask

    min_passed_args = {}
    if self.thresholds_opt['tau_eff_var'] is not None:
      tau_eff_var = self.thresholds_opt['tau_eff_var']
      _, tau_passed = tau_eff_var.get_total_and_passed()
      min_passed_args['min_passed_L1Tau_pt'] = tau_eff_var.dataset['L1Tau_pt']
      min_passed_args['min_passed_L1Tau_eta'] = tau_eff_var.dataset['L1Tau_eta']
      min_passed_args['min_passed_L1Tau_NNtag'] = tau_eff_var.dataset['L1Tau_NNtag']
      min_passed_args['min_passed_L1Tau_ptReg'] = tau_eff_var.dataset['L1Tau_ptReg']
      min_passed_args['min_passed_L1Tau_type'] = tau_eff_var.dataset['L1Tau_type']
      min_passed_args['min_passed_var'] = tau_eff_var.dataset[tau_eff_var.column]
      min_passed_args['min_passed_bins'] = tau_eff_var.bins
      #min_passed_args['min_passed_counts'] = np.floor(np.histogram(tau_passed, bins=tau_eff_var.bins)[0] * 0.95)
      min_passed_args['min_passed_counts'] = np.array(np.floor(np.histogram(tau_passed, bins=tau_eff_var.bins)[0] \
        * self.thresholds_opt['tau_eff_scale']), dtype=int)


    eff, rate, nn_thrs = find_nn_thrs(df_eff['L1Tau_pt'], df_eff['L1Tau_eta'], df_eff_scores['L1Tau_NNtag'],
                                      df_eff_scores['L1Tau_ptReg'], df_rate['L1Tau_pt'], df_rate['L1Tau_eta'],
                                      df_rate_scores['L1Tau_NNtag'], df_rate_scores['L1Tau_ptReg'], self.sel_fn,
                                      n_nn_thrs, self.n_taus, self.thresholds_opt['target_rate'],
                                      initial_thrs=self.thresholds_opt['initial_thresholds'],
                                      extra_cond_eff=extra_cond_eff, step=self.thresholds_opt['step'], verbose=1,
                                      **min_passed_args)
    thrs_str = ', '.join([ f'{thr:.2f}' for thr in nn_thrs ])
    print(f'Optimized thresholds for {self.name}: eff={eff}, rate={rate}, thrs=[{thrs_str}]')
    self.set_thresholds(nn_thrs)

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

    event_metrics_dir = os.path.join(base_dir, self.cfg['event_metrics_dir'])
    self.event_metrics = EventMetricCollection(self.cfg['event_metrics'], event_metrics_dir, self.algos, self.datasets)
    variables_dir = os.path.join(base_dir, self.cfg['differential_efficiency_dir'])
    self.variables = VariableCollection(self.cfg['variables'], variables_dir, self.algos, self.datasets)
    for var_name, var in self.variables.items():
      var.dataset.columns.add(var.column)
      var.dataset.columns.add('L1Tau_type')

    algo_columns = set()
    for algo in self.algos.values():
      algo.initialize_simple(self.algos, self.event_metrics, self.variables)
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

  n_metrics_not_ready = -1
  n_vars_not_ready = -1
  n_plots_not_ready = -1
  while n_metrics_not_ready != 0 or n_vars_not_ready != 0:
    n_metrics_ready, n_metrics_evaluated, n_metrics_not_ready = setup.event_metrics.eval()
    print(f'Metrics: {n_metrics_ready} ready, {n_metrics_evaluated} evaluated in this iteration,'
          f' {n_metrics_not_ready} not ready')
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
