import awkward as ak
from enum import Enum
import json
import os

from .eval_tools import get_eff, get_rate,  get_differential_efficiency, is_gen_tau

class ValueType(Enum):
  efficiency = 0
  rate = 1

class VariableScope(Enum):
  dataset = 0
  event = 1
  tau = 2

class BaseVariable:
  def __init__(self, name, full_name, path, algo, dataset, cfg):
    self.name = name
    self.full_name = full_name
    self.path = path
    self.algo = algo
    self.dataset = dataset
    self.ds_columns = set()
    self.scope = VariableScope[cfg['scope']]
    self.value_type = cfg.get('value_type', ValueType.efficiency.name)
    self.value_type = ValueType[self.value_type]
    self.data = None
    if os.path.exists(self.path):
      with open(self.path, 'r') as f:
        data = json.load(f)
      self.from_json(data)

  def from_json(self, data):
    self.data = data

  def to_json(self):
    return self.data

  def save(self):
    os.makedirs(os.path.dirname(self.path), exist_ok=True)
    data = self.to_json()
    with open(self.path, 'w') as f:
      json.dump(data, f, indent=2)

  @property
  def value(self):
    self.check_ready()
    return self.data['value']

  @property
  def value_up(self):
    self.check_ready()
    return self.data['value_up']

  @property
  def value_down(self):
    self.check_ready()
    return self.data['value_down']

  def is_ready(self):
    return self.data is not None

  def check_ready(self):
    if not self.is_ready():
      raise RuntimeError(f'Event variable {self.name} is not ready')

  def is_ready_for_eval(self):
    return self.algo.is_ready()

  def eval(self):
    if self.is_ready():
      raise RuntimeError(f'Variable {self.full_name} is already evaluated')
    if not self.is_ready_for_eval():
      raise RuntimeError(f'Variable {self.full_name} is not ready for evaluation')


class GlobalVariable(BaseVariable):
  def __init__(self, name, full_name, path, algo, dataset, cfg):
    super().__init__(name, full_name, path, algo, dataset, cfg)
    if self.value_type == ValueType.efficiency:
      self.value_fn = get_eff
    elif self.value_type == ValueType.rate:
      self.value_fn = get_rate
    else:
      raise RuntimeError(f'Unsupported value type {self.value_type}')

  def set_value(self, value, value_up, value_down):
    self.data = { 'value': float(value), 'value_up': float(value_up), 'value_down': float(value_down) }
    self.save()

  def eval(self):
    super().eval()
    require_gen_match = self.value_type == ValueType.efficiency
    n_total = self.dataset.get_size()
    n_passed = self.algo.get_passed(self.dataset, require_gen_match=require_gen_match)
    value, value_up, value_down = self.value_fn(n_passed, n_total, return_errors=True)
    self.set_value(value, value_up, value_down)

class Variable(BaseVariable):
  def __init__(self, name, full_name, path, algo, dataset, cfg):
    self.bins = cfg['bins']
    super().__init__(name, full_name, path, algo, dataset, cfg)
    self.full_data = []
    self.column = self.name if 'column' not in cfg else cfg['column']
    self.ds_columns.add(self.column)
    self.major_ticks = cfg.get('major_ticks', None)
    self.minor_ticks = cfg.get('minor_ticks', None)
    self.xlabel = cfg['xlabel']
    self.xscale = cfg.get('xscale', 'linear')
    self.yscale = cfg.get('yscale', 'linear')
    self.xlim = cfg.get('xlim', None)
    if self.xlim is None:
      self.xlim = (self.bins[0], self.bins[-1])
    self.ylim = cfg.get('ylim', None)

  def from_json(self, data):
    self.full_data = data
    for entry in data:
      if entry['bins'] == self.bins:
        self.data = entry
        return

  def to_json(self):
    return self.full_data

  @property
  def x(self):
    self.check_ready()
    return self.data['x']

  @property
  def x_up(self):
    self.check_ready()
    return self.data['x_up']

  @property
  def x_down(self):
    self.check_ready()
    return self.data['x_down']

  def set_value(self, **kwargs):
    if self.data is None:
      self.data = { 'bins': self.bins }
      self.full_data.append(self.data)
    for attr in [ 'x', 'x_up', 'x_down', 'value', 'value_up', 'value_down' ]:
      v = kwargs.pop(attr)
      if type(v) != list:
        v = v.tolist()
      self.data[attr] = v
    if len(kwargs) > 0:
      raise RuntimeError(f'Variable.set_value: unexpected arguments {kwargs}')
    self.save()

  def get_total_and_passed(self):
    require_gen_match = self.value_type == ValueType.efficiency
    total = self.dataset[self.column]
    if self.scope == VariableScope.tau:
      mask_total = is_gen_tau(self.dataset['L1Tau_type'])
      mask = self.algo.get_tau_pass_mask(self.dataset, require_gen_match=require_gen_match)
      total = total[mask_total]
    else:
      mask = self.algo.get_pass_mask(self.dataset, require_gen_match=require_gen_match)
    passed = self.dataset[self.column][mask]
    if self.scope == VariableScope.tau:
      total = ak.flatten(total)
      passed = ak.flatten(passed)
    return total, passed

  def eval(self):
    super().eval()
    total, passed = self.get_total_and_passed()
    eff = get_differential_efficiency(self.bins, total=total, passed=passed)
    self.set_value(x=eff['x'], x_up=eff['x_up'], x_down=eff['x_down'], value=eff['values']['passed']['y'],
                   value_up=eff['values']['passed']['y_up'], value_down=eff['values']['passed']['y_down'])

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

class VariableCollection(CollectionBase):
  def __init__(self, cfg, base_dir, algorithms, datasets):
    self.variables = {}
    for var_name, var_entry in cfg.items():
      scope = VariableScope[var_entry['scope']]
      self.variables[var_name] = {}
      for ds_name in var_entry['datasets']:
        dataset = datasets[ds_name]
        self.variables[var_name][ds_name] = {}
        for algo_name, algo in algorithms.items():
          if scope == VariableScope.tau and not algo.valid_per_tau: continue
          var_path = os.path.join(base_dir, var_name, ds_name, f'{algo_name}.json')
          full_name = f'{var_name}/{ds_name}/{algo_name}'
          var_type = GlobalVariable if scope == VariableScope.dataset else Variable
          var = var_type(var_name, full_name, var_path, algo, dataset, var_entry)
          self.variables[var_name][ds_name][algo_name] = var

  def items(self):
    for var_name, var_entry in self.variables.items():
      for ds_name, ds_entry in var_entry.items():
        for algo_name, variable in ds_entry.items():
          yield f'{var_name}/{ds_name}/{algo_name}', variable

  def get(self, var_name=None, ds_name=None, algo_name=None):
    vars = self.variables
    prefix = []
    for name, entry in [ ('var_name', var_name), ('ds_name', ds_name), ('algo_name', algo_name) ]:
      if entry is None:
        if len(vars) != 1:
          msg = f'Variable collection has multiple {name}'
          if prefix != '':
            msg += f' for {"/".join(prefix)}'
          raise RuntimeError(msg)
        entry = list(vars.keys())[0]
      prefix.append(entry)
      vars = vars[entry]
    return vars