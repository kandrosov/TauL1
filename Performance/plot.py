import copy
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker

from .variable import CollectionBase, VariableScope

class PlotBase:
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

  def plot(self):
    raise NotImplementedError('PlotBase.plot not implemented')

  def eval(self):
    if not self.is_ready_for_eval():
      raise RuntimeError(f'Plot {self.name} is not ready for evaluation')
    os.makedirs(os.path.dirname(self.output), exist_ok=True)
    with PdfPages(self.output) as pdf:
      fig = self.plot()
      pdf.savefig(fig, bbox_inches='tight')
      plt.close(fig)

class PlotEfficiency(PlotBase):
  def plot(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    legend_entries = []
    legend_names = []
    for entry in self.entries:
      var = entry['variable']
      plot_entry = ax.errorbar(var.x, var.value, xerr=(var.x_down, var.x_up),
                                yerr=(var.value_down, var.value_up), fmt='.', color=entry['color'],
                                markersize=8, linestyle='none')
      legend_entries.append(plot_entry)
      title = entry.get('title', entry['algo'])
      legend_names.append(title)
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
      ax.set_xticks(var.minor_ticks, minor=True, labels=[ '' ] * len(var.minor_ticks))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    return fig

class PlotDatasetEfficiency(PlotBase):
  def plot(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = []
    x_label = []
    bar_labels = []
    for entry in reversed(self.entries):
      var = entry['variable']
      x.append(var.value)
      title = entry.get('title', entry['algo'])
      x_label.append(title)
      err = max(var.value_up, var.value_down)
      bar_label = f'{var.value * 100:.1f} $\\pm$ {err * 100:.1f} %'
      bar_labels.append(bar_label)
    bar = ax.barh(x_label, x, color='blue', alpha=0.5)
    ax.bar_label(bar, labels=bar_labels, padding=-100, color='white', fontweight='bold', fontsize=12)
    ax.set_xlabel('Efficiency')
    return fig

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
              scope = entry['variable'].scope
              entries.append(entry)
            plot_type = PlotDatasetEfficiency if scope == VariableScope.dataset else PlotEfficiency
            self.plots[plot_name][variant_name][variable_name][ds_name] = plot_type(plot_name, plot_path, entries)

  def items(self):
    for plot_name, plot_entry in self.plots.items():
      for variant_name, variant_entry in plot_entry.items():
        for variable_name, variable_entry in variant_entry.items():
          for ds_name, plot in variable_entry.items():
            yield f'{plot_name}/{variant_name}/{variable_name}/{ds_name}', plot
