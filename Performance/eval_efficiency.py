import numpy as np
import uproot
import os
import shutil
import sys
import yaml
import awkward as ak

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import *

def get_shortest_interval(x, alpha=0.68, step=0.01, rel_tolerance=0.01):
  assert(alpha > 0 and alpha < 1)
  q_values_left = np.arange(0, 1 - alpha + step, step).tolist()
  n_tot = len(q_values_left)
  half_interval = (1 - alpha) / 2
  q_values_left = sorted(q_values_left, key=lambda x: abs(half_interval - x))
  q_values_left = np.array(q_values_left)
  q_values_right = q_values_left + alpha
  q_values = np.concatenate((q_values_left, q_values_right))
  q_limits = np.quantile(x, q_values)
  q_limits_left = q_limits[:n_tot]
  q_limits_right = q_limits[n_tot:]
  shortest = None
  shortest_n = -1
  for n in range(n_tot):
    current_len = q_limits_right[n] - q_limits_left[n]
    if shortest is None or (shortest > 0 and (shortest - current_len) / shortest > rel_tolerance):
      shortest = current_len
      shortest_n = n
  assert(shortest_n != -1)
  return (q_limits_left[shortest_n], q_limits_right[shortest_n]), \
         (q_values_left[shortest_n], q_values_right[shortest_n])

def plot_bin(diff, bin_low, bin_high, output_path, cfg, cfg_est):
  bin_name = cfg['bin_name'].format(bin_low, bin_high)
  if 'range' in cfg:
    abs_range = cfg['range']
  else:
    abs_range = cfg['range_factor'] * bin_high
  with PdfPages(os.path.join(output_path, f'{bin_name}.pdf')) as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for est_name, est_entry in diff.items():
      entry = ax.hist(est_entry, bins=cfg['n_bins'], range=(-abs_range, abs_range), histtype='step',
                      color=cfg_est[est_name]['color'], label=est_name)
    ax.legend(loc='upper right')
    ax.set_xlabel(cfg['x_label'])
    ax.set_ylabel(cfg['y_label'])
    ax.set_title(cfg['title'].format(bin_low, bin_high))
    pdf.savefig(fig, bbox_inches='tight')
  plt.close()

def plot_combined(total, output_file, cfg, cfg_est):
  with PdfPages(output_file) as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for est_name, est_entry in total.items():
      entry,  = ax.plot(est_entry['x'], est_entry['y'], '-', color=cfg_est[est_name]['color'])
      ax.fill_between(est_entry['x'], est_entry['y_down'], est_entry['y_up'], facecolor=cfg_est[est_name]['color'],
                      alpha=cfg['alpha_fill'], linewidth=0, label=est_name)


    ax.set_xlim(((cfg['bins'][0] + cfg['bins'][1]) / 2, (cfg['bins'][-2] + cfg['bins'][-1]) / 2))
    ax.legend(loc='upper right')
    ax2 = ax.twinx()
    ax2.plot([], [], label='mean value', color='black')
    ax2.fill_between([], [], [], facecolor='black', alpha=cfg['alpha_fill'], linewidth=0, label='68% CI')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='upper center')
    ax.set_xlabel(cfg['x_label'])
    ax.set_ylabel(cfg['y_label'])
    ax.set_title(cfg['title'])
    pdf.savefig(fig, bbox_inches='tight')
  plt.close()

def eval_efficiency(cfg, dataset_file, scores_file, output_path):
  if os.path.exists(output_path):
    shutil.rmtree(output_path)
  os.makedirs(output_path)
  combined_path = output_path + '.pdf'
  if os.path.exists(combined_path):
    os.remove(combined_path)

  events = uproot.open(dataset_file)['Events'].arrays(cfg['columns']['dataset'])
  scores = uproot.open(scores_file)['Events'].arrays(cfg['columns']['scores'])

  bins = cfg['bin_settings']['combined']['bins']

  total = {}
  for est_name in cfg['estimators']:
    total[est_name] = { 'x': [], 'y': [], 'y_up': [], 'y_down': [] }
  for bin_idx in range(len(bins) - 1):
    bin_sel = (events['L1Tau_gen_pt'] >= bins[bin_idx]) & (events['L1Tau_gen_pt'] < bins[bin_idx+1]) \
              & (events['L1Tau_type'] == TauType.tau)
    diff = {}
    diff_rel = {}
    for est_name, est_entry in cfg['estimators'].items():
      if 'validity' in est_entry:
        est_sel = bin_sel & eval(est_entry['validity'])
      else:
        est_sel = bin_sel
      est = ak.flatten(eval(est_entry['values'])[est_sel])
      gen_pt = ak.flatten(events['L1Tau_gen_pt'][est_sel])
      if len(gen_pt) > 0:
        diff[est_name] = est - gen_pt
        diff_rel[est_name] = diff[est_name] / gen_pt
        (y_down, y_up), _ = get_shortest_interval(diff_rel[est_name])
        y = np.mean(diff_rel[est_name])
        total[est_name]['x'].append((bins[bin_idx] + bins[bin_idx+1]) / 2)
        total[est_name]['y'].append(y)
        total[est_name]['y_up'].append(y_up)
        total[est_name]['y_down'].append(y_down)
    if len(diff) > 0:
      plot_bin(diff, bins[bin_idx], bins[bin_idx+1], output_path, cfg['bin_settings']['diff'], cfg['estimators'])
      plot_bin(diff_rel, bins[bin_idx], bins[bin_idx+1], output_path, cfg['bin_settings']['diff_rel'],
               cfg['estimators'])
  plot_combined(total, combined_path, cfg['bin_settings']['combined'], cfg['estimators'])

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', required=True, type=str)
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--scores', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  args = parser.parse_args()

  with open(args.cfg) as f:
    cfg = yaml.safe_load(f)

  eval_efficiency(cfg, args.dataset, args.scores, args.output)
