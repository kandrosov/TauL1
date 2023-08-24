import numpy as np
import pandas as pd
import os
from statsmodels.stats.proportion import proportion_confint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker


def eval_rate(dataset_path, output_path, thr):
  df = pd.read_hdf(dataset_path, key='events')

  tau_pts = np.arange(20, 256, 1)

  n_total = df.shape[0]
  n_passed = np.zeros((len(tau_pts), 2, 2))
  eff = np.zeros((len(tau_pts), 2, 2, 3))

  for idx in range(len(tau_pts)):
    tau_pt = tau_pts[idx]
    passed_cut = np.zeros(df.shape[0], dtype=int)
    passed_dnn = np.zeros(df.shape[0], dtype=int)
    for tau_idx in range(12):
      sel = (df[f'L1Tau_pt_{tau_idx}'] > tau_pt) & (np.abs(df[f'L1Tau_eta_{tau_idx}']) <= 2.131)
      passed_cut += sel & (df[f'L1Tau_hwIso_{tau_idx}'] > 0 )
      passed_dnn += sel & (df[f'nn_score_{tau_idx}'] > thr)

    for n in range(2):
      n_passed[idx, 0, n] = np.sum(passed_cut >= n + 1)
      n_passed[idx, 1, n] = np.sum(passed_dnn >= n + 1)
  eff[:, :, :, 0] = n_passed / n_total
  ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
  eff[:, :, :, 1] = ci_low
  eff[:, :, :, 2] = ci_upp

  rate_hz = 40e6
  eff *= rate_hz

  pt_ranges = [ (70, 150), (20, 50),  ]
  rate_ranges = [ (0, 5000), (0, 10000),  ]

  for n in range(2):
    with PdfPages(os.path.join(output_path, f'rate_{n+1}taus.pdf')) as pdf:
      fig, ax = plt.subplots(1, 1, figsize=(7, 7))
      legend_entries = []
      legend_names = []
      for idx, (entry_name, color) in enumerate([('L1 tau iso', 'r'), ('ShallowTau', 'b')]):
        entry, = ax.plot(tau_pts, eff[:, idx, n, 0], color=color, linestyle='-', linewidth=2)
        ax.fill_between(tau_pts, eff[:, idx, n, 1], eff[:, idx, n, 2], alpha=0.2, color=color)
        legend_entries.append(entry)
        legend_names.append(entry_name)
      ax.legend(legend_entries, legend_names, loc='upper right')

      ax.set_xlabel('L1 tau pT threshold (GeV)')
      ax.set_ylabel('Rate (Hz)')
      #ax.set_xscale(bin_settings[var].get('xscale', 'linear'))
      ax.set_xlim(pt_ranges[n][0], pt_ranges[n][-1])
      ax.set_ylim(rate_ranges[n][0], rate_ranges[n][-1])
      ax.grid(True, which='both')

      pdf.savefig(fig, bbox_inches='tight')
    for pt_idx, pt in enumerate(tau_pts):
      print(f'{n+1} taus: pt > {pt}, iso_rate = {eff[pt_idx, 0, n, 0]}, nn_rate = {eff[pt_idx, 1, n, 0]}')

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--thr', required=True, type=float)
  args = parser.parse_args()

  eval_rate(args.dataset, args.output, args.thr)
