import numpy as np
import pandas as pd
import os
import sys
import shutil
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import proportion_confint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import *

bin_settings = {
  'nPV': {
    'bins': [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 17., 20.],
    'major_ticks': [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 17., 20.],
    'xlabel': r'number of good HLT PF PV',
  },
  'L1Tau_gen_pt': {
    'bins': [ 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350,
              400, 450, 500, 600, 800, 1000 ],
    'major_ticks': [ 20, 30, 40, 50, 70, 100, 200, 300, 500, 1000 ],
    'minor_ticks': [ 60, 80, 90, 400, 600, 700, 800, 900 ],
    'xlabel': r'gen visible $p_{T}$ (GeV)',
    'xscale': 'log',
  },
  'L1Tau_gen_eta': {
    'bins': np.arange(-2.5, 2.6, 0.5),
    'xlabel': r'gen visible $\tau_h$ $\eta$ (GeV)',
  },
  'L1Tau_pt': {
    #'bins': [ 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 255, 300 ],
    'bins': [ 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 255, 300 ],
    #'major_ticks': [ 20, 30, 40, 50, 70, 100, 140, 200, 255 ],
    'major_ticks': [ 80, 100, 150, 200, 255 ],
    #'minor_ticks': [ 60, 80, 90 ],
    #'minor_ticks': [ 90, 110, 120, 130, 140, 160, 170, 180, 190 ],
    'xlabel': r'L1Tau $p_{T}$ (GeV)',
    'xscale': 'log',
    'ylim': [0.0, 1.005],
  },
  'L1Tau_eta': {
    'bins': np.arange(-2.5, 2.6, 0.5),
    'xlabel': r'L1Tau $\eta$ (GeV)',
  },
  'Jet_pt': {
    'bins': [ 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 300, 500, 1000 ],
    'major_ticks': [ 20, 30, 40, 50, 70, 100, 140, 200, 300, 500, 1000 ],
    'minor_ticks': [ 60, 80, 90 ],
    'xlabel': r'PF Jet $p_{T}$ (GeV)',
    'xscale': 'log',
  },
  'Jet_pt_corr': {
    'bins': [ 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 300, 500, 1000 ],
    'major_ticks': [ 20, 30, 40, 50, 70, 100, 140, 200, 300, 500, 1000 ],
    'minor_ticks': [ 60, 80, 90 ],
    'xlabel': r'PF Jet PNet-corrected $p_{T}$ (GeV)',
    'xscale': 'log',
  },

}

eff_y_label = {
  TauType.e: r'ele faking $\tau_h$ efficiency',
  TauType.jet: r'jet faking $\tau_h$ efficiency',
  TauType.tau: r'$\tau_h$ efficiency',
}

# process.hltVerticesPFSelector = cms.EDFilter( "PrimaryVertexObjectFilter",
#     filterParams = cms.PSet(
#       maxZ = cms.double( 24.0 ),
#       minNdof = cms.double( 4.0 ),
#       maxRho = cms.double( 2.0 ),
#       pvSrc = cms.InputTag( "hltVerticesPF" )
#     ),
#     src = cms.InputTag( "hltVerticesPF" )
# )
# process.hltVerticesPFFilter = cms.EDFilter( "VertexSelector",
#     src = cms.InputTag( "hltVerticesPFSelector" ),
#     cut = cms.string( "!isFake" ),
#     filter = cms.bool( True )
# )

def make_rate_list(tpr, fpr, threasholds, min_tpr_diff=0.001):
  result = []
  prev_tpr = -1
  for n in range(len(tpr)):
    if n != len(tpr) - 1 and tpr[n] - prev_tpr < min_tpr_diff:
      continue
    result.append({
      'tpr': tpr[n],
      'fpr': fpr[n],
      'thr': threasholds[n]
    })
    prev_tpr = tpr[n]
  return result

def write_rate_list(rate_list, output_file):
  with open(output_file, 'w') as f:
    f.write('thr,tpr,fpr\n')
    for item in rate_list:
      f.write(f'{item["thr"]:.3f},{item["tpr"]:.3f},{item["fpr"]:.4f}\n')

def plot_eff(var, df_all, df_thr, df_iso, y_title, output_file):
  hists = {}
  for df_name, df in [('all', df_all), ('thr', df_thr), ('iso', df_iso)]:
    hists[df_name] = np.histogram(df[var], bins=bin_settings[var]['bins'])[0]

  eff = {}
  ds_names = ['thr', 'iso']
  for ds_name in ds_names:
    sel = hists['all'] > 0

    orig_bins = bin_settings[var]['bins']
    bins_all = np.zeros((len(orig_bins) - 1, 2))
    bins_all[:, 0] = orig_bins[:-1]
    bins_all[:, 1] = orig_bins[1:]
    bins = bins_all[sel]
    hist_all = hists['all'][sel]
    hist_num = hists[ds_name][sel]

    central = hist_num / hist_all
    ci_low, ci_upp = proportion_confint(hist_num, hist_all, alpha=1-0.68, method='beta')
    eff[ds_name] = {
      'x': (bins[:, 1] + bins[:, 0]) / 2,
      'x_up': (bins[:, 1] - bins[:, 0]) / 2,
      'x_down': (bins[:, 1] - bins[:, 0]) / 2,
      'y': central,
      'y_up': ci_upp - central,
      'y_down': central - ci_low,
    }

  with PdfPages(output_file) as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    legend_entries = []
    legend_names = []
    for ds_name, entry_name, color in [('iso', 'L1 tau iso', 'r'), ('thr', 'ShallowTau', 'b')]:
      entry = ax.errorbar(eff[ds_name]['x'], eff[ds_name]['y'],
                          xerr=(eff[ds_name]['x_down'], eff[ds_name]['x_up']),
                          yerr=(eff[ds_name]['y_down'], eff[ds_name]['y_up']),
                          fmt='.', color=color, markersize=8, linestyle='none')
      legend_entries.append(entry)
      legend_names.append(entry_name)
    ax.legend(legend_entries, legend_names, loc='lower right')

    ax.set_xlabel(bin_settings[var]['xlabel'])
    ax.set_ylabel(y_title)
    ax.set_xscale(bin_settings[var].get('xscale', 'linear'))
    ax.set_xlim(bin_settings[var]['bins'][0], bin_settings[var]['bins'][-1])
    if 'ylim' in bin_settings[var]:
      ax.set_ylim(*bin_settings[var]['ylim'])

    if 'major_ticks' in bin_settings[var]:
      ax.set_xticks(bin_settings[var]['major_ticks'], minor=False)
      ax.set_xticks(bin_settings[var].get('minor_ticks', []), minor=True)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    pdf.savefig(fig, bbox_inches='tight')


def eval_perf(dataset_path, output_path, vars, thr):
  if os.path.exists(output_path):
    shutil.rmtree(output_path)
  os.makedirs(output_path)

  df = pd.read_hdf(dataset_path, key='taus')
  df['Jet_pt_corr'] = df['Jet_pt'] * df['Jet_PNet_ptcorr']

  df_te = df[(df['L1Tau_type'] == TauType.e) | (df['L1Tau_type'] == TauType.tau)]
  fpr, tpr, threasholds = roc_curve(df_te['L1Tau_type'] == TauType.tau, df_te['nn_score'])
  rate_list = make_rate_list(tpr, fpr, threasholds)
  write_rate_list(rate_list, os.path.join(output_path, 'roc_e.csv'))

  hw_fpr, hw_tpr, hw_threasholds = roc_curve(df_te['L1Tau_type'] == TauType.tau, df_te['L1Tau_hwIso'])
  hw_rate_list = make_rate_list(hw_tpr, hw_fpr, hw_threasholds)
  write_rate_list(hw_rate_list, os.path.join(output_path, 'iso_roc_e.csv'))

  with PdfPages(os.path.join(output_path, 'roc_e.pdf')) as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(tpr, fpr)
    ax.errorbar(hw_tpr, hw_fpr, fmt='o', markersize=2, color='red')
    ax.set_xlabel(r'true $\tau_h$ efficiency')
    ax.set_ylabel(r'prompt electron faking $\tau_h$ efficiency')
    pdf.savefig(fig, bbox_inches='tight')


  df_tj = df[(df['L1Tau_type'] == TauType.jet) | (df['L1Tau_type'] == TauType.tau)]
  fpr, tpr, threasholds = roc_curve(df_tj['L1Tau_type'] == TauType.tau, df_tj['nn_score'])
  rate_list = make_rate_list(tpr, fpr, threasholds)
  write_rate_list(rate_list, os.path.join(output_path, 'roc.csv'))

  hw_fpr, hw_tpr, hw_threasholds = roc_curve(df_tj['L1Tau_type'] == TauType.tau, df_tj['L1Tau_hwIso'])
  hw_rate_list = make_rate_list(hw_tpr, hw_fpr, hw_threasholds)
  write_rate_list(hw_rate_list, os.path.join(output_path, 'iso_roc.csv'))

  with PdfPages(os.path.join(output_path, 'roc.pdf')) as pdf:
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(tpr, fpr)
    ax.errorbar(hw_tpr, hw_fpr, fmt='o', markersize=2, color='red')
    ax.set_xlabel(r'true $\tau_h$ efficiency')
    ax.set_ylabel(r'jet faking $\tau_h$ efficiency')
    pdf.savefig(fig, bbox_inches='tight')

  for tau_type_name in [ 'tau', 'jet', 'e']:
    tau_type = getattr(TauType, tau_type_name)
    pnet_thr = 0.001
    cond_ref = (df['L1Tau_type'] == tau_type) & (df['Jet_PNet_ptcorr'] > pnet_thr) & (df['Jet_pt_corr'] > 20) & (np.abs(df['Jet_eta']) <= 2.1)
    df_tau = df[cond_ref]
    for var in vars:
      cond = cond_ref
      if var != 'L1Tau_pt':
        cond = cond & (df['L1Tau_pt'] >= 20) & (np.abs(df['L1Tau_eta']) <= 2.131)
      # nn_score_formula = ((df['nn_score'] > 0.99) & (df['L1Tau_pt'] >= 20)) | \
      #                    ((df['nn_score'] > 0.9) & (df['L1Tau_pt'] >= 25)) | \
      #                    ((df['nn_score'] > 0.8) & (df['L1Tau_pt'] >= 30)) | \
      #                    ((df['nn_score'] > 0.7) & (df['L1Tau_pt'] >= 35)) | \
      #                    ((df['nn_score'] > 0.6) & (df['L1Tau_pt'] >= 40)) | \
      #                    ((df['nn_score'] > 0.5) & (df['L1Tau_pt'] >= 45)) | \
      #                    ((df['nn_score'] > 0.4) & (df['L1Tau_pt'] >= 50)) | \
      #                    ((df['nn_score'] > 0.2) & (df['L1Tau_pt'] >= 55)) | \
      #                    ((df['nn_score'] > 0.05) & (df['L1Tau_pt'] >= 60)) | \
      #                    ((df['nn_score'] > 0.01) & (df['L1Tau_pt'] >= 70))
      nn_score_formula = ((df['nn_score'] > 1.99) & (df['L1Tau_pt'] >= 60)) | \
                         ((df['nn_score'] > 1.9) & (df['L1Tau_pt'] >= 70)) | \
                         ((df['nn_score'] > 1.8) & (df['L1Tau_pt'] >= 80)) | \
                         ((df['nn_score'] > 1.7) & (df['L1Tau_pt'] >= 90)) | \
                         ((df['nn_score'] > 0.98) & (df['L1Tau_pt'] >= 100)) | \
                         ((df['nn_score'] > 0.8) & (df['L1Tau_pt'] >= 110)) | \
                         ((df['nn_score'] > 0.035) & (df['L1Tau_pt'] >= 120)) | \
                         ((df['nn_score'] > 0.030) & (df['L1Tau_pt'] >= 140)) | \
                         ((df['nn_score'] > 0.02) & (df['L1Tau_pt'] >= 150)) | \
                         ((df['nn_score'] > 0.02) & (df['L1Tau_pt'] >= 200))
      df_thr = df[nn_score_formula & cond]
      df_iso = df[(df['L1Tau_hwIso'] > 0) & (df['L1Tau_pt'] >= 120) & cond ]

      if var not in bin_settings:
        print(f'WARNING: no bin settings for {var}. Skipping.')
        continue
      plot_eff(var, df_tau, df_thr, df_iso, eff_y_label[tau_type],
               os.path.join(output_path, f'eff_{tau_type_name}_{var}.pdf'))

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--thr', required=True, type=float)
  parser.add_argument('--vars', required=False, type=str, default=','.join([
    'nPV', 'L1Tau_gen_pt', 'L1Tau_gen_eta', 'L1Tau_pt', 'L1Tau_eta', 'Jet_pt_corr'
  ]))
  args = parser.parse_args()

  eval_perf(args.dataset, args.output, args.vars.split(','), args.thr)
