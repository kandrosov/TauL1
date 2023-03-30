import os
import sys
import math
import numpy as np
import yaml

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.dirname(file_dir)
  if base_dir not in sys.path:
    sys.path.append(base_dir)
  __package__ = os.path.split(file_dir)[-1]

from .AnalysisTools import *

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()

class MixStep:
  pass

def GetBinContent(hist, pt, eta):
  x_axis = hist.GetXaxis()
  x_bin = x_axis.FindFixBin(pt)
  y_axis = hist.GetYaxis()
  y_bin = y_axis.FindFixBin(eta)
  return hist.GetBinContent(x_bin, y_bin)

def analyse_mix(cfg_file):
  with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)
  mix_steps = []
  pt_bin_edges = cfg['bin_edges']['pt']
  eta_bin_edges = cfg['bin_edges']['eta']
  batch_size = 0
  for bin_idx, bin in enumerate(cfg['bins']):
    if len(bin['counts']) != len(pt_bin_edges) - 1:
      raise ValueError("Number of counts does not match number of pt bins")
    for pt_bin_idx, count in enumerate(bin['counts']):
      if count == 0: continue
      step = MixStep()
      step.input_setups = bin['input_setups']
      step.inputs = []
      for input_setup in bin['input_setups']:
        step.inputs.extend(cfg['inputs'][input_setup])
      selection = cfg['bin_selection'].format(pt_low=pt_bin_edges[pt_bin_idx],
                                              pt_high=pt_bin_edges[pt_bin_idx + 1],
                                              eta_low=eta_bin_edges[bin['eta_bin']],
                                              eta_high=eta_bin_edges[bin['eta_bin'] + 1])
      step.selection = f'L1Tau_type == static_cast<int>(TauType::{bin["tau_type"]}) && ({selection})'
      step.tau_type = bin['tau_type']
      step.eta_bin = bin['eta_bin']
      step.pt_bin = pt_bin_idx
      step.bin_idx = bin_idx
      step.start_idx = batch_size
      step.stop_idx = batch_size + count
      step.count = count
      mix_steps.append(step)
      batch_size += count
  print(f'Number of mix steps: {len(mix_steps)}')
  print(f'Batch size: {batch_size}')
  step_stat = np.zeros(len(mix_steps))
  n_taus = { }
  n_taus_batch = { }
  for step_idx, step in enumerate(mix_steps):
    n_available = 0
    for input in step.inputs:
      input_path = os.path.join(cfg['spectrum_root'], f'{input}.root')
      file = ROOT.TFile.Open(input_path, "READ")
      hist_name = f'pt_eta_{step.tau_type}'
      hist = file.Get(hist_name)
      n_available += hist.GetBinContent(step.pt_bin + 1, step.eta_bin + 1)
    n_taus[step.tau_type] = n_available + n_taus.get(step.tau_type, 0)
    n_taus_batch[step.tau_type] = step.count + n_taus_batch.get(step.tau_type, 0)
    n_batches = math.floor(n_available / step.count)
    step_stat[step_idx] = n_batches
  step_idx = np.argmin(step_stat)
  step = mix_steps[step_idx]
  print(f'Total number of samples = {sum(n_taus.values())}: {n_taus}')
  n_taus_active = { name: x * step_stat[step_idx] for name, x in n_taus_batch.items()}
  print(f'Total number of used samples = {sum(n_taus_active.values())}: {n_taus_active}')
  n_taus_frac = { name: n_taus_active[name] / x for name, x in n_taus.items()}
  print(f'Used fraction: {n_taus_frac}')
  print(f'Number of samples per batch: {n_taus_batch}')
  n_taus_rel = { name: x / n_taus_batch['tau'] for name, x in n_taus_batch.items()}
  print(f'Relative number of samples: {n_taus_rel}')
  print('Step with minimum number of batches:')
  print(f'n_batches: {step_stat[step_idx]}')
  print(f'taus/batch: {step.count}')
  print(f'inputs: {step.input_setups}')
  print(f'eta bin: {step.eta_bin}')
  print(f'pt bin: {step.pt_bin}')
  print(f'bin idx: {step.bin_idx}')
  print(f'tau_type: {step.tau_type}')
  print(f'selection: {step.selection}')



if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make mix.')
  parser.add_argument('--cfg', required=True, type=str, help="Mix config file")
  args = parser.parse_args()

  analyse_mix(args.cfg)
