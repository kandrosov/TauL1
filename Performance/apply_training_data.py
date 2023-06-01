import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import uproot
from numba import njit
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import *

@njit
def fill_taus(tau_data, index, var_values):
  n_evt = tau_data.shape[0]
  for evt_idx in range(n_evt):
    for tau_idx in range(len(var_values[evt_idx])):
      tau_data[evt_idx, tau_idx, index] = var_values[evt_idx][tau_idx]

@njit
def fill_towers(tower_data, L1Tau_pt, L1TauTowers_tauIdx,
                L1Tau_tower_relEta, L1Tau_tower_relPhi,
                L1Tau_tower_hwEtEm, L1Tau_tower_hwEtHad):
  n_evt = tower_data.shape[0]
  n_eta = tower_data.shape[2]
  n_phi = tower_data.shape[3]
  for evt_idx in range(n_evt):
    for tau_idx in range(len(L1Tau_pt[evt_idx])):
      phi_ref = 1000
      eta_ref = 1000
      for tower_idx in range(len(L1TauTowers_tauIdx[evt_idx])):
        if L1TauTowers_tauIdx[evt_idx][tower_idx] == tau_idx:
          phi_ref = min(phi_ref, L1Tau_tower_relPhi[evt_idx][tower_idx])
          eta_ref = min(eta_ref, L1Tau_tower_relEta[evt_idx][tower_idx])
      for tower_idx in range(len(L1TauTowers_tauIdx[evt_idx])):
        if L1TauTowers_tauIdx[evt_idx][tower_idx] == tau_idx:
          eta = L1Tau_tower_relEta[evt_idx][tower_idx] - eta_ref
          phi = L1Tau_tower_relPhi[evt_idx][tower_idx] - phi_ref
          tower_data[evt_idx, tau_idx, eta, phi, 0] = L1Tau_tower_hwEtEm[evt_idx][tower_idx]
          tower_data[evt_idx, tau_idx, eta, phi, 1] = L1Tau_tower_hwEtHad[evt_idx][tower_idx]

tower_vars = [
  'L1TauTowers_tauIdx', 'L1TauTowers_relEta', 'L1TauTowers_relPhi', 'L1TauTowers_hwEtEm', 'L1TauTowers_hwEtHad',
]
all_vars = meta_vars_data + tau_vars_data + tower_vars


def get_inputs(values):
  n_evt = len(values['event'])
  n_phi = 9
  n_eta = 6
  max_n_tau = 12

  tau_data = np.zeros((n_evt, max_n_tau, len(tau_vars_data)))
  tower_data = np.zeros((n_evt, max_n_tau, n_eta, n_phi, 2))
  meta_data = np.zeros((n_evt, len(meta_vars_data)))

  for i, var in enumerate(tau_vars_data):
    fill_taus(tau_data, i, values[var])
  fill_towers(tower_data, values['L1Tau_pt'], values['L1TauTowers_tauIdx'], values['L1TauTowers_relEta'],
              values['L1TauTowers_relPhi'], values['L1TauTowers_hwEtEm'], values['L1TauTowers_hwEtHad'])
  for i, var in enumerate(meta_vars_data):
    meta_data[:, i] = values[var]

  return tau_data, tower_data, meta_data

def apply_training(dataset_path, model_path, output_file, batch_size):
  model = tf.keras.models.load_model(model_path)
  output_dir = os.path.dirname(output_file)
  if len(output_dir) > 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if os.path.exists(output_file):
    os.remove(output_file)

  print('Loading inputs')
  file = uproot.open(dataset_path)
  events = file['Events']
  with tqdm(total=events.num_entries) as pbar:
    for columns in events.iterate(all_vars, step_size=batch_size):
      tau_data, tower_data, meta_data = get_inputs(columns)
      n_evt = tau_data.shape[0]
      n_tau = tau_data.shape[1]
      tower_input = tf.reshape(tower_data, [ n_evt * n_tau, ] + list(tower_data.shape[2: ]))
      tau_pt = tf.reshape(tau_data[:, :, tau_vars_data.index('L1Tau_hwPt')], (n_evt * n_tau, 1))
      tau_eta = tf.reshape(tau_data[:, :, tau_vars_data.index('L1Tau_towerIEta')], (n_evt * n_tau, 1))
      tau_input = tf.concat([tau_pt, tau_eta], axis=1)
      pred = model((tower_input, tau_input))
      pred = tf.reshape(pred, (n_evt, n_tau))
      values = { }
      for tau_idx in range(n_tau):
        values[f'nn_score_{tau_idx}'] = pred[:, tau_idx].numpy().flatten()
        for var in ['L1Tau_pt', 'L1Tau_eta', 'L1Tau_hwIso']:
          values[f'{var}_{tau_idx}'] = tau_data[:, tau_idx, tau_vars_data.index(var)].flatten()
      for i, var in enumerate(meta_vars_data):
        values[var] = meta_data[:, i].flatten()

      pd_dataset = pd.DataFrame(values)
      pd_dataset.to_hdf(output_file, key='events', append=True, complevel=1, complib='zlib')
      pbar.update(n_evt)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--batch-size', required=False, type=int, default=2500)
  args = parser.parse_args()

  apply_training(args.dataset, args.model, args.output, args.batch_size)
