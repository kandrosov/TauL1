import uproot
import numpy as np
import awkward as ak
import os
import gc
from CommonDef import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf


input_idx = 0
stop_entry = None

def get_inputs():
  file = uproot.open(f'skim_v1/taus_{input_idx}.root')
  events = file['Events']
  print(events.keys())
  values = events.arrays(all_vars, entry_stop=stop_entry)

  n_tau = len(values['L1Tau_type'])
  n_phi = 9
  n_eta = 6

  n_true_tau = 87
  batch_size = 300
  weight = n_true_tau / (batch_size - n_true_tau)

  x = np.zeros((n_tau, n_eta, n_phi, 5))
  y = np.zeros((n_tau, 1))
  w = np.ones((n_tau, 1))
  meta = np.zeros((n_tau, len(meta_vars)))
  y[:, 0] = values['L1Tau_type'] == TauType.tau
  w[values['L1Tau_type'] != TauType.tau] = weight
  for i, var in enumerate(meta_vars):
    meta[:, i] = values[var]

  x[:, :, :, 0] = np.reshape(values['L1Tau_hwPt'], (n_tau, 1, 1))
  x[:, :, :, 1] = np.reshape(values['L1Tau_towerIEta'], (n_tau, 1, 1))
  phi_ref = ak.min(values['L1Tau_tower_relPhi'], axis=1)
  eta_ref = ak.min(values['L1Tau_tower_relEta'], axis=1)
  for phi in range(n_phi):
    for eta in range(n_eta):
      sel = (values['L1Tau_tower_relPhi'] - phi_ref == phi) & (values['L1Tau_tower_relEta'] - eta_ref == eta)
      x[:, eta, phi, 2] = np.reshape(values['L1Tau_tower_hwEtEm'][sel], (n_tau, ))
      x[:, eta, phi, 3] = np.reshape(values['L1Tau_tower_hwEtHad'][sel], (n_tau, ))
      x[:, eta, phi, 4] = np.reshape(values['L1Tau_tower_hwPt'][sel], (n_tau, ))
  return x, y, w, meta

print('Loading inputs')
x, y, w, meta = get_inputs()
print('Collecting garbage')
gc.collect()
print('Creating dataset')
dataset = tf.data.Dataset.from_tensor_slices((x, y, w, meta))
print('Saving dataset')
dataset.save(f'taus_{input_idx}', compression='GZIP')
print('All done')
