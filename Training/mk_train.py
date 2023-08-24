import uproot
import numpy as np
import awkward as ak
import os
import sys
import gc

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.dirname(file_dir)
  if base_dir not in sys.path:
    sys.path.append(base_dir)
  __package__ = os.path.split(file_dir)[-1]

from CommonDef import *

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

debug = False
stop_entry = None

def get_inputs_fake():
  n_tau = 500 * 50000
  n_phi = 9
  n_eta = 6

  x = np.random.rand(n_tau, n_eta, n_phi, 2)
  y = np.random.rand(n_tau, 1)
  w = np.random.rand(n_tau, 1)
  meta = np.random.rand(n_tau, len(meta_vars))

  return x, y, w, meta

def get_inputs():
  file = uproot.open('/data2/Run3_HLT/prod_v3_skim_v2.root')
  events = file['Events']
  print(events.keys())
  vars_to_load = list(set(all_vars) - set(ref_vars))
  values = events.arrays(vars_to_load, entry_stop=stop_entry)

  n_tau = len(values['L1Tau_type'])
  print(f'Loaded {n_tau} entries')
  print('Creating features')
  n_phi = 9
  n_eta = 6

  n_true_tau = 160
  batch_size = 500
  n_fake_tau = batch_size - n_true_tau
  weight_true = batch_size / (2 * n_true_tau)
  weight_fake = batch_size / (2 * n_fake_tau)

  x = np.zeros((n_tau, n_eta, n_phi, 2))
  y = np.zeros((n_tau, 1))
  w = np.ones((n_tau, 1))
  meta = np.zeros((n_tau, len(meta_vars)))
  y[:, 0] = values['L1Tau_type'] == TauType.tau
  w[values['L1Tau_type'] == TauType.tau] = weight_true
  w[values['L1Tau_type'] != TauType.tau] = weight_fake

  ref_dict = {
    'phi_ref': ak.min(values['L1Tau_tower_relPhi'], axis=1),
    'eta_ref': ak.min(values['L1Tau_tower_relEta'], axis=1),
  }

  for i, var in enumerate(meta_vars):
    if var in ref_vars:
      meta[:, i] = ref_dict[var]
    else:
      meta[:, i] = values[var]

  phi_ref = ref_dict['phi_ref']
  eta_ref = ref_dict['eta_ref']

  for phi in range(n_phi):
    for eta in range(n_eta):
      print(f'Creating grid features: phi={phi+1}/{n_phi} eta={eta+1}/{n_eta}')
      sel = (values['L1Tau_tower_relPhi'] - phi_ref == phi) & (values['L1Tau_tower_relEta'] - eta_ref == eta)
      x[:, eta, phi, 0] = np.reshape(values['L1Tau_tower_hwEtEm'][sel], (n_tau, ))
      x[:, eta, phi, 1] = np.reshape(values['L1Tau_tower_hwEtHad'][sel], (n_tau, ))
  return x, y, w, meta

print('Loading inputs')
if debug:
  x, y, w, meta = get_inputs_fake()
else:
  x, y, w, meta = get_inputs()
print('Collecting garbage')
gc.collect()
print('Creating dataset')
dataset = tf.data.Dataset.from_tensor_slices((x, y, w, meta))
print('Saving dataset')
dataset.save(f'skim_v2', compression='GZIP')
print('All done')
