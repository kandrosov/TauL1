import os
import sys
from tqdm import tqdm
import uproot
from numba import njit
import awkward as ak
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

@njit
def fill_taus(tau_data, index, var_values):
  n_evt = len(var_values)
  flat_tau_idx = 0
  for evt_idx in range(n_evt):
    for tau_idx in range(len(var_values[evt_idx])):
      tau_data[flat_tau_idx, index] = var_values[evt_idx][tau_idx]
      flat_tau_idx += 1

@njit
def fill_towers(tower_data, counts, L1TauTowers_tauIdx,
                L1Tau_tower_relEta, L1Tau_tower_relPhi,
                L1Tau_tower_hwEtEm, L1Tau_tower_hwEtHad):
  n_evt = len(L1TauTowers_tauIdx)
  flat_tau_idx = 0
  for evt_idx in range(n_evt):
    for tau_idx in range(counts[evt_idx]):
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
          tower_data[flat_tau_idx, eta, phi, 0] = L1Tau_tower_hwEtEm[evt_idx][tower_idx]
          tower_data[flat_tau_idx, eta, phi, 1] = L1Tau_tower_hwEtHad[evt_idx][tower_idx]
      flat_tau_idx += 1

tower_vars = [
  'L1TauTowers_tauIdx', 'L1TauTowers_relEta', 'L1TauTowers_relPhi', 'L1TauTowers_hwEtEm', 'L1TauTowers_hwEtHad',
]

tau_vars = [ 'L1Tau_hwPt', 'L1Tau_towerIEta' ]

all_vars = tau_vars + tower_vars

def fn(y_true, y_pred):
  pass


def get_inputs(columns):

  ref_column = columns[tau_vars[0]]
  n_phi = 9
  n_eta = 6
  n_tau = len(ak.flatten(ref_column))

  counts = ak.num(ref_column)
  tau_data = np.zeros((n_tau, len(tau_vars)))
  tower_data = np.zeros((n_tau, n_eta, n_phi, 2))

  for i, var in enumerate(tau_vars):
    fill_taus(tau_data, i, columns[var])
  fill_towers(tower_data, counts, columns['L1TauTowers_tauIdx'], columns['L1TauTowers_relEta'],
              columns['L1TauTowers_relPhi'], columns['L1TauTowers_hwEtEm'], columns['L1TauTowers_hwEtHad'])

  return counts, tau_data, tower_data

def apply_training(dataset_path, model_path, output_file, batch_size):
  custom_objects = [ 'l1tau_loss', 'id_loss', 'pt_loss', 'id_acc' ]
  custom_objects = { name: fn for name in custom_objects }
  model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
  output_dir = os.path.dirname(output_file)
  if len(output_dir) > 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if os.path.exists(output_file):
    os.remove(output_file)

  print('Loading inputs')

  file = uproot.open(dataset_path)
  events = file['Events']
  with uproot.recreate(output_file, compression=uproot.LZMA(9)) as out_file:
    with tqdm(total=events.num_entries) as pbar:
      for columns in events.iterate(all_vars, step_size=batch_size):
        n_evt = len(columns[tau_vars[0]])
        counts, tau_data, tower_data = get_inputs(columns)
        pred = model((tower_data, tau_data)).numpy()
        data = {
          'L1Tau_NNtag': ak.unflatten(pred[:, 0], counts),
          'L1Tau_ptReg': ak.unflatten(pred[:, 1], counts),
        }

        if 'Events' in out_file:
          out_file['Events'].extend(data)
        else:
          out_file['Events'] = data

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
