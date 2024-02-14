import os
import sys
from tqdm import tqdm
import uproot
from numba import njit
import awkward as ak
import numpy as np
import shutil
import yaml

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from TauL1.Performance.model_tools import load_model
from TauL1.hls4ml.load_model import load_hls4ml_model
from TauL1.Training.model import make_input_fn

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

def get_inputs(columns, input_fn):

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
  tau_data_tf = tf.convert_to_tensor(tau_data, dtype=tf.float32)
  tower_data_tf = tf.convert_to_tensor(tower_data, dtype=tf.float32)
  inputs = input_fn(tower_data_tf, tau_data_tf)
  return counts, inputs

def apply_training(dataset_path, model, cfg, output_file, batch_size, has_pt_node=False, is_hls4ml=False):
  output_dir = os.path.dirname(output_file)
  if len(output_dir) > 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
  output_file_tmp = output_file + '.tmp.root'
  if os.path.exists(output_file):
    os.remove(output_file)
  if os.path.exists(output_file_tmp):
    os.remove(output_file_tmp)

  input_fn = make_input_fn(cfg['setup']['reduce_calo_precision'], cfg['setup']['reduce_center_precision'],
                           cfg['setup']['apply_avg_pool'], cfg['setup']['concat_input'], to_train=False,
                           to_numpy=is_hls4ml)

  print('Loading inputs')

  file = uproot.open(dataset_path)
  events = file['Events']
  with uproot.recreate(output_file_tmp, compression=uproot.LZMA(9)) as out_file:
    with tqdm(total=events.num_entries) as pbar:
      for columns in events.iterate(all_vars, step_size=batch_size):
        n_evt = len(columns[tau_vars[0]])
        counts, inputs = get_inputs(columns, input_fn)
        if is_hls4ml:
          pred = model.predict(inputs)
          suffix = '_q'
        else:
          pred = model(inputs).numpy()
          suffix = ''

        n_finite = np.sum(np.isfinite(pred), dtype=int)
        n_tot = np.shape(pred)[0] * np.shape(pred)[1]
        n_nonfinite = n_tot - n_finite
        if n_nonfinite > 0:
          raise RuntimeError(f'{n_nonfinite} nonfinite values found in predictions')

        nn_tags = ak.unflatten(pred[:, 0], counts)
        if has_pt_node:
          pt_reg = ak.unflatten(pred[:, 1], counts)
        else:
          pt_reg = ak.zeros_like(nn_tags)

        data = {
          'L1Tau_NNtag' + suffix: nn_tags,
          'L1Tau_ptReg' + suffix: pt_reg,
        }

        if 'Events' in out_file:
          out_file['Events'].extend(data)
        else:
          out_file['Events'] = data

        pbar.update(n_evt)
  shutil.move(output_file_tmp, output_file)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--model', required=True, type=str)
  parser.add_argument('--model-cfg', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--batch-size', required=False, type=int, default=2500)
  parser.add_argument('--use-hls4ml', action='store_true')
  parser.add_argument('--has-pt-node', action='store_true')
  parser.add_argument('--hls4ml-model', required=False, type=str, default=None)
  parser.add_argument('--hls4ml-config', required=False, type=str, default=None)
  parser.add_argument('--fpga-part', required=False, type=str, default=None)

  args = parser.parse_args()

  if args.use_hls4ml:
    assert(args.hls4ml_model is not None)
    assert(args.hls4ml_config is not None)
    assert(args.fpga_part is not None)
    model = load_hls4ml_model(args.model, args.hls4ml_config, fpga_part=args.fpga_part, output_path=args.hls4ml_model,
                              compile=True)
  else:
    model = load_model(args.model)

  with open(args.model_cfg) as f:
    cfg = yaml.safe_load(f)

  apply_training(args.dataset, model, cfg, args.output, args.batch_size, args.has_pt_node, args.use_hls4ml)
