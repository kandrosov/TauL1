import os
import shutil
import sys
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

def make_mix(cfg_file, output, input_range):
  with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)
  if os.path.exists(output):
    shutil.rmtree(output)
  os.makedirs(output, exist_ok=True)
  mix_steps = []
  pt_bin_edges = cfg['bin_edges']['pt']
  eta_bin_edges = cfg['bin_edges']['eta']
  batch_size = 0
  for bin in cfg['bins']:
    if len(bin['counts']) != len(pt_bin_edges) - 1:
      raise ValueError("Number of counts does not match number of pt bins")
    for pt_bin_idx, count in enumerate(bin['counts']):
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
      step.start_idx = batch_size
      step.stop_idx = batch_size + count
      mix_steps.append(step)
      batch_size += count
  print(f'Number of mix steps: {len(mix_steps)}')
  print(f'Batch size: {batch_size}')
  for step_idx, step in enumerate(mix_steps):
    print(f'{timestamp_str()}{step_idx}/{len(mix_steps)}: processing...')
    input_files = []
    for input in step.inputs:
      input_path = os.path.join(cfg['input_root'], input)
      input_files.extend(MakeFileList(input_path))
    input_files_vec = ListToVector(input_files, "string")
    df_in = ROOT.RDataFrame(cfg['tree_name'], input_files_vec)
    if 'event_sel' in cfg:
      df_in = df_in.Filter(cfg['event_sel'])
    df_in = ApplyCommonDefinitions(df_in)
    df_in = df_in.Define('L1Tau_sel', step.selection)
    df_in = df_in.Define('L1Tau_pt_sel', 'L1Tau_pt[L1Tau_sel]')
    df_in = df_in.Define('L1Tau_eta_sel', 'L1Tau_eta[L1Tau_sel]')
    df_in = df_in.Define('L1Tau_phi_sel', 'L1Tau_phi[L1Tau_sel]')

    if step_idx == 0:
      nTau_out = batch_size * cfg['n_batches']
      df_out = ROOT.RDataFrame(nTau_out)
    else:
      output_prev_step = os.path.join(output, f'step_{step_idx - 1}.root')
      df_out = ROOT.RDataFrame(cfg['tree_name'], output_prev_step)

    columns_in = [ 'L1Tau_pt_sel', 'L1Tau_eta_sel', 'L1Tau_phi_sel']
    columns_out = [ 'L1Tau_pt', 'L1Tau_eta', 'L1Tau_phi' ]
    columns_in_v = ListToVector(columns_in)
    columns_out_v = ListToVector(columns_out)

    column_types = [ str(df_in.GetColumnType(c)) for c in columns_in ]
    nTau_in = (step.stop_idx - step.start_idx) * cfg['n_batches']
    print(f'nTaus = {nTau_in}')
    print(f'inputs: {step.input_setups}')
    print(f'selection: {step.selection}')
    #df_in = df_in.Range(0, nTau_in)
    tuple_maker = ROOT.analysis.TupleMaker(*column_types)(100, nTau_in)
    df_out = tuple_maker.process(ROOT.RDF.AsRNode(df_in), ROOT.RDF.AsRNode(df_out), columns_in_v,
                                 step.start_idx, step.stop_idx, batch_size)
    default_value = 0
    for column_idx in range(len(columns_in)):
      if step_idx == 0:
        df_out = df_out.Define(columns_out[column_idx],
                               f'_entry.valid ? _entry.float_values.at({column_idx}) : {default_value}')
      else:
        df_out = df_out.Redefine(columns_out[column_idx],
                                 f'_entry.valid ? _entry.float_values.at({column_idx}) : {columns_out[column_idx]}')
    opt = ROOT.RDF.RSnapshotOptions()
    opt.fCompressionAlgorithm = ROOT.ROOT.kLZMA
    opt.fCompressionLevel = 9
    opt.fMode = 'RECREATE'
    output_step = os.path.join(output, f'step_{step_idx}.root')
    df_out.Snapshot(cfg['tree_name'], output_step, columns_out_v, opt)
    tuple_maker.join()
    print(f'{timestamp_str()}{step_idx}/{len(mix_steps)}: done.')

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make mix.')
  parser.add_argument('--cfg', required=True, type=str, help="Mix config file")
  parser.add_argument('--output', required=True, type=str, help="output directory")
  parser.add_argument('--input-range', required=False, type=str, default=None,
                      help="[begin, end) range for the inputs expressed in the fraction of the total events")
  args = parser.parse_args()

  PrepareRootEnv()
  input_range = [ float(x) for x in args.input_range.split(',') ] if args.input_range else None
  make_mix(args.cfg, args.output, input_range)
