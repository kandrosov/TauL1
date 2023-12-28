import os
import sys

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

def make_eval(dataset_path, output_file, is_data, event_range=None):
  output_dir = os.path.dirname(output_file)
  if len(output_dir) > 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
  if os.path.exists(output_file):
    os.remove(output_file)

  print('Loading inputs')
  dataset_path = dataset_path if dataset_path.endswith('.root') else os.path.join(dataset_path, '*.root')
  df_in = ROOT.RDataFrame('Events', dataset_path)
  if event_range is not None:
    df_in = df_in.Range(0, event_range)
  df_in = ApplyCommonDefinitions(df_in, isData=is_data)

  if not is_data:
    df_in = df_in.Define('nGenLep', 'genLeptons.size()')
    n_tot = df_in.Count()
    df_in = df_in.Filter('nGenLep == 2')
    n_2 = df_in.Count()
    for lep_idx in range(2):
      df_in = df_in.Filter(f'''
        genLeptons[{lep_idx}].visibleP4().pt() > 20 &&
        abs(genLeptons[{lep_idx}].visibleP4().eta()) < 2.1 &&
        genLeptons[{lep_idx}].kind() == reco_tau::gen_truth::GenLepton::Kind::TauDecayedToHadrons''')
    n_2_sel = df_in.Count()
    df_in = df_in.Define('htt_indices', 'FindParticles(25, {15, -15}, GenPart_pdgId, GenPart_genPartIdxMother)') \
                 .Define('htt_idx', 'htt_indices.size() == 1 ? *htt_indices.begin() : -1') \
                 .Define('gen_htt_p4', 'htt_idx >= 0 ? GenPart_p4[htt_idx] : LorentzVectorM()') \
                 .Define('GenHtt_pt', 'htt_idx >= 0 ? GenPart_pt[htt_idx] : -1.f') \
                 .Define('hbb_indices', 'FindParticles(25, {5, -5}, GenPart_pdgId, GenPart_genPartIdxMother)') \
                 .Define('hbb_idx', 'hbb_indices.size() == 1 ? *hbb_indices.begin() : -1') \
                 .Define('gen_hbb_p4', 'hbb_idx >= 0 ? GenPart_p4[hbb_idx] : LorentzVectorM()') \
                 .Define('GenHbb_pt', 'hbb_idx >= 0 ? GenPart_pt[hbb_idx] : -1.f') \
                 .Define('gen_hh_p4', 'gen_htt_p4 + gen_hbb_p4') \
                 .Define('GenHH_pt', 'htt_idx >= 0 && hbb_idx >= 0 ? static_cast<float>(gen_hh_p4.pt()) : -1.f') \
                 .Define('GenHH_mass', 'htt_idx >= 0 && hbb_idx >= 0 ? static_cast<float>(gen_hh_p4.mass()) : -1.f') \
                 .Define('gen_htt_vis_p4', 'genLeptons[0].visibleP4() + genLeptons[1].visibleP4()') \
                 .Define('GenHtt_vis_pt', 'static_cast<float>(gen_htt_vis_p4.pt())') \
                 .Define('GenHtt_vis_mass', 'static_cast<float>(gen_htt_vis_p4.mass())')
    n_with_htt = df_in.Filter('htt_idx >= 0').Count()
    n_with_hh = df_in.Filter('htt_idx >= 0 && hbb_idx >= 0').Count()


  l1tau_columns = [
    'pt', 'eta', 'phi', 'hwEtSum', 'hwEta', 'hwIso', 'hwPhi', 'hwPt', 'isoEt', 'nTT', 'rawEt', 'towerIEta',
    'towerIPhi', 'type' ]
  if not is_data:
    l1tau_columns.extend([ 'gen_pt', 'gen_eta', 'gen_phi', 'gen_mass', 'gen_charge', 'gen_partonFlavour' ])

  l1tautower_columns = [
    'nL1TauTowers', 'L1TauTowers_hwEtEm', 'L1TauTowers_hwEtHad', 'L1TauTowers_hwPt', 'L1TauTowers_relEta',
    'L1TauTowers_relPhi', 'L1TauTowers_tauIdx'
  ]

  l1_bits = [ 'L1_HTT120er', 'L1_HTT160er', 'L1_HTT200er', 'L1_HTT255er', 'L1_HTT280er', 'L1_HTT320er', 'L1_HTT360er', 'L1_HTT400er', 'L1_HTT450er' ]

  gen_higgs_columns = [ 'GenHtt_pt', 'GenHbb_pt', 'GenHH_pt', 'GenHH_mass', 'GenHtt_vis_pt', 'GenHtt_vis_mass' ]

  df_in = df_in.Define(f'L1Tau_nPV', 'RVecI(L1Tau_pt.size(), nPFPrimaryVertex)')
  df_in = df_in.Define(f'L1Tau_event', 'RVecI(L1Tau_pt.size(), static_cast<int>(event))')
  df_in = df_in.Define(f'L1Tau_luminosityBlock', 'RVecI(L1Tau_pt.size(), static_cast<int>(luminosityBlock))')
  df_in = df_in.Define(f'L1Tau_run', 'RVecI(L1Tau_pt.size(), static_cast<int>(run))')
  event_columns = [ 'nPFPrimaryVertex', 'event', 'luminosityBlock', 'run' ]
  other_columns = set()

  tau_columns = [ 'Tau_pt', 'Tau_eta', 'Tau_phi', 'Tau_mass', 'Tau_deepTauVSjet' ]
  for c in tau_columns:
    df_in = df_in.Define(f'L1Tau_{c}', f'GetVar(L1Tau_tauIdx, {c}, -1.f)')
    other_columns.add(c)

  df_in = df_in.Define('Jet_PNet_probtauh', 'Jet_PNet_probtauhm + Jet_PNet_probtauhp')
  jet_columns = [ 'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_PNet_probtauh', 'Jet_PNet_ptcorr' ]
  for c in jet_columns:
    df_in = df_in.Define(f'L1Tau_{c}', f'GetVar(L1Tau_jetIdx, {c}, -1.f)')
    other_columns.add(c)

  l1tau_columns.extend(other_columns)

  columns_in = [ 'nL1Tau' ]
  for col in l1tau_columns:
    columns_in.append(f'L1Tau_{col}')
  columns_in.extend(event_columns)
  columns_in.extend(l1tautower_columns)
  columns_in.extend(l1_bits)

  if not is_data:
    columns_in.extend(gen_higgs_columns)

  columns_in_v = ListToVector(columns_in)

  opt = ROOT.RDF.RSnapshotOptions()
  opt.fCompressionAlgorithm = ROOT.ROOT.kLZMA
  opt.fCompressionLevel = 9
  opt.fMode = 'RECREATE'
  df_in.Snapshot('Events', output_file, columns_in_v, opt)
  if not is_data:
    print(f'Total {n_tot.GetValue()} events, {n_2.GetValue()} with 2 gen leptons,'
          f' {n_2_sel.GetValue()} with 2 gen tauh with visible pt > 20 && |eta| < 2.1,'
          f' {n_with_htt.GetValue()} with H->tautau,'
          f' {n_with_hh.GetValue()} with H->tautau and H->bb')


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Make evaluation dataset.')
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--is-data', action='store_true')
  parser.add_argument('--range', required=False, type=int, default=None)
  args = parser.parse_args()

  PrepareRootEnv()
  make_eval(args.dataset, args.output, args.is_data, args.range)
