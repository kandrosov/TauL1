
class TauType:
  e = 0
  tau = 2
  jet = 3
event_vars = [
  'run', 'luminosityBlock', 'event', 'nPV', 'step_idx'
]

gen_vars = [
  'L1Tau_type', 'L1Tau_gen_pt', 'L1Tau_gen_eta', 'L1Tau_gen_phi', 'L1Tau_gen_mass',
  'L1Tau_gen_charge', 'L1Tau_gen_partonFlavour'
]

reco_vars = [
  'L1Tau_pt', 'L1Tau_eta', 'L1Tau_phi', 'L1Tau_hwIso', 'L1Tau_isoEt', 'L1Tau_nTT', 'L1Tau_rawEt',
]

hw_vars = [
  'L1Tau_hwPt', 'L1Tau_hwEta', 'L1Tau_hwPhi', 'L1Tau_towerIEta', 'L1Tau_towerIPhi', 'L1Tau_hwEtSum'
]

tower_vars = [
  'L1Tau_tower_relEta', 'L1Tau_tower_relPhi', 'L1Tau_tower_hwEtEm', 'L1Tau_tower_hwEtHad', 'L1Tau_tower_hwPt',
]

pftau_vars = [
  'Tau_pt', 'Tau_eta', 'Tau_phi', 'Tau_mass', 'Tau_deepTauVSjet'
]

pfjet_vars = [
  'Jet_pt', 'Jet_eta', 'Jet_phi', 'Jet_mass', 'Jet_PNet_probtauh', 'Jet_PNet_ptcorr'
]

ref_vars = [ 'phi_ref', 'eta_ref' ]

meta_vars = event_vars + gen_vars + reco_vars + hw_vars + pftau_vars + pfjet_vars + ref_vars
all_vars = meta_vars + tower_vars

meta_vars_data = [
  'run', 'luminosityBlock', 'event', 'nPFPrimaryVertex'
]

tau_vars_data = [
  'L1Tau_pt', 'L1Tau_eta', 'L1Tau_phi', 'L1Tau_hwIso', 'L1Tau_isoEt', 'L1Tau_nTT', 'L1Tau_rawEt',
  'L1Tau_hwPt', 'L1Tau_hwEta', 'L1Tau_hwPhi', 'L1Tau_towerIEta', 'L1Tau_towerIPhi', 'L1Tau_hwEtSum'
]

def get_index(name):
  return meta_vars.index(name)
