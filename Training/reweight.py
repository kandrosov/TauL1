import numpy as np
import tensorflow as tf
import math 
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

all_vars = event_vars + gen_vars + reco_vars + hw_vars + tower_vars
meta_vars = event_vars + gen_vars + reco_vars + hw_vars

def get_index(name):
  return meta_vars.index(name)

def get_weight(x,y,w,meta):
    return w[:276]

def get_gen_pt(x,y,w,meta):
    return meta[:276, get_index('L1Tau_gen_pt')]-20

def reweight(x,y,w,meta):
    a = 5.5
    b = 1.5
    gen_pt0 = 20
    w = w *( a * tf.experimental.numpy.log10(meta[:,get_index('L1Tau_gen_pt')] / gen_pt0 ) + b)
    #w = w *( a * (meta[:,get_index('L1Tau_gen_pt')] - gen_pt0 ) + b)
    return w[:]
input_idx = 2
dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{input_idx}', compression='GZIP')
old_weight = np.concatenate(list(dataset.batch(300).map(get_weight).take(10).as_numpy_iterator()))
new_weight = np.concatenate(list(dataset.batch(300).map(reweight).take(10).as_numpy_iterator()))
gen_pt = np.concatenate(list(dataset.batch(300).map(get_gen_pt).take(10).as_numpy_iterator()))
print(old_weight, gen_pt, new_weight)
