import numpy as np
import tensorflow as tf
import sys
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

def get_meta_var(name):
  return meta_vars[:,meta_vars.index(name)]

def get_index(name):
  return meta_vars.index(name)

def get_weight(x,y,w,meta):
    return w[:276]

def get_gen_pt(x,y,w,meta):
    return 5.5*tf.experimental.numpy.log10(meta[:276, get_index('L1Tau_gen_pt')]/20)+1.5
   
def reweight(x,y,w,meta):
    a = 5.5
    b = 1.5
    gen_pt0 = 20
    print(f"w shape is {w.shape}")
    print(f"meta shape is {meta.shape}")
    #w = w *( a * (meta[:,get_index('L1Tau_gen_pt')] - gen_pt0 ) + b)
    w = w*( a * tf.experimental.numpy.log10( meta[:,get_index('L1Tau_gen_pt')]/20) + b)
    return w[:276] 
    

def get_weight_shape(x,y,w,meta):
    print(f"w shape is {w.shape}")
    return w

input_idx = 2
dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{input_idx}', compression='GZIP')
old_weight = np.concatenate(list(dataset.batch(300).map(get_weight).take(10).as_numpy_iterator()))
old_weight = np.concatenate(list(dataset.batch(300).map(get_weight_shape).take(10).as_numpy_iterator()))

new_weight = np.concatenate(list(dataset.batch(300).map(reweight).take(10).as_numpy_iterator()))
print(type(new_weight), new_weight.shape)
print("old_weight shape")
print(type(old_weight), old_weight.shape)

#gen_pt = np.concatenate(list(dataset.batch(300).map(get_gen_pt).take(10).as_numpy_iterator()))
##new_weights=tf.reshape(new_weight, old_weight.shape)
#j=0
#for k,z,g in zip(old_weight, new_weight,gen_pt):
#    if j>10: break
#    print(f"old weight = {k} \t gen_pt = {g} \t new_weight = {z}")
#    j+=1
###print(old_weight.shape, new_weight.shape, gen_pt.shape, new_weights.shape)
##print(f"old weight shape = {old_weight.shape}, new weight shape = {new_weight.shape}, gen pt shape = {gen_pt.shape}")
#
#

