import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

model = keras.models.load_model('models/model_v1')

input_idx = 1
dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{input_idx}', compression='GZIP')
#dataset = tf.data.Dataset.load(f'taus_{input_idx}', compression='GZIP')
def to_pred(x, y, w, meta):
  return x[:276, :, :, :4]

ds_pred = dataset.batch(300).take(100).map(to_pred)
pred = model.predict(ds_pred)

def to_gen(x, y, w, meta):
  return y[:276]

y = np.concatenate(list(dataset.batch(300).map(to_gen).take(100).as_numpy_iterator()))
fpr, tpr, threasholds = roc_curve(y, pred)

def to_hwIso(x, y, w, meta):
  return meta[:276, get_index('L1Tau_hwIso')]

hwIso = np.concatenate(list(dataset.batch(300).map(to_hwIso).take(100).as_numpy_iterator()))
hw_fpr, hw_tpr, hw_threasholds = roc_curve(y, hwIso)

with PdfPages('roc_v2.pdf') as pdf:
  fig, ax = plt.subplots(1, 1, figsize=(7, 7))
  ax.plot(tpr, fpr, )
  ax.errorbar(hw_tpr, hw_fpr, fmt='o', markersize=2, color='red')
  ax.set_xlabel('true tau_h efficiency')
  ax.set_ylabel('jet->tau efficiency')
  plt.subplots_adjust(hspace=0)
  pdf.savefig(fig, bbox_inches='tight')
