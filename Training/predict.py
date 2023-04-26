import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve
import numpy as np
from CommonDef import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

model = keras.models.load_model('models/model_v1')

input_idx = 1
dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{input_idx}', compression='GZIP')
#dataset = tf.data.Dataset.load(f'taus_{input_idx}', compression='GZIP')

ds_pred = dataset.batch(300).take(100).map(to_pred)
pred = model.predict(ds_pred)


y = np.concatenate(list(dataset.batch(300).map(to_gen).take(100).as_numpy_iterator()))
fpr, tpr, threasholds = roc_curve(y, pred)


hwIso = np.concatenate(list(dataset.batch(300).map(to_hwIso).take(100).as_numpy_iterator()))
hw_fpr, hw_tpr, hw_threasholds = roc_curve(y, hwIso)
print(f'hw_fpr = {hw_fpr} \nhw_tpr={hw_tpr} \nhw_thresholds={hw_threasholds}')
for fpr_idx in range(0, len(fpr)):
  if(abs(fpr[fpr_idx]-hw_fpr[1])<0.001):
    print(abs(fpr[fpr_idx]-hw_fpr[1]), fpr[fpr_idx], hw_fpr[1], threasholds[fpr_idx])
#for tpr_idx in range(0, len(tpr)):
#  if(tpr[tpr_idx]-hw_tpr[1]<0.00000001):
#    print(tpr[tpr_idx], hw_tpr[1], threasholds[tpr_idx])

with PdfPages('plots/roc_v1.pdf') as pdf:
  fig, ax = plt.subplots(1, 1, figsize=(7, 7))
  ax.plot(tpr, fpr, )
  ax.errorbar(hw_tpr, hw_fpr, fmt='o', markersize=2, color='red')
  ax.set_xlabel('true tau_h efficiency')
  ax.set_ylabel('jet->tau efficiency')
  plt.subplots_adjust(hspace=0)
  pdf.savefig(fig, bbox_inches='tight')
