import os
import tensorflow as tf
from tensorflow import keras
import numpy as np


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

def to_pred(x, y, w, meta):
    return x[:276, :, :, :4]

def get_x_var(x, y, w, meta):
    return meta[:276, get_index(var)]

def get_y_info(x,y,w,meta):
    return y[:276]

def get_hw_info(x,y,w,meta):
    return meta[:276, 'L1Tau_hwIso']

def add_prediction(input_idx,var,x_bins, model='model_v1'):
	model = keras.models.load_model(f'models/{model}')
	dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{input_idx}', compression='GZIP')
	ds_pred = dataset.batch(300).map(to_pred)
	pred = model.predict(ds_pred)
	var_den_presel = np.concatenate(list(dataset.batch(300).map(get_x_var).as_numpy_iterator()))
	gen_truth = np.concatenate(list(dataset.batch(300).map(get_y_info).as_numpy_iterator()))
	hw_iso = np.concatenate(list(dataset.batch(300).map(get_hw_info).as_numpy_iterator()))
	all_var = np.vstack((var_den_presel[:], pred[:,0], gen_truth[:,0], hw_iso[:, 0])).T
	condition1 = all_var[:,2]==1
	condition2 = all_var[:,1]>0.40430108# 0.6482158
	condition3 = all_var[:,3]==1
	condition = condition1 & condition2
	condition_iso = condition1 & condition3
	var_den = all_var[condition1][:,0]
	var_num = all_var[condition][:,0]
	var_num_iso = all_car[condition_iso][:,0]
	print(var_num.min(), var_num.max())
	print(f"len of {var} num = {len(var_num)}, len of {var} den = {len(var_den)}")
	import matplotlib.pyplot as plt
	val_of_bins_num_iso, edges_of_bins_num_iso, patches_num_iso = plt.hist(var_num_iso, x_bins, range=(0,250), histtype='step', label="num")
	val_of_bins_num, edges_of_bins_num, patches_num = plt.hist(var_num, x_bins, range=(0,250), histtype='step', label="num")
	val_of_bins_den, edges_of_bins_den, patches_den = plt.hist(var_den, x_bins, range=(0,250), histtype='step', label="den")
	print("bins:", edges_of_bins_num)
	ratio = np.divide(val_of_bins_num,
		          val_of_bins_den,
		          where=(val_of_bins_den != 0))
	ratio_iso = np.divide(val_of_bins_num_iso,
		          val_of_bins_den,
		          where=(val_of_bins_den != 0))

	print("ratio:", ratio)
	print("ratio_iso:", ratio_iso)

	fig = plt.figure(figsize=(10.,6.))
	error = np.divide(val_of_bins_num * np.sqrt(val_of_bins_den) + val_of_bins_den * np.sqrt(val_of_bins_num),
			          np.power(val_of_bins_den, 2),
			          where=(val_of_bins_den != 0))

	error_iso = np.divide(val_of_bins_num_iso * np.sqrt(val_of_bins_den) + val_of_bins_den * np.sqrt(val_of_bins_num_iso),
			          np.power(val_of_bins_den, 2),
			          where=(val_of_bins_den != 0))
	#print("error:", error)
	# --- efficiency VS variable
	plt.ylabel('efficiency')
	plt.xlabel(f'{var}')

	bincenter = 0.5 * (edges_of_bins_num[1:] + edges_of_bins_num[:-1])
	bincenter_iso = 0.5 * (edges_of_bins_num_iso[1:] + edges_of_bins_num_iso[:-1])
	plt.errorbar(bincenter, ratio, yerr=error, fmt='.', color='b')
	plt.errorbar(bincenter_iso, ratio_iso, yerr=error, fmt='.', color='r')


	plt.savefig(f"plots/{var}_efficiency_1.png")



var = 'L1Tau_gen_pt'
x_bins=[0,40,60,80,100,150,200,250,300,500]
add_prediction(2, var, x_bins)
