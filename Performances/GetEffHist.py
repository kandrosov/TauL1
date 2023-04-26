import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import statsmodels.stats.proportion as ssp
import argparse
from CommonDef import *


def get_x_var(x, y, w, meta):
    return meta[:276, get_index(args.var)]


class TauType:
  ele = 0
  tau = 2
  jet = 3



def add_prediction(dataset,pred, var ,x_bins ,thr,required_type='tau'):
	var_den_presel = np.concatenate(list(dataset.batch(300).map(get_x_var).as_numpy_iterator()))
	gen_truth = np.concatenate(list(dataset.batch(300).map(get_y_info).as_numpy_iterator()))
	tau_type = np.concatenate(list(dataset.batch(300).map(get_tauType_info).as_numpy_iterator()))
	hw_iso = np.concatenate(list(dataset.batch(300).map(to_hwIso).as_numpy_iterator()))
	#print(hw_iso.shape,tau_type.shape)
	all_var = np.vstack((var_den_presel[:], pred[:,0], gen_truth[:,0], hw_iso[:], tau_type[:])).T
	print(len(all_var[all_var[:,4]==2]))
	condition_type = all_var[:,4]==getattr(TauType, required_type)
	condition_gen_truth = all_var[:,2]==1
	condition_thr = all_var[:,1]> thr #0.40430108# 0.6482158
	condition_hwIso = all_var[:,3]==1 if required_type=='tau'else all_var[:,3]==0
	condition_nn_based =  condition_type & condition_thr
	condition_cut_based = condition_type & condition_hwIso
	var_den = all_var[condition_type][:,0]
	var_num = all_var[condition_nn_based][:,0]
	var_num_iso = all_var[condition_cut_based][:,0]
	#print(var_num.min(), var_num.max())
	print(f"len of {var} num iso = {len(var_num_iso)}, len of {var} den = {len(var_den)}")
	print(f"len of {var} num = {len(var_num)}, len of {var} den = {len(var_den)}")

	import matplotlib.pyplot as plt

	val_of_bins_num_iso, edges_of_bins_num_iso, patches_num_iso = plt.hist(var_num_iso, x_bins, range=(var_num.min(),x_bin.max()), histtype='step', label="num_iso")
	val_of_bins_num, edges_of_bins_num, patches_num = plt.hist(var_num, x_bins, range=(var_num.min(),x_bin.max()), histtype='step', label="num")
	#print("a")
	val_of_bins_den, edges_of_bins_den, patches_den = plt.hist(var_den, x_bins, range=(var_num.min(),x_bin.max()), histtype='step', label="den")

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
	error_u,error_d = ssp.proportion_confint(val_of_bins_num,val_of_bins_den,alpha=1-0.68,method='beta')
	error_iso_u,error_iso_d = ssp.proportion_confint(val_of_bins_num_iso,val_of_bins_den,alpha=1-0.68,method='beta')

	#print("error:", error)
	# --- efficiency VS variable
	plt.ylabel('efficiency')
	plt.xlabel(f'{var}')

	bincenter = 0.5 * (edges_of_bins_num[1:] + edges_of_bins_num[:-1])
	bincenter_iso = 0.5 * (edges_of_bins_num_iso[1:] + edges_of_bins_num_iso[:-1])
	plt.errorbar(bincenter, ratio, uplims=error_u,lolims=error_d, color='b', markersize=8, marker='o', linestyle='none')
	plt.errorbar(bincenter_iso, ratio_iso, uplims=error_iso_u,lolims=error_iso_d, fmt='.', color='r', markersize=8, marker='o', linestyle='none')


	plt.savefig(f"plots/{var}_efficiency_{tau_type}.png")




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=int, default = 2)
parser.add_argument('--threshold', required=False, type=float, default = 0.40430108)
parser.add_argument('--model_version', required=False, type=int, default = 8)
parser.add_argument('--var', required=False, type=str, default = 'L1Tau_gen_pt')
args = parser.parse_args()

#var = 'L1Tau_gen_pt'
x_bins=[0,40,60,80,100,150,200,250]

model = keras.models.load_model(f'models/model_v{args.model_version}')
dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{args.dataset}', compression='GZIP')
ds_pred = dataset.batch(300).map(to_pred)
pred = model.predict(ds_pred)
print(pred.shape)
tau_type='tau'
add_prediction(dataset,pred, args.var ,x_bins ,args.threshold,tau_type)
'''
for tau_type in ['e', 'jet', 'tau']:
    add_prediction(dataset,pred, args.var ,x_bins ,args.threshold,tau_type)
'''
