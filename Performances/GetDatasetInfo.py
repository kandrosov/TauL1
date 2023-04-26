import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import statsmodels.stats.proportion as ssp
import argparse
from CommonDef import *


def get_x_var(x, y, w, meta):
    return meta[:, get_index(args.var)]


def getInfo(dataset,pred, var ,thr,required_type='tau'):
	print(f"required type is {required_type}")
	var_den_presel = np.concatenate(list(dataset.batch(300).map(get_x_var).as_numpy_iterator()))
	gen_truth = np.concatenate(list(dataset.batch(300).map(get_y_info).as_numpy_iterator()))
	tau_type = np.concatenate(list(dataset.batch(300).map(get_tauType_info).as_numpy_iterator()))
	hw_iso = np.concatenate(list(dataset.batch(300).map(to_hwIso).as_numpy_iterator()))
	all_var = np.vstack((var_den_presel[:], pred[:,0], gen_truth[:,0], hw_iso[:], tau_type[:])).T
	#all_var = np.vstack((var_den_presel[:], gen_truth[:,0],tau_type[:])).T
	condition_type = all_var[:,4]==getattr(TauType, required_type)
	condition_gen_truth = all_var[:,2]==1
	condition_thr = all_var[:,1]> thr #0.40430108# 0.6482158
	condition_hwIso = all_var[:,3]==1 if required_type=='tau' else all_var[:,3]==0
	condition_nn_based =  condition_type & condition_thr
	condition_cut_based = condition_type & condition_hwIso
	var_den = all_var[condition_type][:,0]
	var_num = all_var[condition_nn_based][:,0]
	var_num_iso = all_var[condition_cut_based][:,0]
	#print(var_num.min(), var_num.max())
	print(f"len of {var} num iso = {len(var_num_iso)}, len of {var} den = {len(var_den)}")
	print(f"len of {var} num = {len(var_num)}, len of {var} den = {len(var_den)}")




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=int, default = 2)
parser.add_argument('--threshold', required=False, type=float, default = 0.40430108)
parser.add_argument('--model_version', required=False, type=int, default = 7)
parser.add_argument('--var', required=False, type=str, default = 'L1Tau_gen_pt')
args = parser.parse_args()

#var = 'L1Tau_gen_pt'

model = keras.models.load_model(f'models/model_v{args.model_version}')
dataset = tf.data.Dataset.load(f'skim_v1_tf_v1/taus_{args.dataset}', compression='GZIP')
ds_pred = dataset.batch(300).map(to_pred)
pred = model.predict(ds_pred)
getInfo(dataset,pred,args.var,args.threshold,'e')
'''
for tau_type in ['e', 'tau', 'jet']:
	print(f"tauType is {tau_type}")
	getInfo(dataset,pred, args.var ,args.threshold,tau_type)
'''
