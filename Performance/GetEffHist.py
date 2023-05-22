import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import statsmodels.stats.proportion as ssp
import argparse
import sys
import os

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from .CommonDef import *

def to_hwIso(x,y,w,meta):
    return meta[:, get_index('L1Tau_hwIso')]

def get_y_info(x,y,w,meta):
    return y

def to_gen(x, y, w, meta):
  return y

def get_tauType_info(x,y,w,meta):
    return meta[:, get_index('L1Tau_type')]


def get_x_var(x, y, w, meta):
    return meta[:, get_index(args.var)]


def add_roc(dataset,pred,model_number):
	y = np.concatenate(list(dataset.batch(300).map(to_gen).as_numpy_iterator()))
	tau_type = np.concatenate(list(dataset.batch(300).map(get_tauType_info).as_numpy_iterator()))
	hwIso = np.concatenate(list(dataset.batch(300).map(to_hwIso).as_numpy_iterator()))
	hwIso_noEle=hwIso[tau_type[:]!=TauType.e]
	y_noEle=y[tau_type[:]!=TauType.e]
	fpr, tpr, threasholds = roc_curve(y_noEle, pred)
	hw_fpr, hw_tpr, hw_threasholds = roc_curve(y, hwIso_noEle)
	print(f'hw_fpr = {hw_fpr} \nhw_tpr={hw_tpr} \nhw_thresholds={hw_threasholds}')
	for fpr_idx in range(0, len(fpr)):
	  if(abs(fpr[fpr_idx]-hw_fpr[1])<0.001):
	    print(abs(fpr[fpr_idx]-hw_fpr[1]), fpr[fpr_idx], hw_fpr[1], threasholds[fpr_idx])
	#for tpr_idx in range(0, len(tpr)):
	#  if(tpr[tpr_idx]-hw_tpr[1]<0.00000001):
	#    print(tpr[tpr_idx], hw_tpr[1], threasholds[tpr_idx])

	with PdfPages(f'../Training/plots/roc_model_v{model_number}.pdf') as pdf:
  		fig, ax = plt.subplots(1, 1, figsize=(7, 7))
  		ax.plot(tpr, fpr, )
  		ax.errorbar(hw_tpr, hw_fpr, fmt='o', markersize=2, color='red')
  		ax.set_xlabel('true tau_h efficiency')
  		ax.set_ylabel('jet->tau efficiency')
  		plt.subplots_adjust(hspace=0)
  		pdf.savefig(fig, bbox_inches='tight')


def to_pred(x, y, w, meta):
  return (x[:,:,:,2:4], x[:, 0, 0, :2]), meta[:, get_index(args.var)],

def add_prediction(dataset, var, x_bins, thr, required_type='tau'):
  pred = None
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

	val_of_bins_num_iso, edges_of_bins_num_iso, patches_num_iso = plt.hist(var_num_iso, x_bins, range=(var_num.min(),var_num.max()), histtype='step', label="num_iso")
	val_of_bins_num, edges_of_bins_num, patches_num = plt.hist(var_num, x_bins, range=(var_num.min(),var_num.max()), histtype='step', label="num")
	#print("a")
	val_of_bins_den, edges_of_bins_den, patches_den = plt.hist(var_den, x_bins, range=(var_num.min(),var_num.max()), histtype='step', label="den")

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


	plt.savefig(f"plots/{var}_efficiency_{required_type}.pdf")




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=int, default = 2)
parser.add_argument('--threshold', required=False, type=float, default = 0.40430108)
parser.add_argument('--model_version', required=False, type=int, default = 5)
parser.add_argument('--var', required=False, type=str, default = 'L1Tau_gen_pt')
args = parser.parse_args()

#var = 'L1Tau_gen_pt'
x_bins=[0,40,60,80,100,150,200,250]

model = keras.models.load_model(f'../Training/models/model_v{args.model_version}')
dataset = tf.data.Dataset.load(f'../Training/skim_v1_tf_v1/taus_{args.dataset}', compression='GZIP')
ds_pred = dataset.batch(300).map(to_pred)
pred = model.predict(ds_pred)
print(pred.shape)
tau_type='tau'
#add_prediction(dataset,pred, args.var ,x_bins ,args.threshold,tau_type)
#add_roc(dataset,pred,args.model_version)

for tau_type in ['e', 'jet', 'tau']:
    add_prediction(dataset,pred, args.var ,x_bins ,args.threshold,tau_type)

