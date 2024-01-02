import awkward as ak
import json
import numba
import numpy as np
import os
import sys
import uproot

from statsmodels.stats.proportion import proportion_confint

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
  __package__ = 'TauL1'

from TauL1.CommonDef import TauType

collision_rate = 40e6 # Hz

def get_shortest_interval(x, alpha=0.68, step=0.01, rel_tolerance=0.01):
  assert(alpha > 0 and alpha < 1)
  q_values_left = np.arange(0, 1 - alpha + step, step).tolist()
  n_tot = len(q_values_left)
  half_interval = (1 - alpha) / 2
  q_values_left = sorted(q_values_left, key=lambda x: abs(half_interval - x))
  q_values_left = np.array(q_values_left)
  q_values_right = q_values_left + alpha
  q_values = np.concatenate((q_values_left, q_values_right))
  q_limits = np.quantile(x, q_values)
  q_limits_left = q_limits[:n_tot]
  q_limits_right = q_limits[n_tot:]
  shortest = None
  shortest_n = -1
  for n in range(n_tot):
    current_len = q_limits_right[n] - q_limits_left[n]
    if shortest is None or (shortest > 0 and (shortest - current_len) / shortest > rel_tolerance):
      shortest = current_len
      shortest_n = n
  assert(shortest_n != -1)
  return (q_limits_left[shortest_n], q_limits_right[shortest_n]), \
         (q_values_left[shortest_n], q_values_right[shortest_n])

def numpy_to_list(x_in):
  x_in_type = type(x_in)
  if x_in_type == np.ndarray:
    return x_in.tolist()
  if x_in_type == ak.Array:
    return ak.to_list(x_in)
  if x_in_type == list:
    return [ numpy_to_list(x) for x in x_in ]
  if x_in_type == dict:
    return { key : numpy_to_list(value) for key, value in x_in.items() }
  return x_in

def list_to_str(l):
  return '[' + ','.join([ str(x) for x in l ]) + ']'

def get_eff(n_passed, n_total, return_errors=False):
  assert(n_total > 0)
  assert(n_passed >= 0 and n_passed <= n_total)
  eff = n_passed / float(n_total)
  if return_errors:
    ci_low, ci_upp = proportion_confint(n_passed, n_total, alpha=1-0.68, method='beta')
    eff_up = ci_upp - eff
    eff_down = eff - ci_low
    return eff, eff_up, eff_down
  return eff

def get_rate(n_passed, n_total, return_errors=False):
  eff_out = get_eff(n_passed, n_total, return_errors=return_errors)
  if return_errors:
    eff, eff_up, eff_down = eff_out
    return eff * collision_rate, eff_up * collision_rate, eff_down * collision_rate
  return eff_out * collision_rate

def pass_masks_to_count(*pass_masks):
  if len(pass_masks) == 0:
    return 0
  cmb_mask = pass_masks[0]
  for mask in pass_masks[1:]:
    cmb_mask = cmb_mask | mask
  return ak.sum(cmb_mask)

@numba.jit(nopython=True)
def pass_eta_acceptance_float(L1Tau_eta, eta_thr=2.131):
  return abs(L1Tau_eta) <= eta_thr

def pass_eta_acceptance(L1Tau_eta, eta_thr=2.131):
  return np.abs(L1Tau_eta) <= eta_thr

def is_gen_tau(L1Tau_type):
  return L1Tau_type == TauType.tau

def get_tau_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_thr, require_iso):
  assert(tau_pt_thr > 0)
  sel = (L1Tau_pt >= tau_pt_thr) & pass_eta_acceptance(L1Tau_eta)
  if require_iso:
    sel = sel & (L1Tau_hwIso > 0)
  return sel

def get_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_thr, n_expected, require_iso, L1Tau_type=None,
                          require_gen_match=False):
  sel = get_tau_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_thr, require_iso)
  if require_gen_match:
    assert(L1Tau_type is not None)
    sel = sel & is_gen_tau(L1Tau_type)
  return ak.sum(sel, axis=1) >= n_expected

def get_pass_mask_default_or(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_iso_thr, tau_pt_noiso_thr, n_expected,
                             L1Tau_type=None, require_gen_match=False):
  assert(tau_pt_iso_thr > 0 or tau_pt_noiso_thr > 0)
  masks = []
  if tau_pt_iso_thr > 0:
    masks.append(get_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_iso_thr, n_expected, True,
                                       L1Tau_type, require_gen_match))
  if tau_pt_noiso_thr > 0:
    masks.append(get_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_noiso_thr, n_expected, False,
                                       L1Tau_type, require_gen_match))
  if len(masks) == 0:
    raise RuntimeError('No selection provided')
  mask = masks[0]
  for other_mask in masks[1:]:
    mask = mask | other_mask
  return mask

def get_pass_mask_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn, n_expected,
                     L1Tau_type=None, require_gen_match=False):
  sel = sel_fn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs)
  if require_gen_match:
    assert(L1Tau_type is not None)
    sel = sel & is_gen_tau(L1Tau_type)
  return ak.sum(sel, axis=1) >= n_expected

def get_passed_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn, n_expected, extra_cond=None,
                  L1Tau_type=None, require_gen_match=False):
  masks = [ get_pass_mask_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn, n_expected,
                             L1Tau_type, require_gen_match) ]
  if extra_cond is not None:
    masks.append(extra_cond)
  return pass_masks_to_count(*masks)

def mk_sel_fn(sel_bins, sel_var='L1Tau_pt'):
  @numba.jit(nopython=True)
  def _tau_pass(pt, eta, nn_score, nn_thrs):
    if not pass_eta_acceptance_float(eta) or pt < sel_bins[0]:
      return False
    for bin_idx in range(len(sel_bins) - 1):
      if pt >= sel_bins[bin_idx] and pt < sel_bins[bin_idx + 1]:
        return nn_score >= nn_thrs[bin_idx]
    return nn_score >= nn_thrs[-1]

  @numba.jit(nopython=True, parallel=True)
  def _sel_fn_loop(pt, eta, nn_score, nn_thrs, sel_result):
    for tau_idx in numba.prange(len(pt)):
      sel_result[tau_idx] = _tau_pass(pt[tau_idx], eta[tau_idx], nn_score[tau_idx], nn_thrs)

  def _sel_fn(pt, eta, nn_score, nn_thrs):
    assert(len(sel_bins) == len(nn_thrs))
    pt_flat = ak.flatten(pt)
    sel = np.empty(len(pt_flat), dtype=bool)
    _sel_fn_loop(pt_flat, ak.flatten(eta), ak.flatten(nn_score), nn_thrs, sel)
    return ak.unflatten(sel, ak.num(pt))

  if sel_var == 'L1Tau_pt':
    def sel_fn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs):
      return _sel_fn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, nn_thrs)
  elif sel_var == 'L1Tau_ptReg':
    def sel_fn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs):
      return _sel_fn(L1Tau_ptReg, L1Tau_eta, L1Tau_NNtag, nn_thrs)
  else:
    raise RuntimeError(f'Unknown selection variable {sel_var}')

  if len(sel_bins) == 0:
    raise RuntimeError('No selection bins provided')
  for bin_idx in range(1, len(sel_bins)):
    if sel_bins[bin_idx] <= sel_bins[bin_idx - 1]:
      raise RuntimeError('Selection bins must be in increasing order')
  return sel_fn

def get_differential_efficiency(bins, eff_round_decimals=4, **selections):
  hists = {}
  total_name = 'total'
  if total_name not in selections:
    raise RuntimeError('No total selection provided')
  for name, x in selections.items():
    hists[name] = np.histogram(x, bins=bins)[0]

  bin_sel = hists[total_name] > 0
  bins_all = np.zeros((len(bins) - 1, 2))
  bins_all[:, 0] = bins[:-1]
  bins_all[:, 1] = bins[1:]
  bins_selected = bins_all[bin_sel]
  hist_total = hists[total_name][bin_sel]

  eff = {
    'x': (bins_selected[:, 1] + bins_selected[:, 0]) / 2,
    'x_up': (bins_selected[:, 1] - bins_selected[:, 0]) / 2,
    'x_down': (bins_selected[:, 1] - bins_selected[:, 0]) / 2,
    'values': {}
  }
  for name, hist in hists.items():
    if name == total_name: continue
    hist_num = hist[bin_sel]
    central = hist_num / hist_total
    ci_low, ci_upp = proportion_confint(hist_num, hist_total, alpha=1-0.68, method='beta')
    eff['values'][name] = {
      'y': np.around(central, eff_round_decimals),
      'y_up': np.around(ci_upp - central, eff_round_decimals),
      'y_down': np.around(central - ci_low, eff_round_decimals),
    }
  return eff

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, type=str)
  parser.add_argument('--scores', required=False, default=None, type=str)
  parser.add_argument('--method', required=True, type=str)
  parser.add_argument('method_args', nargs='*', type=str)
  args = parser.parse_args()

  if args.method in [ 'rate_default', 'eff_default' ]:
    n_taus = int(args.method_args[0])
    columns = ['L1Tau_pt', 'L1Tau_eta', 'L1Tau_hwIso']

    pt_iso_thr = float(args.method_args[1]) if len(args.method_args) > 1 else None
    pt_noiso_thr = float(args.method_args[2]) if len(args.method_args) > 2 else None
    if len(args.method_args) > 3:
      extra_cond_col = args.method_args[3]
      columns.append(extra_cond_col)
    else:
      extra_cond_col = None
    df = uproot.open(args.dataset)['Events'].arrays(columns)
    extra_cond = df[extra_cond_col] if extra_cond_col is not None else None
    if pt_iso_thr is None:
      pt_iso_thr = 34 if n_taus > 1 else -1
      pt_noiso_thr = 70 if n_taus > 1 else 120
    n_total = len(df['L1Tau_pt'])
    masks = [ get_pass_mask_default_or(df['L1Tau_pt'], df['L1Tau_eta'], df['L1Tau_hwIso'], pt_iso_thr, pt_noiso_thr, n_taus) ]
    if extra_cond is not None:
      masks.append(extra_cond)
    n_passed = pass_masks_to_count(*masks)
    if args.method == 'rate_default':
      rate = get_rate(n_passed, n_total)
      print(f'Rate for n_taus >= {n_taus} pt_iso_thr = {pt_iso_thr:.0f} GeV and pt_noiso_thr = {pt_noiso_thr:.0f} GeV'
            f' is {rate:.1f} Hz')
    else:
      eff = get_eff(n_passed, n_total)
      print(f'Efficiency for n_taus >= {n_taus} is {eff*100:.2f}%')
  elif args.method in [ 'rate_nn', 'eff_nn' ]:
    assert(args.scores is not None)
    n_taus = int(args.method_args[0])
    columns = [ 'L1Tau_pt', 'L1Tau_eta' ]
    nn_columns = [ 'L1Tau_NNtag', 'L1Tau_ptReg' ]
    sel_var = args.method_args[1]
    sel_bins = [ float(x) for x in  args.method_args[2].split(',') ]
    nn_thrs = [ float(x) for x in args.method_args[3].split(',') ]
    if len(args.method_args) > 4:
      extra_cond_col = args.method_args[4]
      columns.append(extra_cond_col)
    else:
      extra_cond_col = None
    df = uproot.open(args.dataset)['Events'].arrays(columns)
    df_scores = uproot.open(args.scores)['Events'].arrays(nn_columns)
    extra_cond = df[extra_cond_col] if extra_cond_col is not None else None
    assert(len(sel_bins) == len(nn_thrs))
    sel_fn = mk_sel_fn(sel_bins, sel_var)
    n_total = len(df['L1Tau_pt'])
    n_passed = get_passed_nn(df['L1Tau_pt'], df['L1Tau_eta'], df_scores['L1Tau_NNtag'],
                             df_scores['L1Tau_ptReg'], nn_thrs, sel_fn, n_taus, extra_cond)
    if args.method == 'rate_nn':
      rate = get_rate(n_passed, n_total)
      print(f'NN rate for n_taus >= {n_taus} is {rate:.1f} Hz')
    else:
      eff = get_eff(n_passed, n_total)
      print(f'NN efficiency for n_taus >= {n_taus} is {eff*100:.2f}%')
  elif args.method == 'eff_diff':
    var_name = args.method_args[0]
    bins = np.array([ float(x) for x in args.method_args[1].split(',') ])
    columns = [ 'L1Tau_pt', 'L1Tau_eta', 'L1Tau_hwIso', 'L1Tau_type', 'L1Tau_gen_pt', 'L1Tau_gen_eta' ]
    nn_columns = [ 'L1Tau_NNtag', 'L1Tau_ptReg' ]
    if var_name not in nn_columns and var_name not in columns:
      columns.append(var_name)
    df = uproot.open(args.dataset)['Events'].arrays(columns)
    df_scores = uproot.open(args.scores)['Events'].arrays(nn_columns)
    sel_var = args.method_args[2]
    sel_bins = [ float(x) for x in  args.method_args[3].split(',') ]
    nn_thrs = [ float(x) for x in args.method_args[4].split(',') ]
    assert(len(sel_bins) == len(nn_thrs))
    sel_fn = mk_sel_fn(sel_bins, sel_var)
    base_sel = (df['L1Tau_gen_pt'] >= 20) & (np.abs(df['L1Tau_gen_eta']) < 2.5) & (df['L1Tau_type'] == TauType.tau)
    default_sel = base_sel & (df['L1Tau_pt'] >= 34) & (np.abs(df['L1Tau_eta']) <= 2.131) & (df['L1Tau_hwIso'] > 0)
    nn_sel = base_sel & sel_fn(df['L1Tau_pt'], df['L1Tau_eta'], df_scores['L1Tau_NNtag'], df_scores['L1Tau_ptReg'],
                               nn_thrs)
    var_df = df if var_name in columns else df_scores
    selections = {
      'total': ak.flatten(var_df[var_name][base_sel]),
      'default': ak.flatten(var_df[var_name][default_sel]),
      'nn': ak.flatten(var_df[var_name][nn_sel]),
    }
    eff = get_differential_efficiency(bins, **selections)
    eff = numpy_to_list(eff)
    print(json.dumps(eff, indent=2))
  elif args.method == 'find_nn_thrs':
    n_expected_taus = int(args.method_args[0])
    target_rate = float(args.method_args[1])
    dataset_data = args.method_args[2]
    scores_data = args.method_args[3]
    sel_var = args.method_args[4]
    sel_bins = [ float(x) for x in  args.method_args[5].split(',') ]
    n_nn_thrs = len(sel_bins)
    sel_fn = mk_sel_fn(sel_bins, sel_var)
    initial_thrs = None
    if len(args.method_args) > 6 and args.method_args[6] != 'None':
      initial_thrs = [ float(x) for x in args.method_args[6].split(',') ]
      assert(len(initial_thrs) == n_nn_thrs)
    columns = [ 'L1Tau_pt', 'L1Tau_eta' ]
    nn_columns = [ 'L1Tau_NNtag', 'L1Tau_ptReg' ]
    if len(args.method_args) > 7:
      extra_cond_col = args.method_args[7]
      columns.append(extra_cond_col)
    else:
      extra_cond_col = None

    df_mc = uproot.open(args.dataset)['Events'].arrays(columns)
    df_mc_scores = uproot.open(args.scores)['Events'].arrays(nn_columns)
    df_data = uproot.open(dataset_data)['Events'].arrays(columns)
    df_data_scores = uproot.open(scores_data)['Events'].arrays(nn_columns)

    extra_cond = df_mc[extra_cond_col] if extra_cond_col is not None else None
    eff, rate, nn_thrs = find_nn_thrs(df_mc['L1Tau_pt'], df_mc['L1Tau_eta'], df_mc_scores['L1Tau_NNtag'],
                                      df_mc_scores['L1Tau_ptReg'],
                                      df_data['L1Tau_pt'], df_data['L1Tau_eta'], df_data_scores['L1Tau_NNtag'],
                                      df_data_scores['L1Tau_ptReg'],
                                      sel_fn, n_nn_thrs, n_expected_taus, target_rate, initial_thrs=initial_thrs,
                                      extra_cond_eff=extra_cond, verbose=1)
    thr_str = '[' + ','.join([ str(x) for x in nn_thrs ]) + ']'
    print(f'NN efficiency for n_taus >= {n_expected_taus} with thresholds={thr_str} is {eff*100:.2f}%'
          f' with data rate = {rate:.1f} Hz')
  else:
    raise RuntimeError(f'Unknown method {args.method}')
