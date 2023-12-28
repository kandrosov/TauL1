import awkward as ak
import json
import numpy as np
import os
import sys
import uproot
import scipy.optimize

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

def passed_to_rate(passed, n_total):
  return passed / n_total * collision_rate

def pass_masks_to_count(*pass_masks):
  if len(pass_masks) == 0:
    return 0
  cmb_mask = pass_masks[0]
  for mask in pass_masks[1:]:
    cmb_mask = cmb_mask | mask
  return ak.sum(cmb_mask)

def pass_eta_acceptance(L1Tau_eta, eta_thr=2.131):
  return np.abs(L1Tau_eta) <= eta_thr

def get_tau_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_thr, require_iso):
  assert(tau_pt_thr > 0)
  sel = (L1Tau_pt >= tau_pt_thr) & pass_eta_acceptance(L1Tau_eta)
  if require_iso:
    sel = sel & (L1Tau_hwIso > 0)
  return sel

def get_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_thr, n_expected, require_iso):
  sel = get_tau_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_thr, require_iso)
  return ak.sum(sel, axis=1) >= n_expected

def get_pass_mask_default_or(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_iso_thr, tau_pt_noiso_thr, n_expected):
  assert(tau_pt_iso_thr > 0 or tau_pt_noiso_thr > 0)
  masks = []
  if tau_pt_iso_thr > 0:
    masks.append(get_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_iso_thr, n_expected, True))
  if tau_pt_noiso_thr > 0:
    masks.append(get_pass_mask_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_noiso_thr, n_expected, False))
  if len(masks) == 0:
    raise RuntimeError('No selection provided')
  mask = masks[0]
  for other_mask in masks[1:]:
    mask = mask | other_mask
  return mask

def get_passed_default(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_iso_thr, tau_pt_noiso_thr, n_expected, extra_cond=None):
  assert(tau_pt_iso_thr > 0 or tau_pt_noiso_thr > 0)
  masks = [ get_pass_mask_default_or(L1Tau_pt, L1Tau_eta, L1Tau_hwIso, tau_pt_iso_thr, tau_pt_noiso_thr, n_expected) ]
  if extra_cond is not None:
    masks.append(extra_cond)
  return pass_masks_to_count(*masks)

def get_tau_pass_mask_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn):
  return sel_fn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs)

def get_pass_mask_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn, n_expected):
  sel = get_tau_pass_mask_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn)
  return ak.sum(sel, axis=1) >= n_expected

def get_passed_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn, n_expected, extra_cond=None):
  masks = [ get_pass_mask_nn(L1Tau_pt, L1Tau_eta, L1Tau_NNtag, L1Tau_ptReg, nn_thrs, sel_fn, n_expected) ]
  if extra_cond is not None:
    masks.append(extra_cond)
  return pass_masks_to_count(*masks)

def mk_sel_fn(sel_bins, sel_var='L1Tau_pt'):
  def _sel_fn(pt, eta, nn_score, nn_thrs):
    assert(len(sel_bins) == len(nn_thrs))
    sel_base = pass_eta_acceptance(eta)
    sel = ak.zeros_like(pt, dtype=bool)
    for bin_idx in range(len(sel_bins) - 1):
      sel = sel | ( (pt >= sel_bins[bin_idx]) & (pt < sel_bins[bin_idx + 1]) & (nn_score >= nn_thrs[bin_idx]) )
    sel = sel | ( (pt >= sel_bins[-1]) & (nn_score >= nn_thrs[-1]) )
    return sel_base & sel

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

def find_nn_thrs(L1Tau_pt_mc, L1Tau_eta_mc, L1Tau_NNtag_mc, L1Tau_ptReg_mc,
                 L1Tau_pt_data, L1Tau_eta_data, L1Tau_NNtag_data, L1Tau_ptReg_data,
                 sel_fn, n_nn_thrs, n_expected_taus, target_rate,
                 initial_thrs=None, extra_cond_eff=None, step=0.01, use_bfgs=False, force_monotonic_thrs=False,
                 min_passed_L1Tau_pt=None, min_passed_L1Tau_eta=None, min_passed_L1Tau_NNtag=None,
                 min_passed_L1Tau_ptReg=None, min_passed_L1Tau_type=None, min_passed_var=None, min_passed_bins=None, min_passed_counts=None,
                 verbose=0):
  n_total_data = len(L1Tau_pt_data)
  n_total_mc =len(L1Tau_pt_mc)
  max_thr_int_value = int(1/step) + 1

  cache = {}

  def get_eff_rate(nn_thrs_int):
    key = tuple(nn_thrs_int.tolist())
    if key in cache:
      eff, rate, passed_min_limit, passed_counts = cache[key]
      return eff, rate, passed_min_limit, passed_counts
    nn_thrs = nn_thrs_int.astype(float) * step
    n_passed_mc = get_passed_nn(L1Tau_pt_mc, L1Tau_eta_mc, L1Tau_NNtag_mc, L1Tau_ptReg_mc,
                                nn_thrs, sel_fn, n_expected_taus, extra_cond=extra_cond_eff)
    eff = n_passed_mc / n_total_mc
    n_passed_data = get_passed_nn(L1Tau_pt_data, L1Tau_eta_data, L1Tau_NNtag_data, L1Tau_ptReg_data,
                                  nn_thrs, sel_fn, n_expected_taus)
    rate = passed_to_rate(n_passed_data, n_total_data)
    if min_passed_counts is not None:
      mask = get_tau_pass_mask_nn(min_passed_L1Tau_pt, min_passed_L1Tau_eta, min_passed_L1Tau_NNtag,
                                  min_passed_L1Tau_ptReg, nn_thrs, sel_fn)
      mask = mask & (min_passed_L1Tau_type == TauType.tau)
      passed_var = ak.flatten(min_passed_var[mask])
      passed_counts = np.histogram(passed_var, bins=min_passed_bins)[0]
      passed_min_limit = np.all(passed_counts >= min_passed_counts)
    else:
      passed_counts = None
      passed_min_limit = True

    cache[key] = (eff, rate, passed_min_limit, passed_counts)
    if verbose > 0:
      thr_int_str = '[' + ','.join([ str(x) for x in nn_thrs_int ]) + ']'
      print(f'--> thr_int={thr_int_str}, eff={eff}, rate={rate}, passed_min_limit={passed_min_limit}')
    return eff, rate, passed_min_limit, passed_counts

  def align_thrs(thrs):
    if force_monotonic_thrs:
      for thr_idx in reversed(range(1, n_nn_thrs)):
        thrs[thr_idx] = max(min(max_thr_int_value, thrs[thr_idx]), 0)
        if thrs[thr_idx] > thrs[thr_idx - 1]:
          thrs[thr_idx - 1] = thrs[thr_idx]

  def update_state(state=None, diff=None):
    if state is None:
      state = {
        'ref_weights': 1. / np.array(list(range(1, n_nn_thrs+1))),
        'alt_weights': 1. / np.array(list(range(1, n_nn_thrs+1))),
        'stage2_cnt': 0,
      }
    if diff is not None:
      for n in range(n_nn_thrs):
        if diff[n] != 0:
          col = 'ref_weights' if diff[n] < 0 else 'alt_weights'
          state[col][n] += 1
    for key, value in state.items():
      if key == 'stage2_cnt': continue
      state[key] = value / np.sum(value) * n_nn_thrs
    return state

  def get_range(weights):
    x = list(range(n_nn_thrs))
    return sorted(x, key=lambda x: -weights[x])

  def minimization_step_stage1(eff, passed_counts, thr_int, state):
    best_eff = eff
    best_thrs = thr_int
    has_change = False
    for thr_idx in reversed(range(n_nn_thrs)):
      if best_thrs[thr_idx] == 0: continue
      if state['stage2_cnt'] == 0:
        thr_step = best_thrs[thr_idx]
      else:
        thr_step = 1
      while thr_step > 0:
        thr_int_upd = best_thrs.copy()
        thr_int_upd[thr_idx] -= thr_step
        thr_step = thr_step // 2
        align_thrs(thr_int_upd)
        if np.array_equal(thr_int_upd, best_thrs): continue
        eff_upd, rate_upd, passed_min_limit_upd, passed_counts_upd = get_eff_rate(thr_int_upd)
        if rate_upd > target_rate or not passed_min_limit_upd: continue
        if best_eff < eff_upd or (best_eff == eff_upd and np.sum(thr_int_upd) < np.sum(best_thrs)):
          best_eff = eff_upd
          best_thrs = thr_int_upd
          has_change = True
    return has_change, best_thrs

  def minimization_step_stage2(eff, rate, thr_int, state):
    state['stage2_cnt'] += 1
    range_ref = get_range(state['ref_weights'])
    range_alt = get_range(state['alt_weights'])
    if verbose > 0:
      print(f'---> range_ref={range_ref}')
      print(f'---> range_alt={range_alt}')
    for thr_idx_ref in range_ref:
      if thr_int[thr_idx_ref] == 0: continue
      thr_step_ref = 1
      eff_upd_ref = 0
      while thr_int[thr_idx_ref] - thr_step_ref >= 0:
        thr_int_upd = thr_int.copy()
        thr_int_upd[thr_idx_ref] -= thr_step_ref
        align_thrs(thr_int_upd)
        if np.array_equal(thr_int_upd, thr_int):
          break
        if verbose > 0:
          print(f'---> thr_idx_ref={thr_idx_ref} thr_step_ref={thr_step_ref}')
        eff_upd_ref, _, _, _ = get_eff_rate(thr_int_upd)
        if eff_upd_ref > eff:
          break
        if thr_step_ref < 4:
          thr_step_ref += 1
        else:
          thr_step_ref = max(min(thr_step_ref * 2, thr_int[thr_idx_ref]), thr_step_ref + 1)
      if eff_upd_ref <= eff: continue
      for thr_idx in range_alt:
        if thr_idx == thr_idx_ref: continue
        thr_step = 1
        passed_min_upd = True
        while thr_int[thr_idx] + thr_step <= max_thr_int_value and passed_min_upd:
          thr_step_ref_upd = thr_step_ref
          best_eff_upd = eff
          best_thr_int_upd = thr_int
          thr_int_upd_prev = thr_int
          while thr_int_upd[thr_idx_ref] - thr_step_ref_upd >= 0:
            thr_int_upd = thr_int.copy()
            thr_int_upd[thr_idx_ref] -= thr_step_ref_upd
            thr_int_upd[thr_idx] += thr_step
            align_thrs(thr_int_upd)
            if np.array_equal(thr_int_upd, thr_int_upd_prev): break
            thr_int_upd_prev = thr_int_upd
            if verbose > 0:
              print(f'---> thr_idx_ref={thr_idx_ref} thr_step_ref={thr_step_ref_upd}'
                    f' thr_idx={thr_idx} thr_step={thr_step}')
            thr_step_ref_upd = max(min(thr_step_ref_upd * 2, thr_int_upd[thr_idx_ref]), thr_step_ref_upd + 1)
            eff_upd, rate_upd, passed_min_upd, _ = get_eff_rate(thr_int_upd)
            if rate_upd > target_rate or not passed_min_upd: break
            if eff_upd > best_eff_upd or (eff_upd == best_eff_upd and np.sum(thr_int_upd) < np.sum(best_thr_int_upd)):
              best_eff_upd = eff_upd
              best_thr_int_upd = thr_int_upd
          if best_eff_upd > eff or np.sum(best_thr_int_upd) < np.sum(thr_int):
            return True, best_thr_int_upd
          thr_step = max(min(thr_step * 2, max_thr_int_value - thr_int[thr_idx]), thr_step + 1)
    return False, thr_int

  def minimization_step_bfgs(eff, rate, thr_int, state):
    best_eff = eff
    best_thrs = thr_int
    has_change = False
    def loss_fn(nn_thrs):
      nonlocal best_eff, best_thrs, has_change
      nn_thrs_int = (np.array(nn_thrs, dtype=float) / step).astype(int)
      align_thrs(nn_thrs_int)
      eff, rate, _, _ = get_eff_rate(nn_thrs_int)
      loss = -eff
      if rate > target_rate:
        loss += rate / target_rate - 1
      elif best_eff < eff or (best_eff == eff and np.sum(nn_thrs_int) < np.sum(best_thrs)):
        best_eff = eff
        best_thrs = nn_thrs_int
        has_change = True
        if verbose > 0:
          thr_int_str = '[' + ','.join([ str(x) for x in thr_int ]) + ']'
          print(f'-> thr_int={thr_int_str}, eff={eff}, rate={rate}')
      return loss

    bounds = [ (-step, 1 + step) ] * n_nn_thrs
    x0 = thr_int.astype(float) * step

    scipy.optimize.minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds, options={ 'eps': step })
    return has_change, best_thrs

  has_change = True
  if initial_thrs is None:
    thr_int = np.zeros(n_nn_thrs, dtype=int)
  else:
    initial_thrs = np.array(initial_thrs, dtype=float)
    thr_int = np.array(initial_thrs / step, dtype=int)
  state = update_state()
  while has_change:
    for idx in range(n_nn_thrs):
      max_passed = thr_int[idx]
      min_notpassed = None
      max_allowed = max_thr_int_value #if idx == 0 else thr_int[idx - 1]
      while True:
        eff, rate, passed_min_limit, passed_counts = get_eff_rate(thr_int)
        if verbose > 0:
          thr_int_str = '[' + ','.join([ str(x) for x in thr_int ]) + ']'
          msg = f'thr_int={thr_int_str}, eff={eff}, rate={rate}'
          if passed_counts is not None:
            delta = passed_counts - min_passed_counts
            delta_str = '[' + ','.join([ str(x) for x in delta ]) + ']'
            msg += f', passed_counts_delta={delta_str}'
          print(msg)
        if not passed_min_limit:
          min_notpassed = thr_int[idx] if min_notpassed is None else min(min_notpassed, thr_int[idx])
        else:
          max_passed = max(thr_int[idx], max_passed)
        if passed_min_limit and rate <= target_rate: break
        if max_passed == max_allowed or max_passed+1 == min_notpassed:
          if passed_min_limit: break
          new_value = max_passed
        else:
          if min_notpassed is None:
              new_value = max_allowed
          else:
            new_value = (min_notpassed - max_passed) // 2 + max_passed
        if verbose > 0:
          print(f'max_passed={max_passed}, min_notpassed={min_notpassed},'
                f' new thr_int[idx]: {thr_int[idx]} -> {new_value}')
        if new_value == thr_int[idx]:
          raise RuntimeError(f'Failed to find new threshold for idx={idx}')
        thr_int[idx] = new_value
      if rate <= target_rate: break
    if rate > target_rate:
      raise RuntimeError('Unable to find thresholds with rate <= target_rate')
    print('opt stage 1 ...')
    has_change, thr_int_upd = minimization_step_stage1(eff, passed_counts, thr_int, state)
    if not has_change:
      if use_bfgs:
        print('opt stage bfgs ...')
        has_change, thr_int_upd = minimization_step_bfgs(eff, rate, thr_int, state)
      if not has_change:
        print('opt stage 2 ...')
        has_change, thr_int_upd = minimization_step_stage2(eff, rate, thr_int, state)
    diff = thr_int_upd - thr_int
    state = update_state(state, diff)
    thr_int = thr_int_upd

  best_eff, best_rate, _, _ = get_eff_rate(thr_int)
  best_nn_thrs = thr_int.astype(float) * step
  return best_eff, best_rate, best_nn_thrs

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
    n_passed = get_passed_default(df['L1Tau_pt'], df['L1Tau_eta'], df['L1Tau_hwIso'], pt_iso_thr, pt_noiso_thr, n_taus,
                                  extra_cond)
    if args.method == 'rate_default':
      rate = passed_to_rate(n_passed, n_total)
      print(f'Rate for n_taus >= {n_taus} pt_iso_thr = {pt_iso_thr:.0f} GeV and pt_noiso_thr = {pt_noiso_thr:.0f} GeV'
            f' is {rate:.1f} Hz')
    else:
      eff = n_passed / n_total
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
      rate = passed_to_rate(n_passed, n_total)
      print(f'NN rate for n_taus >= {n_taus} is {rate:.1f} Hz')
    else:
      eff = n_passed / n_total
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
