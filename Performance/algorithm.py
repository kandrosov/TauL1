import awkward as ak
import json
import numpy as np
import os
import scipy.optimize


from TauL1.RunKit.run_tools import print_ts
from .eval_tools import get_pass_mask_default_or, get_pass_mask_nn, mk_sel_fn, pass_masks_to_count, \
                        get_eff, get_rate, get_tau_pass_mask_default, is_gen_tau, list_to_str, get_passed_nn


def find_nn_thrs(L1Tau_pt_mc, L1Tau_eta_mc, L1Tau_NNtag_mc, L1Tau_ptReg_mc, L1Tau_type_mc,
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
                                nn_thrs, sel_fn, n_expected_taus, extra_cond=extra_cond_eff,
                                L1Tau_type=L1Tau_type_mc, require_gen_match=True)
    eff = get_eff(n_passed_mc, n_total_mc)
    n_passed_data = get_passed_nn(L1Tau_pt_data, L1Tau_eta_data, L1Tau_NNtag_data, L1Tau_ptReg_data,
                                  nn_thrs, sel_fn, n_expected_taus)
    rate = get_rate(n_passed_data, n_total_data)
    if min_passed_counts is not None:
      mask = sel_fn(min_passed_L1Tau_pt, min_passed_L1Tau_eta, min_passed_L1Tau_NNtag, min_passed_L1Tau_ptReg, nn_thrs)
      mask = mask & is_gen_tau(min_passed_L1Tau_type)
      passed_var = ak.flatten(min_passed_var[mask])
      passed_counts = np.histogram(passed_var, bins=min_passed_bins)[0]
      passed_min_limit = np.all(passed_counts >= min_passed_counts)
    else:
      passed_counts = None
      passed_min_limit = True

    cache[key] = (eff, rate, passed_min_limit, passed_counts)
    if verbose > 1:
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
        thr_int_upd[thr_idx] = max(thr_int_upd[thr_idx] - thr_step, 0)
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
    if verbose > 1:
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
        if verbose > 1:
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
            if verbose > 1:
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
        if verbose > 1:
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
      max_passed = None
      min_notpassed = None
      max_allowed = max_thr_int_value
      while True:
        eff, rate, passed_min_limit, passed_counts = get_eff_rate(thr_int)
        if verbose > 0:
          msg = f'thr_int={list_to_str(thr_int)}, eff={eff}, rate={rate}, cache_size={len(cache)}'
          if passed_counts is not None:
            delta = passed_counts - min_passed_counts
            msg += f', passed_counts_delta={list_to_str(delta)}'
          print_ts(msg)
        if not passed_min_limit:
          min_notpassed = thr_int[idx] if min_notpassed is None else min(min_notpassed, thr_int[idx])
        else:
          max_passed = thr_int[idx] if max_passed is None else max(thr_int[idx], max_passed)
        if passed_min_limit and rate <= target_rate: break
        if max_passed is not None and (max_passed == max_allowed or max_passed+1 == min_notpassed):
          if passed_min_limit: break
          new_value = max_passed
        else:
          if min_notpassed is None:
            new_value = max_allowed
          elif max_passed is None:
            new_value = 0
          else:
            new_value = (min_notpassed - max_passed) // 2 + max_passed
        if verbose > 1:
          print(f'max_passed={max_passed}, min_notpassed={min_notpassed},'
                f' new thr_int[idx]: {thr_int[idx]} -> {new_value}')
        if new_value == thr_int[idx]:
          raise RuntimeError(f'Failed to find new threshold for idx={idx}')
        thr_int[idx] = new_value
      if rate <= target_rate: break
    if rate > target_rate:
      raise RuntimeError('Unable to find thresholds with rate <= target_rate')
    if verbose > 1:
      print('opt stage 1 ...')
    has_change, thr_int_upd = minimization_step_stage1(eff, passed_counts, thr_int, state)
    if not has_change:
      if use_bfgs:
        if verbose > 1:
          print('opt stage bfgs ...')
        has_change, thr_int_upd = minimization_step_bfgs(eff, rate, thr_int, state)
      if not has_change:
        if verbose > 1:
          print('opt stage 2 ...')
        has_change, thr_int_upd = minimization_step_stage2(eff, rate, thr_int, state)
    diff = thr_int_upd - thr_int
    state = update_state(state, diff)
    thr_int = thr_int_upd

  best_eff, best_rate, _, _ = get_eff_rate(thr_int)
  best_nn_thrs = thr_int.astype(float) * step
  return best_eff, best_rate, best_nn_thrs

class Algorithm:
  def __init__(self, name, cfg, algo_params_dir, datasets, lut_bins):
    try:
      self.name = name
      self.algo = cfg['algo']
      self.columns = set(['L1Tau_type'])
      self.initialized = True
      if self.algo == 'composite':
        self.is_composite = True
        self.sub_algos = cfg['sub_algos']
        self.initialized = False
      else:
        self.is_composite = False
        self.valid_per_tau = cfg.get('valid_per_tau', True)
        if self.algo == 'l1_flag':
          self.l1_flag = cfg['l1_flag']
          self.columns.add(self.l1_flag)
        else:
          self.n_taus = int(cfg['n_taus'])
          if self.algo == 'default':
            self.pt_iso_thr = float(cfg.get('pt_iso_thr', -1))
            self.pt_noiso_thr = float(cfg.get('pt_noiso_thr', -1))
            self.columns.update(['L1Tau_pt', 'L1Tau_eta', 'L1Tau_hwIso'])
          elif self.algo == 'nn':
            self.lut_var = cfg['lut_var']
            self.lut_bins = lut_bins[cfg['lut_bins']]
            self.columns.update([ 'L1Tau_NNtag', 'L1Tau_ptReg', self.lut_var ])
            self.sel_fn = mk_sel_fn(self.lut_bins, self.lut_var)
            opt = cfg['thresholds_opt']
            self.thresholds_opt = {}
            self.thresholds_opt['dataset_eff'] = datasets[opt['dataset_eff']]
            self.thresholds_opt['dataset_rate'] = datasets[opt['dataset_rate']]
            self.thresholds_opt['extra_algos_eff'] = opt.get('extra_algos_eff', [])
            self.thresholds_opt['target_rate'] = opt['target_rate']
            self.thresholds_opt['initial_thresholds'] = opt.get('initial_thresholds', None)
            self.thresholds_opt['step'] = opt.get('step', 0.01)
            if 'tau_eff_var' in opt:
              self.thresholds_opt['tau_eff_var'] = (opt['tau_eff_var'], opt['tau_eff_dataset'], opt['tau_eff_algo'])
              self.thresholds_opt['tau_eff_scale'] = opt.get('tau_eff_scale', 1.)
            else:
              self.thresholds_opt['tau_eff_var'] = None
              self.thresholds_opt['tau_eff_scale'] = None
            self.initialized = False
            self.thresholds = None
            self.thresholds_path = os.path.join(algo_params_dir, f'{self.name}.json')
            if os.path.exists(self.thresholds_path):
              with open(self.thresholds_path, 'r') as f:
                params = json.load(f)
                entry = params[0]
                entry_bins = np.array(entry['lut_bins'], dtype=np.float32)
                if not np.array_equal(entry_bins, self.lut_bins):
                  raise RuntimeError(f'Algorithm {self.name}: stored bins {entry_bins} != {self.lut_bins}')
                self.thresholds = np.array(entry['thresholds'], dtype=np.float32)
          else:
            raise RuntimeError(f'Unknown algorithm {self.algo}')
    except KeyError as e:
      raise RuntimeError(f'algorithm {name}: missing key {e}')

  def initialize_composite(self, algos):
    if not self.initialized and self.is_composite:
      self.sub_algos = { algo_name: algos[algo_name] for algo_name in self.sub_algos }
      self.valid_per_tau = True
      for algo in self.sub_algos.values():
        self.columns.update(algo.columns)
        self.valid_per_tau = self.valid_per_tau and algo.valid_per_tau
      self.initialized = True

  def initialize_simple(self, algos, variables):
    if self.initialized or self.is_composite:
      return
    if self.algo == 'nn':
      self.thresholds_opt['target_rate'] = variables.get(var_name='rate', algo_name=self.thresholds_opt['target_rate'])
      self.thresholds_opt['extra_algos_eff'] = \
        { algo_name: algos[algo_name] for algo_name in self.thresholds_opt['extra_algos_eff'] }
      if self.thresholds_opt['initial_thresholds'] is not None:
        self.thresholds_opt['initial_thresholds'] = algos[self.thresholds_opt['initial_thresholds']]
      if self.thresholds_opt['tau_eff_var'] is not None:
        var_name, ds_name, algo_name = self.thresholds_opt['tau_eff_var']
        self.thresholds_opt['tau_eff_var'] = variables.get(var_name=var_name, ds_name=ds_name, algo_name=algo_name)
    self.initialized = True

  def set_thresholds(self, thresholds):
    if self.algo != 'nn':
      raise RuntimeError(f'Algorithm {self.name} is not a neural network')
    self.thresholds = np.array(thresholds, dtype=np.float32)
    with open(self.thresholds_path, 'w') as f:
      json.dump([ { 'lut_bins': self.lut_bins.tolist(), 'thresholds': thresholds.tolist() }], f)

  def is_ready(self):
    if not self.initialized:
      return False
    if self.is_composite:
      for sub_algo in self.sub_algos.values():
        if not sub_algo.is_ready():
          return False
    else:
      if self.algo == 'nn':
        if self.thresholds is None:
          return False
    return True

  def is_ready_for_opt(self):
    if self.algo != 'nn':
      return False
    if type(self.thresholds_opt['target_rate']) != float:
      if not self.thresholds_opt['target_rate'].is_ready():
        return False
      self.thresholds_opt['target_rate'] = self.thresholds_opt['target_rate'].value
    if self.thresholds_opt['initial_thresholds'] is not None:
      if type(self.thresholds_opt['initial_thresholds']) != np.ndarray:
        if self.thresholds_opt['initial_thresholds'].thresholds is None:
          return False
        self.thresholds_opt['initial_thresholds'] = self.thresholds_opt['initial_thresholds'].thresholds
    if self.thresholds_opt['tau_eff_var'] is not None:
      if not self.thresholds_opt['tau_eff_var'].is_ready():
        return False
    for extra_algo in self.thresholds_opt['extra_algos_eff'].values():
      if not extra_algo.is_ready():
        return False
    return True

  def _get_combined_mask(self, dataset, mask_fn, require_gen_match):
    pass_mask = None
    for sub_algo in self.sub_algos.values():
      pass_sub = getattr(sub_algo, mask_fn)(dataset, require_gen_match=require_gen_match)
      if pass_mask is None:
        pass_mask = pass_sub
      else:
        pass_mask = pass_mask | pass_sub
    return pass_mask

  def get_tau_pass_mask(self, dataset, require_gen_match=False):
    if not self.valid_per_tau:
      raise RuntimeError(f'Algorithm {self.name} is not valid per tau')
    if not self.is_ready():
      raise RuntimeError(f'Algorithm {self.name} is not ready')
    if self.is_composite:
      return self._get_combined_mask(dataset, 'get_tau_pass_mask', require_gen_match)
    else:
      if self.algo == 'default':
        df = dataset.get_input()
        if self.pt_iso_thr > 0:
          pt_thr = self.pt_iso_thr
          require_iso = True
        else:
          pt_thr = self.pt_noiso_thr
          require_iso = False
        mask = get_tau_pass_mask_default(df['L1Tau_pt'], df['L1Tau_eta'], df['L1Tau_hwIso'], pt_thr, require_iso)
      elif self.algo == 'nn':
        df = dataset.get_input()
        df_scores = dataset.get_scores()
        mask = self.sel_fn(df['L1Tau_pt'], df['L1Tau_eta'], df_scores['L1Tau_NNtag'], df_scores['L1Tau_ptReg'],
                                self.thresholds)
      else:
        raise RuntimeError(f'get_tau_pass_mask: algorithm {self.algo} not supported')

      if require_gen_match:
        mask = mask & is_gen_tau(dataset['L1Tau_type'])
      return mask

  def get_pass_mask(self, dataset, require_gen_match=False):
    if not self.is_ready():
      raise RuntimeError(f'Algorithm {self.name} is not ready')
    if self.is_composite:
      return self._get_combined_mask(dataset, 'get_pass_mask', require_gen_match)
    else:
      if self.algo == 'l1_flag':
        return dataset.get_input()[self.l1_flag]
      elif self.algo in 'default':
        df = dataset.get_input()
        return get_pass_mask_default_or(df['L1Tau_pt'], df['L1Tau_eta'], df['L1Tau_hwIso'], self.pt_iso_thr,
                                        self.pt_noiso_thr, self.n_taus, df['L1Tau_type'], require_gen_match)
      elif self.algo == 'nn':
        df = dataset.get_input()
        df_scores = dataset.get_scores()
        return get_pass_mask_nn(df['L1Tau_pt'], df['L1Tau_eta'], df_scores['L1Tau_NNtag'], df_scores['L1Tau_ptReg'],
                                self.thresholds, self.sel_fn, self.n_taus, df['L1Tau_type'], require_gen_match)
      else:
        raise RuntimeError(f'get_pass_mask: algorithm {self.algo} not supported')

  def get_passed(self, dataset, require_gen_match=False):
    mask = self.get_pass_mask(dataset, require_gen_match=require_gen_match)
    return pass_masks_to_count(mask)

  def optimize_thresholds(self):
    if self.algo != 'nn':
      raise RuntimeError(f'optimize_thresholds: algorithm {self.name} is not a neural network')
    if not self.is_ready_for_opt():
      raise RuntimeError(f'optimize_thresholds: algorithm {self.name} is not ready for optimization')

    ds_eff = self.thresholds_opt['dataset_eff']
    ds_rate = self.thresholds_opt['dataset_rate']

    n_nn_thrs = len(self.lut_bins)

    extra_cond_eff=None
    for algo in self.thresholds_opt['extra_algos_eff'].values():
      algo_mask = algo.get_pass_mask(ds_eff)
      if extra_cond_eff is None:
        extra_cond_eff = algo_mask
      else:
        extra_cond_eff = extra_cond_eff | algo_mask

    min_passed_args = {}
    if self.thresholds_opt['tau_eff_var'] is not None:
      tau_eff_var = self.thresholds_opt['tau_eff_var']
      _, tau_passed = tau_eff_var.get_total_and_passed()
      min_passed_args['min_passed_L1Tau_pt'] = tau_eff_var.dataset['L1Tau_pt']
      min_passed_args['min_passed_L1Tau_eta'] = tau_eff_var.dataset['L1Tau_eta']
      min_passed_args['min_passed_L1Tau_NNtag'] = tau_eff_var.dataset['L1Tau_NNtag']
      min_passed_args['min_passed_L1Tau_ptReg'] = tau_eff_var.dataset['L1Tau_ptReg']
      min_passed_args['min_passed_L1Tau_type'] = tau_eff_var.dataset['L1Tau_type']
      min_passed_args['min_passed_var'] = tau_eff_var.dataset[tau_eff_var.column]
      min_passed_args['min_passed_bins'] = tau_eff_var.bins
      #min_passed_args['min_passed_counts'] = np.ce(np.histogram(tau_passed, bins=tau_eff_var.bins)[0] * 0.95)
      min_passed_counts = np.histogram(tau_passed, bins=tau_eff_var.bins)[0]
      tau_eff_scale = self.thresholds_opt['tau_eff_scale']
      if tau_eff_scale != 1:
        min_passed_counts = np.ceil(min_passed_counts * tau_eff_scale)
      min_passed_args['min_passed_counts'] = np.array(min_passed_counts, dtype=int)

    eff, rate, nn_thrs = find_nn_thrs(ds_eff['L1Tau_pt'], ds_eff['L1Tau_eta'], ds_eff['L1Tau_NNtag'],
                                      ds_eff['L1Tau_ptReg'], ds_eff['L1Tau_type'],
                                      ds_rate['L1Tau_pt'], ds_rate['L1Tau_eta'], ds_rate['L1Tau_NNtag'],
                                      ds_rate['L1Tau_ptReg'], self.sel_fn,
                                      n_nn_thrs, self.n_taus, self.thresholds_opt['target_rate'],
                                      initial_thrs=self.thresholds_opt['initial_thresholds'],
                                      extra_cond_eff=extra_cond_eff, step=self.thresholds_opt['step'], verbose=1,
                                      **min_passed_args)
    thrs_str = ', '.join([ f'{thr:.2f}' for thr in nn_thrs ])
    print(f'Optimized thresholds for {self.name}: eff={eff}, rate={rate}, thrs=[{thrs_str}]')
    self.set_thresholds(nn_thrs)