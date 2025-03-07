from imports import (
  KFold,
  optuna
)
from OptSearchMap import OptSearchMap
from local_opt import local_opt


def global_opt(trial, global_study, model_name, tr_df,
               level_status, params_obj,
               global_obj_loss,
               cnt_patience_step,
               cnf_yml,
               random_state=42):

  trial_number = trial.number
  if trial_number == 0:
    global_study.set_user_attr("global_opt_loss", 20000)
    global_study.set_user_attr("cnt_patience_step", 0)

  global_opt_loss = global_study.user_attrs["global_opt_loss"]
  cnt_patience_step = global_study.user_attrs["cnt_patience_step"]
  patience_step_threshold = params_obj.patience_step_threshold
  cv_n = params_obj.cv_n
  accum_loss = 0.0

  cv =  KFold(n_splits=cv_n, shuffle=True, random_state=random_state)
  cv_obj = cv.split(tr_df)
  trgt_y = params_obj.trgt_y

  """
  phaseをparams_objに加える
  trgt_yをbc_objに加える
  """

  print(f"\n◎ {level_status}_{model_name}_GLOBAL_TRIAL_{trial.number + 1} strat...")

  # wrapped_pruner = optuna.pruners.HyperbandPruner()
  # pruner = optuna.pruners.PatientPruner(wrapped_pruner=wrapped_pruner, patience=10) if params_obj.USE_OPTUNA_PRUNER is True else None
  pruner = None
  sampler = optuna.samplers.TPESampler(seed=42)
  local_studies = [optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler) for _ in range(cv_n)]
  opt_params = OptSearchMap(trial=trial, model_name=model_name, params_obj=params_obj, cnf_yml=cnf_yml).opt_params
  opt_model_params = opt_params["opt_model_params"]
  opt_fit_params = opt_params["opt_fit_params"]
  global_study.set_user_attr("opt_model_params", opt_model_params)
  global_study.set_user_attr("opt_fit_params", opt_fit_params)


#   print("search_params(local_opt_params):\n", local_opt_params)
  # 値は一定だが早期終了によるloss値が変動する可能性だけある。
  print(f"///// now...{level_status}_{model_name}_local_study_start ! /////")
  print("trial.params:", trial.params)


  for i, local_study in enumerate(local_studies):
    print(f"\n///// now...{level_status}_{model_name}_local_study_{i+1}_fold_cv /////")
    cv_idx_tuple = next(cv_obj)
    local_study.optimize(lambda trial: local_opt(
            trial=trial,
            model_name=model_name, 
            opt_model_params=opt_model_params,
            opt_fit_params=opt_fit_params,
            cv_idx_tuple=cv_idx_tuple, tr_df=tr_df,
            level_status=level_status, params_obj=params_obj), n_trials=1)


    print("-------")
    print(local_study.trials[-1].value)
    print("-------")
    if "min_loss" in local_study.trials[-1].user_attrs:
      # これはskorchのみの処理になる
      loss = local_study.trials[-1].user_attrs["min_loss"]
    else:
      loss = local_study.trials[-1].value
    accum_loss += loss
  mean_loss = accum_loss / cv_n
  if mean_loss > global_opt_loss:
    cnt_patience_step += 1
    global_study.set_user_attr("cnt_patience_step", cnt_patience_step)
    if cnt_patience_step == patience_step_threshold:
      if level_status == "level_1":
        n_trials = params_obj.l1_n_trials
      else:
        n_trials = params_obj.l2_n_trials
      global_study.set_user_attr("total_trial", trial_number + 1)
      print(f"設定したn_trials{n_trials}に達することなくstudyを終了します。")
      print(f"{trial_number + 1}回目のトライアルで終了しました。")
      global_study.stop()
  else:
    global_study.set_user_attr("global_opt_loss", mean_loss)
  print(f"///// now...{level_status}_{model_name}_local_study_all_done ! /////")

  return mean_loss
