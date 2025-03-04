from imports import (
  re,
  datetime,
  pd,
  optuna,
  KFold,
  pickle,
  np
)

from ParamsConfig import ParamsConfig
from cnvrt_best_param_for_nnet import cnvrt_best_param_for_nnet
from train_exe import train_exe
from global_opt import global_opt


class FitModels:
  def __init__(self, tr_df, test_df, trgt_y,
                base_path, cnf_yml, output_dir_obj, wandb_run=None,
                drop_cols=[], exclude=None,
                make_new_exam=True, use_latest_exam=False, use_single=False):

    self.tr_df = tr_df
    self.test_df = test_df
    self.cnf_yml = cnf_yml 
    tr_df_used_len = self.cnf_yml.common.tr_df_used_len
    test_df_used_len = self.cnf_yml.common.test_df_used_len
    if isinstance(tr_df_used_len, int):
      self.tr_df = self.tr_df[:tr_df_used_len].copy()
    if isinstance(test_df_used_len, int):
      self.test_df = self.test_df[:test_df_used_len].copy()
    self.trgt_y = trgt_y
    self.drop_cols = drop_cols
    self.tmp_exclude_cols = ["id", self.trgt_y]
    self.base_path = base_path
    self.cnf_yml = cnf_yml
    self.output_dir_obj = output_dir_obj
    self.wandb_run = wandb_run
    self.load_if_exists = True
    
    self.use_single_model = self.cnf_yml.common.use_single_model
    if not self.use_single_model:
      self.use_lv1_models = self.cnf_yml.common.use_lv1_models
      self.use_lv2_model = self.cnf_yml.common.use_lv2_model

    # ↓trail回数が一回で十分なモデル
    self.no_opt_model = ["gnb"]
    self.best_params_stock = {}
    self.exclude_models = None
    
    if not self.use_single_model:
        pred_cols = [f"{m}_pred" for m in self.use_lv1_models] #各ベースモデルの予測値をメタモデルの入力とする
        target_col = [f"{self.trgt_y}"]
        meta_tr_df = pd.DataFrame(columns=["id"] + pred_cols + target_col)
        meta_tr_df["id"] = self.tr_df["id"]
        meta_tr_df[target_col] = self.tr_df[target_col]
        self.meta_tr_df = meta_tr_df

        meta_test_df = pd.DataFrame(columns=["id"] + pred_cols)
        meta_test_df["id"] = self.test_df["id"]
        self.meta_test_df = meta_test_df
    
    date = datetime.date.today()
    self.exam_year = date.year
    self.exam_month = date.month
    self.exam_day = date.day
    self._make_new_exam = make_new_exam
    self._use_latest_exam = use_latest_exam
    self.random_state = self.cnf_yml.common.random_state
    self._check_dtypes = {}

  def exclude_cols(self, df, drop_cols):
    df_cols = df.columns.tolist()
    print(df_cols)
    print(drop_cols)
    for col in drop_cols:
      if col in df_cols:
        df = df.drop(col, axis=1)
    return df

  def trail_state_callback(self, study, trial):
    print(f'{study.user_attrs["model_name"]}_TRIAL_{trial.number+1} done...')

  def hypara_search(self, model_name, tr_df, n_trials, level_status,
                    random_state, params_obj=None):

    level_status = level_status
    if model_name in self.no_opt_model:
      #　ハイパラ探索をする必要がないので一回でOK。
      # n_trials = 1
      print(f"{model_name}はハイパラ探索しません。")
      return {}
    print(f"///// now... {level_status} //////")
    print(f"///// search for {model_name} /////")
    study_name = f'study_ens_{level_status}_{model_name}'
    file_name = study_name + '.db'
    save_path = self.output_dir_obj / file_name
    # globalobjectiveではprunerは設定しなくてよい
    # wrapped_pruner = optuna.pruners.HyperbandPruner()
    # pruner = optuna.pruners.PatientPruner(wrapped_pruner=wrapped_pruner, patience=1) if CONFIG_MAP[model_name].USE_OPTUNA_PRUNER is True else None
    sampler = optuna.samplers.TPESampler(seed=random_state)

    self.global_study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///" + str(save_path),
        load_if_exists = self.load_if_exists,
        direction="minimize",
        pruner=None,
        sampler=sampler)

    self.global_study.set_user_attr("model_name", f"{model_name}")
    callbacks = [self.trail_state_callback]

    # global_objectiveはcvのmean_lossを返す
    global_obj_loss = float('inf')
    cnt_patience_step = 0
    self.global_study.optimize(lambda trial: global_opt(
                    trial=trial, global_study=self.global_study,
                    model_name=model_name, tr_df=tr_df,
                    level_status=level_status, params_obj=params_obj,
                    global_obj_loss=global_obj_loss,
                    cnt_patience_step=cnt_patience_step,
                    cnf_yml=self.cnf_yml,
                    random_state=random_state),
                    n_trials=n_trials, callbacks=callbacks)

    best_model_params = self.global_study.user_attrs["opt_model_params"]
    best_fit_params = self.global_study.user_attrs["opt_fit_params"]

    return best_model_params, best_fit_params


  def fit_level_one_models(self):
    level_status = "level_1"
    print("///// レベル1の学習を開始します /////")
    print("///// 使用する全てのベースモデル(LV1)は以下の通りです /////")
    for model_name in self.use_lv1_models:
      print(f"\t///// {model_name} /////")

    for model_name in  self.use_lv1_models:
      self.meta_tr_df[f"{model_name}_pred"] = pd.NA
      self.meta_test_df[f"{model_name}_pred"] = pd.NA
      print(f"///// {model_name}の学習をSTART ! /////")
      tr_df, test_df = self.tr_df.copy(), self.test_df.copy()
      self.level_status = "level_1"
      params_obj = ParamsConfig(model_name, self.cnf_yml).set_config
      params_obj.trgt_y = self.trgt_y
      phase = "opt_phase"
      params_obj.phase = phase
      l1_n_trials = params_obj.l1_n_trials
      cv_n = params_obj.cv_n
      
      best_model_params, best_fit_params = self.hypara_search(
        model_name, tr_df=tr_df, n_trials=l1_n_trials, level_status=level_status,
        random_state=self.random_state, params_obj=params_obj)

      phase = "best_phase"
      params_obj.phase = phase
      # パラメータの名前の微調整
      if model_name == "neuralnetreg":
        best_model_params = cnvrt_best_param_for_nnet(best_model_params)

      self.best_params_stock[f"{model_name}_{self.level_status}"] = {
        "best_model_params": best_model_params,
        "best_fit_params": best_fit_params
      }
    #   print(f"///// Best params: {best_params} /////\n")
      print("///// model with optimal parameters 訓練中... //////\n")

      cv = KFold(n_splits=cv_n, shuffle=True, random_state=self.random_state)
      cv_obj = cv.split(tr_df)
      tmp_test_pred_lst = []
      
      for i , cv_idx_tuple in enumerate(cv.split(tr_df)):
        print(f"///// LV1_{model_name}_model_with_best_params /////")
        print(f"///// {i+1}_fold_cv... /////")
        _val_idx = cv_idx_tuple[1]
        _, model_params, fit_params, val_pred, test_pred = train_exe(
                                            model_name=model_name,
                                            phase=phase,
                                            cv_idx_tuple=cv_idx_tuple,
                                            params_obj=params_obj,
                                            opt_model_params=best_model_params,
                                            opt_fit_params=best_fit_params,
                                            tr_df=tr_df,
                                            test_df=test_df,
                                            trial=None
                                            )
        
        self.l1_model_params = model_params
        self.l1_fit_params = fit_params 
        val_pred = pd.DataFrame(val_pred)[0].values
        test_pred = pd.DataFrame(test_pred)[0].values
        
        self.meta_tr_df.loc[_val_idx, f"{model_name}_pred"] = val_pred
        self.meta_tr_df[f"{model_name}_pred"] = pd.to_numeric(self.meta_tr_df[f"{model_name}_pred"], errors="coerce")
        tmp_test_pred_lst.append(test_pred)
      test_preds = np.column_stack(tmp_test_pred_lst)
      cv_mean_test_preds = test_preds.mean(axis=1)
      print(cv_mean_test_preds) 
      self.meta_test_df[f"{model_name}_pred"] = cv_mean_test_preds
      print("\tDone!\n")

  def fit_level_two_model(self):
    print("/////  Lv2 model 訓練中... //////")
    print(f"{self.use_lv2_model} を使用しています...")
    model_name = self.use_lv2_model
    tr_df = self.meta_tr_df.copy()
    test_df = self.meta_test_df.copy()
    self.level_status = "level_2"
    params_obj = ParamsConfig(model_name, self.cnf_yml).set_config
    phase = "opt_phase"
    params_obj.phase = phase
    params_obj.trgt_y = self.trgt_y
    l2_n_trials = params_obj.l2_n_trials
    cv_n = params_obj.cv_n

    print("\toptimal hyperparameters using Optuna　探索中...")

    best_model_params, best_fit_params = self.hypara_search(
      model_name= model_name,
      tr_df=tr_df,
      n_trials=l2_n_trials,
      level_status=self.level_status,
      random_state=self.random_state,
      params_obj=params_obj)

    if model_name == "neuralnetreg":
      best_model_params = self.cnvrt_best_param_for_nnet(best_model_params)

    self.best_params_stock[f"{model_name}_{self.level_status}"] = {
        "best_model_params": best_model_params,
        "best_fit_params": best_fit_params
      }
    # print(f"\n\tBest params: {best_params}\n")

    print("\tmodel with optimal parameters 訓練中...\n")


    cv =  KFold(n_splits=cv_n, shuffle=True, random_state=self.random_state)
    cv_obj = cv.split(tr_df)
    tmp_test_pred_lst = []
    params_obj.phase = "best_phase"
    phase = params_obj.phase
    for i , cv_idx_tuple in enumerate(cv.split(tr_df)):
      print(f"///// LV2_{model_name}_model_with_best_params /////")
      print(f"///// {i+1}_fold_cv... /////")
      _val_idx = cv_idx_tuple[1]
      _, model_params, fit_params, val_pred, test_pred = train_exe(
                                      model_name=model_name,
                                      phase=phase,
                                      cv_idx_tuple=cv_idx_tuple,
                                      params_obj=params_obj,
                                      opt_model_params=best_model_params,
                                      opt_fit_params=best_fit_params,
                                      tr_df=tr_df,
                                      test_df=test_df,
                                      trial=None
                                      )
      
      self.l2_model_params = model_params
      self.l2_fit_params = fit_params
      val_pred = pd.DataFrame(val_pred)[0].values
      test_pred = pd.DataFrame(test_pred)[0].values
      self.meta_tr_df.loc[_val_idx, f"{self.trgt_y}_pred"] = val_pred
      tmp_test_pred_lst.append(test_pred)

    test_preds = np.column_stack(tmp_test_pred_lst)
    cv_mean_test_preds = test_preds.mean(axis=1)
    self.meta_test_df[f"{self.trgt_y}_pred"] = cv_mean_test_preds
    
  def fit_single_model(self):
    print("/////  single model 訓練中... //////")
    print(f"{self.use_single_model} を使用しています...")
    model_name = self.use_single_model
    tr_df = self.tr_df.copy()
    test_df = self.test_df.copy()
    self.level_status = "single_only"
    params_obj = ParamsConfig(model_name, self.cnf_yml).set_config
    phase = "opt_phase"
    params_obj.phase = phase
    params_obj.trgt_y = self.trgt_y
    s_n_trials = params_obj.s_n_trials
    cv_n = params_obj.cv_n

    print("\toptimal hyperparameters using Optuna　探索中...")

    best_model_params, best_fit_params = self.hypara_search(
        model_name=model_name,
        tr_df=tr_df,
        n_trials=s_n_trials,
        level_status=self.level_status,
        random_state=self.random_state,
        params_obj=params_obj)

    if model_name == "neuralnetreg":
      best_model_params = self.cnvrt_best_param_for_nnet(best_model_params)

    self.best_params_stock[f"{model_name}_{self.level_status}"] = {
        "best_model_params": best_model_params,
        "best_fit_params": best_fit_params
      }
    # print(f"\n\tBest params: {best_params}\n")

    print("\tmodel with optimal parameters 訓練中...\n")


    cv =  KFold(n_splits=cv_n, shuffle=True, random_state=self.random_state)
    cv_obj = cv.split(tr_df)
    tmp_test_pred_lst = []
    params_obj.phase = "best_phase"
    phase = params_obj.phase
    for i , cv_idx_tuple in enumerate(cv.split(tr_df)):
      print(f"///// SINGLE_{model_name}_model_with_best_params /////")
      print(f"///// {i+1}_fold_cv... /////")
      _val_idx = cv_idx_tuple[1]
      _, model_params, fit_params, val_pred, test_pred = train_exe(
                                      model_name=model_name,
                                      phase=phase,
                                      cv_idx_tuple=cv_idx_tuple,
                                      params_obj=params_obj,
                                      opt_model_params=best_model_params,
                                      opt_fit_params=best_fit_params,
                                      tr_df=tr_df,
                                      test_df=test_df,
                                      trial=None
                                      )
      
      self.s_model_params = model_params
      self.s_fit_params = fit_params
      val_pred = pd.DataFrame(val_pred)[0].values
      test_pred = pd.DataFrame(test_pred)[0].values
      self.tr_df.loc[_val_idx, f"{self.trgt_y}_pred"] = val_pred
      tmp_test_pred_lst.append(test_pred)

    test_preds = np.column_stack(tmp_test_pred_lst)
    cv_mean_test_preds = test_preds.mean(axis=1)
    self.test_df[f"{self.trgt_y}_pred"] = cv_mean_test_preds
    
  def main_exe(self):
    if len(self.drop_cols) > 0:
      self.tr_df = self.exclude_cols(self.tr_df, self.drop_cols)
      self.test_df = self.exclude_cols(self.test_df, self.drop_cols)

    if self.use_single_model:
      self.fit_single_model()
      return self.tr_df, self.test_df,\
             self.s_model_params, self.s_fit_params
      
    else:
      self.fit_level_one_models()
      self.fit_level_two_model()
      return self.meta_tr_df, self.meta_test_df,\
             self.l1_model_params, self.l2_fit_params,\
             self.l2_model_params, self.l2_fit_params