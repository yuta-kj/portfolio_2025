from imports import (
  np,
  ce,
  pd,
  mean_squared_error,
  mean_absolute_error,
  mean_absolute_percentage_error,
  issparse
)

from set_model_params import set_model_params
from set_fit_params import set_fit_params


def train_exe(model_name, phase, cv_idx_tuple, params_obj,
              opt_model_params=None, opt_fit_params=None,
              tr_df=None, test_df=None,
              trial=None,  acc_local_loss=[]):

  trgt_model = params_obj.model
  trgt_y = params_obj.trgt_y

  if model_name in ["neuralnetreg", "gnb", "rfreg", "svr"]:
      inp_dim = tr_df.drop(params_obj.tmp_drop_cols, axis=1).shape[1]
      params_obj.inp_dim = inp_dim


  tr_idx = cv_idx_tuple[0]
  tr_cv_df = tr_df.iloc[tr_idx]
  y_tr = tr_cv_df[trgt_y]
#   X_tr = tr_cv_df.drop(["id", trgt_y], axis=1)
  X_tr = tr_cv_df.drop(params_obj.tmp_drop_cols, axis=1)
  val_idx = cv_idx_tuple[1]
  val_cv_df = tr_df.iloc[val_idx]
  y_val = val_cv_df[trgt_y]
#   X_val = val_cv_df.drop(["id", trgt_y], axis=1)
  X_val = val_cv_df.drop(params_obj.tmp_drop_cols, axis=1)

  # sklearn系の処理
  if model_name in ["neuralnetreg", "gnb", "rfreg", "svr"]:
    X_tr = X_tr.values.astype(np.float32)
    y_tr = y_tr.values.astype(np.float32).reshape(-1,1)
    X_val = X_val.values.astype(np.float32)
    y_val = y_val.values.astype(np.float32).reshape(-1)

  if model_name in ["xgbreg", "lgbreg", "catreg"]:
    eval_set_idx = np.random.choice(tr_idx, size=int(0.2 * len(tr_idx)), replace=False)
    eval_df = tr_df.iloc[eval_set_idx]
    y_eval = eval_df[trgt_y]
    # X_eval = eval_df.drop(["id", trgt_y], axis=1)
    X_eval = eval_df.drop(params_obj.tmp_drop_cols, axis=1)
    tr_idx = list(set(tr_df.index) - set(eval_set_idx))
    tr_df = tr_df.iloc[tr_idx]
    y_tr = tr_df[trgt_y]
    # X_tr = tr_df.drop(["id", trgt_y], axis=1)
    X_tr = tr_df.drop(params_obj.tmp_drop_cols, axis=1)
    cate_cols = tr_df.dtypes[tr_df.dtypes == "category"].keys().to_list()
    params_obj.cate_cols = cate_cols
    params_obj.X_tr = X_tr
    params_obj.y_tr = y_tr
    params_obj.X_eval = X_eval
    params_obj.y_eval = y_eval

  model_params = set_model_params(model_name, phase, params_obj,
                                  opt_model_params=opt_model_params, trial=trial)
  
  fit_params = set_fit_params(model_name, phase, params_obj,
                              opt_fit_params=opt_fit_params, trial=trial)
                              
  reg = trgt_model(**model_params)
  print(params_obj.params)
  if model_name == "lgbreg" and params_obj.params["model_params"]["device"] == "cuda":
    if issparse(X_tr):
      X_tr = X_tr.toarray()
  if len(fit_params) > 0:
    reg.fit(X=X_tr, y=y_tr, **fit_params)
  else:
    reg.fit(X=X_tr, y=y_tr)

  val_pred = reg.predict(X_val)
  if model_name == "neuralnetreg":
    val_pred = np.squeeze(val_pred, axis=1)

  loss = mean_absolute_percentage_error(y_val, val_pred)
  if phase == "opt_phase":
    print("loss____:", loss)

    return loss

  else:
    tmp_drop_cols = [x for x in params_obj.tmp_drop_cols if x != params_obj.trgt_y]
    test_df = test_df.drop(tmp_drop_cols, axis=1)
    if model_name not in ["lgbreg", "xgbreg", "catreg"]:
        test_df = test_df.values.astype(np.float32)
    test_pred = reg.predict(test_df)
    return loss, model_params, fit_params, val_pred, test_pred

