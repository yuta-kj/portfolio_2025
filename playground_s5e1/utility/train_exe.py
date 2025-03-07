from imports import (
  np,
  ce,
  pd,
  mean_squared_error,
  mean_absolute_error,
  mean_absolute_percentage_error
)

from set_model_params import set_model_params
from set_fit_params import set_fit_params


def cnvrt_to_int(df):
  for col in df.columns:
    if df[col].dtype == "category":
      df[col] = df[col].astype("int")
      df[col] = df[col].apply(lambda x: int(x))
  return df


def one_hot_encoder(tr_df, test_df, nominal_cols, trgt_y=None):
  ce_one_hot_cols = [col for col in nominal_cols if col not in ["id", trgt_y]]
  tmp_tr_df = tr_df.copy()
  tmp_tr_df = cnvrt_to_int(tmp_tr_df)
  tmp_tr_df_2 = tmp_tr_df[ce_one_hot_cols]
  one_hot_en = ce.one_hot.OneHotEncoder(cols=ce_one_hot_cols)
  one_hot_en = one_hot_en.fit(tmp_tr_df_2)
  ce_tr_df = one_hot_en.transform(tmp_tr_df_2)
  not_ce_one_hot_cols = [x for x in tmp_tr_df.columns if x not in ce_one_hot_cols]
  tmp_tr_df = tmp_tr_df[not_ce_one_hot_cols]
  not_ce_one_hot_cols = [x for x in not_ce_one_hot_cols if x not in [trgt_y]]
  tmp_tr_df = pd.concat([tmp_tr_df, ce_tr_df], axis=1)

  if test_df is not None:
    tmp_test_df = test_df.copy()
    tmp_test_df = cnvrt_to_int(tmp_test_df)
    tmp_test_df_2 = tmp_test_df[ce_one_hot_cols]
    ce_test_df = one_hot_en.transform(tmp_test_df_2)
    tmp_test_df = tmp_test_df[not_ce_one_hot_cols]
    tmp_test_df = pd.concat([tmp_test_df, ce_test_df], axis=1)
  else:
    tmp_test_df = None

  return tmp_tr_df, tmp_test_df


def train_exe(model_name, phase, cv_idx_tuple, params_obj,
              opt_model_params=None, opt_fit_params=None,
              tr_df=None, test_df=None,
              trial=None,  acc_local_loss=[]):

  trgt_model = params_obj.model
  trgt_y = params_obj.trgt_y

  if model_name in ["neuralnetreg", "gnb", "rfreg", "svr"]:
      inp_dim = tr_df.drop(["id", trgt_y], axis=1).shape[1]
      params_obj.inp_dim = inp_dim


  tr_idx = cv_idx_tuple[0]
  tr_cv_df = tr_df.iloc[tr_idx]
  y_tr = tr_cv_df[trgt_y]
  X_tr = tr_cv_df.drop(["id", trgt_y], axis=1)
  val_idx = cv_idx_tuple[1]
  val_cv_df = tr_df.iloc[val_idx]
  y_val = val_cv_df[trgt_y]
  X_val = val_cv_df.drop(["id", trgt_y], axis=1)

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
    X_eval = eval_df.drop(["id", trgt_y], axis=1)
    tr_idx = list(set(tr_df.index) - set(eval_set_idx))
    tr_df = tr_df.iloc[tr_idx]
    y_tr = tr_df[trgt_y]
    X_tr = tr_df.drop(["id", trgt_y], axis=1)

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
    test_df = test_df.drop("id", axis=1).copy()
    if model_name not in ["lgbreg", "xgbreg", "catreg"]:
        test_df = test_df.values.astype(np.float32)
    test_pred = reg.predict(test_df)
    return loss, model_params, fit_params, val_pred, test_pred

