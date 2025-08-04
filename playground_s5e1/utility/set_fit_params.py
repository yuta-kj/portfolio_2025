from imports import (
  skorch,
  xgb,
  lgb
)


def set_fit_params(model_name, phase, params_obj,
                   opt_fit_params=None, trial=None):
  trgt_y = params_obj.trgt_y
  fit_params = params_obj.params["fit_params"]
  update_params = {}
  for k, v in fit_params.items():
    update_params[k] = v
  if len(opt_fit_params) > 0:
    for k, v in opt_fit_params.items():
      update_params[k] = v
  fit_params = update_params.copy()

  cate_cols = None
  if hasattr(params_obj, "cate_cols"):
    cate_cols = params_obj.cate_cols
    cate_cols = [col for col in cate_cols if col not in ["id", trgt_y]]
  if model_name in ["xgbreg", "lgbreg", "catreg"]:
    X_tr = params_obj.X_tr
    y_tr = params_obj.y_tr
    X_eval = params_obj.X_eval
    y_eval = params_obj.y_eval
    fit_params["eval_set"] = [(X_tr, y_tr), (X_eval, y_eval)]

    if model_name == "xgbreg":
      if phase == "opt_phase":
        pass
      if phase == "best_phase":
        pass

    if model_name == "lgbreg":
    #   https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
      fit_params["callbacks"] = []
      fit_params["categorical_feature"] = cate_cols
      if phase == "opt_phase":
        fit_params["callbacks"].append(
                  lgb.early_stopping(stopping_rounds=params_obj.early_rounds,
                    first_metric_only=False, verbose=True, min_delta=0.0))

      if phase == "best_phase":
        fit_params["callbacks"].append(
                  lgb.early_stopping(stopping_rounds=params_obj.early_rounds,
                    first_metric_only=False, verbose=True, min_delta=0.0))
        
    if model_name == "catreg":
      if phase == "opt_phase":
        pass
      if phase == "best_phase":
        pass
  return fit_params
