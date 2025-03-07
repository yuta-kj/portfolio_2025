from imports import (
  skorch,
  xgb
)

def set_model_params(model_name, phase, params_obj,
                     opt_model_params=None, trial=None):
  trgt_y = params_obj.trgt_y
  model_params = params_obj.params["model_params"]
  cate_cols = None
  if hasattr(params_obj, "inp_dim"):
    inp_dim = params_obj.inp_dim

  if hasattr(params_obj, "cate_cols"):
    cate_cols = params_obj.cate_cols
    cate_cols = [col for col in cate_cols if col not in ["id", trgt_y]]

  update_params = {}
  for k, v in model_params.items():
    update_params[k] = v
  for k, v in opt_model_params.items():
    update_params[k] = v
  model_params = update_params.copy()

  if model_name == "neuralnetreg":
    model_params["module__input_dim"] = inp_dim
    model_params["callbacks"] = []
    if phase == "opt_phase":
      model_params["callbacks"].append(
            skorch.callbacks.EarlyStopping(monitor='valid_loss',
              patience=params_obj.early_rounds, threshold=0.0001,
              threshold_mode='rel', lower_is_better=True, load_best=False))
      
    if phase == "best_phase":
      model_params["callbacks"].append(
        skorch.callbacks.EarlyStopping(monitor='valid_loss',
          patience=params_obj.early_rounds, threshold=0.0001,
          threshold_mode='rel', lower_is_better=True, load_best=False))

  if model_name in ["gnb", "rfreg", "svr"]:
    if phase == "opt_phase":
      pass
      
    if phase == "best_phase":
      pass

  if model_name == "xgbreg":
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
    # https://xgboost.readthedocs.io/en/stable/parameter.html#xgboost-parameters

    model_params["callbacks"] = []
    # sklearnだと機能しない↓
    # model_params["callbacks"].append(xgb.callback.EvaluationMonitor(rank=10, period=100, show_stdv=False))
    
    if phase == "opt_phase":
      model_params["eval_metric"] = list(params_obj.params["model_params"]["eval_metric"])
      model_params["callbacks"].append(
        xgb.callback.EarlyStopping(rounds=params_obj.early_rounds,
                                     metric_name=model_params["eval_metric"][0],
                                     data_name="validation_0", maximize=False, save_best=True, min_delta=0.0))

    if phase == "best_phase":
      model_params["eval_metric"] = list(params_obj.params["model_params"]["eval_metric"])
      model_params["callbacks"].append(
                xgb.callback.EarlyStopping(rounds=params_obj.early_rounds,
                                           metric_name=model_params["eval_metric"][0],
                                           data_name="validation_0", maximize=False, save_best=True, min_delta=0.0))
    
  if model_name == "lgbreg":
    pass
    if phase == "opt_phase":
      pass
    if phase == "best_phase":
      pass

  if model_name == "catreg":
    # https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
    # https://catboost.ai/docs/en/references/training-parameters/
    # https://catboost.ai/docs/en/concepts/loss-functions
    model_params["cat_features"] = cate_cols
    model_params["early_stopping_rounds"] = params_obj.early_rounds
    model_params["eval_metric"] = params_obj.params["model_params"]["eval_metric"]

  return model_params


