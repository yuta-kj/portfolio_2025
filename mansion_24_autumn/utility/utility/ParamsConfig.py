from imports import (
  NeuralNetRegressor,
  XGBRegressor,
  LGBMRegressor,
  CatBoostRegressor,
  SVR,
  RandomForestRegressor,
  torch
)

from CustomNeuralNetReg import CustomNeuralNetReg



class ParamsConfig:
  def __init__(self, model_name, cnf_yml):
    self.model_name = model_name
    self.model_map = {
      "neuralnetreg": NeuralNetRegressor,
    #   "gnb": GaussianNB,
      "svr": SVR,
      "rfreg": RandomForestRegressor,
      "xgbreg": XGBRegressor,
      "lgbreg": LGBMRegressor,
      "catreg": CatBoostRegressor}

    if "s_n_trials" in cnf_yml.common:
      self.s_n_trials = cnf_yml.common.s_n_trials
    # 50
    if "l1_n_trials" in cnf_yml.common:
      self.l1_n_trials = cnf_yml.common.l1_n_trials
    # 50
    if "l2_n_trials" in cnf_yml.common:
      self.l2_n_trials = cnf_yml.common.l2_n_trials
    # 50
    self.epochs = cnf_yml.common.epochs
    # 20
    self.piriod = cnf_yml.common.piriod
    # 10
    self.depth = cnf_yml.common.depth
    # 5
    self.n_jobs = cnf_yml.common.n_jobs
    # 2
    self.cv_n = cnf_yml.common.cv_n
    # 3
    self.early_rounds = cnf_yml.common.early_rounds
    self.patience_step_threshold = cnf_yml.common.patience_step_threshold
    self.cnf_yml = cnf_yml

  @property
  def set_config(self):
    if self.model_name == "neuralnetreg":
      self.model = NeuralNetRegressor
      self.params = {
        "model_params": {
          "lr": self.cnf_yml.neuralnetreg.lr,
          "max_epochs": self.epochs,
          "batch_size": self.cnf_yml.neuralnetreg.batch_size,
          "module": CustomNeuralNetReg,
          "module__input_dim": self.cnf_yml.neuralnetreg.input_dim,
          "module__hidden_dim_lst": self.cnf_yml.neuralnetreg.hidden_dim_lst,
          "module__activation": torch.nn.ReLU,
          "criterion": torch.nn.BCELoss,
          "optimizer": torch.optim.Adam,
          "device": self.cnf_yml.neuralnetreg.device,
          "verbose": self.piriod},
        "fit_params": {}
      }

    if self.model_name == "svr":
      self.model = SVR
      self.params = {
        "model_params": {
          "C": 0.3,
          "kernel": 'rbf',
          "degree": 3,
          "gamma": 'scale',
          "coef0": 0.0,
          "tol": 1e-4, #収束条件
          "verbose": False,
          "max_iter": self.epochs},
        "fit_params": {}
        }

    if self.model_name == "rfreg":
      self.model = RandomForestRegressor
      self.params = {
        "model_params": {
          "n_estimators": self.epochs, #100
          "criterion": "gini",
          "max_depth": self.depth,
          "ccp_alpha": 0.3,
          "bootstrap": True,
          "oob_score": True,
          "max_features": 'sqrt',
          "n_jobs": self.n_jobs,
          "random_state": 42,
          "verbose": self.piriod},
        "fit_params": {}
      }

    if self.model_name == "xgbreg":
      self.model = XGBRegressor
      self.params = {
        "model_params": {
            "n_estimators": self.epochs,
            "max_depth": self.depth,
            "max_leaves": round(2**(self.depth)*0.7),
            "learning_rate": 0.1*(1+0.5*((100-self.depth+1e-3)/100)),
            "subsample": self.cnf_yml.xgbreg.model_params.subsample,
            "colsample_bytree": self.cnf_yml.xgbreg.model_params.colsample_bytree,
            "objective": self.cnf_yml.xgbreg.model_params.objective,
            "grow_policy": self.cnf_yml.xgbreg.model_params.grow_policy,
            "n_jobs": self.n_jobs,
            "booster": self.cnf_yml.xgbreg.model_params.booster,
            "verbosity": self.cnf_yml.xgbreg.model_params.verbosity,
            "device": self.cnf_yml.xgbreg.model_params.device,
            "reg_alpha": self.cnf_yml.xgbreg.model_params.reg_alpha,
            "reg_lambda": self.cnf_yml.xgbreg.model_params.reg_lambda,
            "enable_categorical": self.cnf_yml.xgbreg.model_params.enable_categorical,
            "eval_metric": self.cnf_yml.xgbreg.model_params.eval_metric
      },
      "fit_params":{
        "verbose": 1000
      }
      }
    
    if self.model_name == "lgbreg":
      self.model = LGBMRegressor
      self.params = {
        "model_params":{
            "num_leaves": round(2**(self.depth)*0.7), # 2^(mad_depth)* 0.7
            "max_depth": self.depth,
            "learning_rate": 0.1*(1+0.5*((100-self.epochs+1e-3)/100)),
            "n_estimators": self.epochs, #100
            "subsample": self.cnf_yml.lgbreg.model_params.subsample,
            "reg_alpha": self.cnf_yml.lgbreg.model_params.reg_alpha,
            "reg_lambda": self.cnf_yml.lgbreg.model_params.reg_lambda,
            "boosting_type": self.cnf_yml.lgbreg.model_params.boosting_type,
            "objective": self.cnf_yml.lgbreg.model_params.objective,
            "n_jobs": self.n_jobs,
            "device": self.cnf_yml.lgbreg.model_params.device,
            "verbose": self.cnf_yml.lgbreg.model_params.verbose,
        },
        "fit_params":{
            "eval_metric": self.cnf_yml.lgbreg.fit_params.eval_metric,
            "eval_names": self.cnf_yml.lgbreg.fit_params.eval_names
        }
      }

    if self.model_name == "catreg":
      self.model = CatBoostRegressor
      self.params = {
        "model_params": {
            "n_estimators": self.epochs,
            "learning_rate": 0.1*(1+0.5*((100-self.epochs+1e-3)/100)),
            "max_depth": self.depth,
            "l2_leaf_reg": self.cnf_yml.catreg.model_params.l2_leaf_reg,
            "loss_function": self.cnf_yml.catreg.model_params.loss_function,
            "task_type": self.cnf_yml.catreg.model_params.task_type,
            "eval_metric": self.cnf_yml.catreg.model_params.eval_metric
        },
        "fit_params":{}
        }
      
    return self


# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
# https://xgboost.readthedocs.io/en/stable/parameter.html#xgboost-parameters
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://catboost.ai/en/docs/concepts/python-reference_catboostRegressor
