from imports import (
  torch,

)

class OptSearchMap:
  def __init__(self, trial, model_name, params_obj, cnf_yml):
    self.model_name = model_name
    self.trial = trial
    if hasattr(params_obj, "depth"):
      self.depth = params_obj.depth
    if hasattr(params_obj, "inp_dim"):
      self.inp_dim = params_obj.inp_dim
    self.n_layers_lst = [1, 2, 3]
    self.cnf_yml = cnf_yml

  def neuralnetreg_search_params(self, trial):
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
    # https://pystyle.info/pytorch-adagrad-rmsprop-adadelta/#index_id1
    optimizer_name = trial.suggest_categorical('optimizer_name', ["Adam", "Adagrad", "RMSprop", "RAdam"])
    if optimizer_name == "Adam":
      # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
      lr = trial.suggest_float('lr', self.cnf_yml.neuralnetreg.opt_model_params.lr_low, self.cnf_yml.neuralnetreg.opt_model_params.lr_high, log=True)
      betas = (trial.suggest_float('betas1', self.cnf_yml.neuralnetreg.opt_model_params.betas1_low, self.cnf_yml.neuralnetreg.opt_model_params.betas1_high),
                trial.suggest_float('betas2', self.cnf_yml.neuralnetreg.opt_model_params.betas2_low, self.cnf_yml.neuralnetreg.opt_model_params.betas2_high))
      optimizer = torch.optim.Adam
      optimizer_params = {
          'optimizer__lr': lr,
          'optimizer__betas': betas
      }

    elif optimizer_name == "RAdam":
      # https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam
      lr = trial.suggest_float('lr', self.cnf_yml.neuralnetreg.opt_model_params.lr_low, self.cnf_yml.neuralnetreg.opt_model_params.lr_high, log=True)
      betas = (trial.suggest_float('betas1', self.cnf_yml.neuralnetreg.opt_model_params.betas1_low, self.cnf_yml.neuralnetreg.opt_model_params.betas1_high),
                trial.suggest_float('betas2', self.cnf_yml.neuralnetreg.opt_model_params.betas1_low, self.cnf_yml.neuralnetreg.opt_model_params.betas1_high))
      optimizer = torch.optim.RAdam
      optimizer_params = {
          'optimizer__lr': lr,
          'optimizer__betas': betas
      }

    elif optimizer_name == "Adagrad":
      # https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
      lr = trial.suggest_float('lr', self.cnf_yml.neuralnetreg.opt_model_params.lr_low, self.cnf_yml.neuralnetreg.opt_model_params.lr_high, log=True)
      lr_decay = trial.suggest_float('lr_decay',  self.cnf_yml.neuralnetreg.opt_model_params.lr_decay_low, self.cnf_yml.neuralnetreg.opt_model_params.lr_decay_high)
      optimizer = torch.optim.Adagrad
      optimizer_params = {
          'optimizer__lr': lr,
          'optimizer__lr_decay': lr_decay
      }

    elif optimizer_name == "RMSprop":
      # https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
      lr = trial.suggest_float('lr', self.cnf_yml.neuralnetreg.opt_model_params.lr_low, self.cnf_yml.neuralnetreg.opt_model_params.lr_high, log=True)
      alpha = trial.suggest_float('alpha', self.cnf_yml.neuralnetreg.opt_model_params.alpha_low, self.cnf_yml.neuralnetreg.opt_model_params.alpha_high, log=True)
      weight_decay = trial.suggest_float('weight_decay', self.cnf_yml.neuralnetreg.opt_model_params.weight_decay_low, self.cnf_yml.neuralnetreg.opt_model_params.weight_decay_high)
      optimizer = torch.optim.RMSprop
      optimizer_params = {
        'optimizer__lr': lr,
        'optimizer__alpha': alpha,
        'optimizer__weight_decay': weight_decay
      }

    activation_name = trial.suggest_categorical('activation_name', ['LeakyReLU', 'ReLU', 'ReLU6'])

    if activation_name == 'LeakyReLU':
      activation = torch.nn.LeakyReLU
    elif activation_name == 'ReLU':
      activation = torch.nn.ReLU
    elif activation_name == 'ReLU6':
      activation = torch.nn.ReLU6

    n_layers=trial.suggest_categorical("n_layers", self.cnf_yml.neuralnetreg.opt_model_params.n_layers_lst)

    opt_model_params = {
      "module__activation": activation,
      "optimizer": optimizer,
      **optimizer_params
    }

    hidden_dim_lst = []
    for i in range(n_layers):
      i = i + 1
      _hidden_dim = trial.suggest_int(f"hidden_dim_{i}", low=self.cnf_yml.neuralnetreg.opt_model_params.hidden_dim_low, high=self.cnf_yml.neuralnetreg.opt_model_params.hidden_dim_high, log=True)
      hidden_dim_lst.append(_hidden_dim)

    opt_model_params["module__hidden_dim_lst"] = hidden_dim_lst
    opt_params = {"opt_mpdel_params": opt_model_params}

    return opt_params

  def rfreg_search_params(self, trial):
    max_depth = self.depth
    opt_params = {
      "opt_model_params":{
        "max_depth": trial.suggest_int("n_estimators", low=max_depth//2, high=max_depth, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", low=10, high=50, log=False),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", low=10, high=100, log=False),
        "ccp_alpha": trial.suggest_float("ccp_alpha", low=0.1, high=0.5, log=False, step=0.05)
        },
      "opt_fit_params": {}
    }

    return opt_params


  def xgbreg_search_params(self, trial):
    max_depth = self.depth
    opt_params = {
      "opt_model_params":{
        "max_depth": trial.suggest_int("max_depth", low=3, high=max_depth, log=False),
        "max_leaves": trial.suggest_int("max_leaves", low=self.cnf_yml.xgbreg.opt_model_params.max_leaves_low, high=self.cnf_yml.xgbreg.opt_model_params.max_leaves_high, log=False),
        "learning_rate": trial.suggest_float("learning_rate", low=self.cnf_yml.xgbreg.opt_model_params.learning_rate_low, high=self.cnf_yml.xgbreg.opt_model_params.learning_rate_high, log=False),
        "reg_alpha": trial.suggest_float("reg_alpha", low=self.cnf_yml.xgbreg.opt_model_params.reg_alpha_low, high=self.cnf_yml.xgbreg.opt_model_params.reg_alpha_high, log=False, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", low=self.cnf_yml.xgbreg.opt_model_params.reg_lambda_low, high=self.cnf_yml.xgbreg.opt_model_params.reg_lambda_high, log=False, step=0.1),
        "subsample": trial.suggest_float("subsample", low=self.cnf_yml.xgbreg.opt_model_params.subsample_low, high=self.cnf_yml.xgbreg.opt_model_params.subsample_high, log=False, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", low=self.cnf_yml.xgbreg.opt_model_params.colsample_bytree_low, high=self.cnf_yml.xgbreg.opt_model_params.colsample_bytree_high, log=False, step=0.1)
      },
      "opt_fit_params": {}
    }

    return opt_params

  def lgbreg_search_params(self, trial):
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    # https://lightgbm.readthedocs.io/en/latest/Parameters.html

    max_depth = self.depth
    opt_params = {
      "opt_model_params":{
        "max_depth": trial.suggest_int("max_depth", low=max_depth//2, high=max_depth, log=True),
        "num_leaves": trial.suggest_int("num_leaves", low=round(2**(max_depth)*0.7), high=round(2**(max_depth)*0.7)*5, log=True),
        "learning_rate": trial.suggest_float("learning_rate", low=self.cnf_yml.lgbreg.opt_model_params.learning_rate_low, high=self.cnf_yml.lgbreg.opt_model_params.learning_rate_high, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", low=self.cnf_yml.lgbreg.opt_model_params.reg_alpha_low, high=self.cnf_yml.lgbreg.opt_model_params.reg_alpha_high, log=False, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", low=self.cnf_yml.lgbreg.opt_model_params.reg_lambda_low, high=self.cnf_yml.lgbreg.opt_model_params.reg_lambda_high, log=False, step=0.1),
        "subsample": trial.suggest_float("subsample", low=self.cnf_yml.lgbreg.opt_model_params.subsample_low, high=self.cnf_yml.lgbreg.opt_model_params.subsample_high, log=False, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", low=self.cnf_yml.lgbreg.opt_model_params.colsample_bytree_low, high=self.cnf_yml.lgbreg.opt_model_params.colsample_bytree_high, log=False, step=0.1)
      },
      "opt_fit_params":{}
    }

    return opt_params

  def catreg_search_params(self, trial):
    # https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier

    max_depth = self.depth
    opt_params = {
      "opt_model_params": {
        "max_depth": trial.suggest_int("max_depth", low=max_depth//2, high=max_depth, log=True),
        "learning_rate": trial.suggest_float("learning_rate", low=self.cnf_yml.catreg.opt_model_params.learning_rate_low, high=self.cnf_yml.catreg.opt_model_params.learning_rate_high, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", low=self.cnf_yml.catreg.opt_model_params.l2_leaf_reg_low, high=self.cnf_yml.catreg.opt_model_params.l2_leaf_reg_high, log=False, step=0.1),
        "subsample": trial.suggest_float("subsample", low=self.cnf_yml.catreg.opt_model_params.subsample_low, high=self.cnf_yml.catreg.opt_model_params.subsample_high, log=False, step=0.1),
      },
      "opt_fit_params": {}
    }

    return opt_params

  @property
  def opt_params(self):
    print(self.model_name)
    if self.model_name == "neuralnetreg":
      opt_params = self.neuralnetreg_search_params(trial=self.trial)
    if self.model_name == "gnb":
      opt_params = self.gnb_search_params(trial=self.trial)
    if self.model_name == "svr":
      opt_params = self.svr_search_params(trial=self.trial)
    if self.model_name == "rfreg":
      opt_params = self.rfreg_search_params(trial=self.trial)
    if self.model_name == "xgbreg":
      opt_params = self.xgbreg_search_params(trial=self.trial)
    if self.model_name == "lgbreg":
      opt_params = self.lgbreg_search_params(trial=self.trial)
    if self.model_name == "catreg":
      opt_params = self.catreg_search_params(trial=self.trial)
    
    return opt_params