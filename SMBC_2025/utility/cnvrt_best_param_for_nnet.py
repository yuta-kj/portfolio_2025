from imports import (
  re,
  torch
)


def cnvrt_best_param_for_nnet(best_params):
  if "n_layers" in best_params.keys():
    # best_params = [key for key in best_params.keys() if key not in "n_layers"]
    del best_params["n_layers"]

  hidden_dim_lst = []
  tmp_best_params = best_params.copy()
  for key in best_params.keys():
    if re.match(r"hidden_dim_(\d)", key):
      hidden_dim = best_params[key]
      hidden_dim_lst.append(hidden_dim)
      del tmp_best_params[key]
  best_params = tmp_best_params
  best_params["module__hidden_dim_lst"] = hidden_dim_lst

  if "optimizer_name" in best_params.keys():
    if "Adam" == best_params["optimizer_name"]:
      best_params["optimizer"] = torch.optim.Adam
      betas1 = best_params["betas1"]
      betas2 = best_params["betas2"]
      betas = (betas1, betas2)
      best_params["optimizer__betas"] = betas
      best_params["optimizer__lr"] = best_params["lr"]
      best_params = {k: v for k, v in best_params.items() if k not in ["optimizer_name", "lr", "betas1", "betas2"]}

    elif "RAdam" == best_params["optimizer_name"]:
      best_params["optimizer"] = torch.optim.RAdam
      best_params["optimizer__lr"] = best_params["lr"]
      betas1 = best_params["betas1"]
      betas2 = best_params["betas2"]
      betas = (betas1, betas2)
      best_params["optimizer__betas"] = betas
      best_params = {k: v for k, v in best_params.items() if k not in ["optimizer_name", "lr", "betas1", "betas2"]}

    elif "Adagrad" == best_params["optimizer_name"]:
      best_params["optimizer"] = torch.optim.Adagrad
      best_params["optimizer__lr"] = best_params["lr"]
      best_params["optimizer__lr_decay"] = best_params["lr_decay"]
      best_params = {k: v for k, v in best_params.items() if k not in ["optimizer_name", "lr", "lr_decay"]}

    elif "RMSprop" == best_params["optimizer_name"]:
      best_params["optimizer"] = torch.optim.RMSprop
      best_params["optimizer__lr"] = best_params["lr"]
      best_params["optimizer__alpha"] = best_params["alpha"]
      best_params["optimizer__weight_decay"] = best_params["weight_decay"]
      best_params = {k: v for k, v in best_params.items() if k not in ["optimizer_name", "lr", "alpha", "weight_decay"]}


  if "activation_name" in best_params.keys():
    if "LeakyReLU" == best_params["activation_name"]:
      best_params["module__activation"] = torch.nn.LeakyReLU
    elif "ReLU" == best_params["activation_name"]:
      best_params["module__activation"] = torch.nn.ReLU
    elif "ReLU6" == best_params["activation_name"]:
      best_params["module__activation"] = torch.nn.ReLU6
    best_params = {k: v for k, v in best_params.items() if k not in ["activation_name"]}

  return best_params