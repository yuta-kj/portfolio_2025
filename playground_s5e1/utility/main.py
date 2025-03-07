from imports import (
  DictConfig,
  OmegaConf,
  Path,
  hydra,
  pickle,
  sys,
  yaml
)

from FitModels import FitModels
from make_new_output_dir import make_new_output_dir

base_path = sys.argv[0]
base_path = base_path.replace("/utility/main.py", "")
# コマンドライン引数をパース
exp_id = sys.argv[1]
# Hydraがargvを処理しないように、必要な引数だけ残して削除
sys.argv = sys.argv[:1]

@hydra.main(version_base=None,
    config_path=f"{base_path}/exp/exp_{exp_id}/cfg_yamls/",
    config_name="params_config")

def main(cfg: DictConfig):
  global exp_id
  global base_path
  trgt_y = cfg.common.trgt_y
  base_path = Path(base_path)
  
  with open(base_path / f"data/cleaned_data/cleaned_tr_df_exp_{exp_id}.pickle", 'rb') as f:
    tr_df = pickle.load(f)
  with open(base_path / f"data/cleaned_data/cleaned_test_df_exp_{exp_id}.pickle", 'rb') as f:
    test_df = pickle.load(f)
    
  output_dir_obj = make_new_output_dir(base_path, exp_id)
  ens_regs = FitModels(tr_df, test_df, trgt_y, base_path, cfg, output_dir_obj, wandb_run=None)
  if cfg.common.use_single_model:
    tr_df, test_df, s_model_params, s_fit_params = ens_regs.main_exe()
  else:
    tr_df, test_df, l1_model_params, l1_fit_params, l2_model_params, l2_fit_params = ens_regs.main_exe()

  if cfg.common.use_single_model:
    with open(output_dir_obj / "tr_df.pickle", 'wb') as f:
      pickle.dump(tr_df, f)

    with open(output_dir_obj / "test_df.pickle", 'wb') as f:
      pickle.dump(test_df, f)
    
    with open(output_dir_obj / "s_model_params.yaml", "w", encoding="utf-8") as f:
      yaml.dump(s_model_params, f, allow_unicode=True, default_flow_style=False)

    with open(output_dir_obj / "s_fit_params.yaml", "w", encoding="utf-8") as f:
      yaml.dump(s_fit_params, f, allow_unicode=True, default_flow_style=False)
  
  else:
    with open(output_dir_obj / "meta_tr_df.pickle", 'wb') as f:
      pickle.dump(tr_df, f)

    with open(output_dir_obj / "meta_test_df.pickle", 'wb') as f:
      pickle.dump(test_df, f)

    with open(output_dir_obj / "l1_model_params.yaml", "w", encoding="utf-8") as f:
      yaml.dump(l1_model_params, f, allow_unicode=True, default_flow_style=False)

    with open(output_dir_obj / "l1_fit_params.yaml", "w", encoding="utf-8") as f:
      yaml.dump(l1_fit_params, f, allow_unicode=True, default_flow_style=False)

    with open(output_dir_obj / "l2_model_params.yaml", "w", encoding="utf-8") as f:
      yaml.dump(l2_model_params, f, allow_unicode=True, default_flow_style=False)

    with open(output_dir_obj / "l2_fit_params.yaml", "w", encoding="utf-8") as f:
      yaml.dump(l2_fit_params, f, allow_unicode=True, default_flow_style=False)

  with open(output_dir_obj / "output_path.pickle", 'wb') as f:
    pickle.dump(output_dir_obj, f)

  OmegaConf.save(config=cfg, f=output_dir_obj / "used_params_config.yaml")
  print("実行は成功しました！")
  
  
if __name__ == "__main__":
  main()