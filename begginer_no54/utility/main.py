# from imports import *
# from ...utility.imports import (
#   DictConfig,
#   Path,
#   hydra,
#   pickle,
#   sys,
# #   yaml
# #   wandb
# )
# import sys
# sys.path.append('/content/drive/MyDrive/signate/beginner_no52')
# # from utility import callbacks

# from utility.imports import (
#   DictConfig,
#   Path,
#   hydra,
#   pickle,
#   sys,
# #   yaml
# #   wandb
# )
# import pdb; pdb.set_trace()

from imports import (
  DictConfig,
  OmegaConf,
  Path,
  hydra,
  pickle,
  sys,
  yaml
#   wandb
)

from FitModels import FitModels
from make_new_output_dir import make_new_output_dir

# yamlファイルのexam_idも併せて変更
# exam_id = 2
# exam_id = sys.argv[1]
# print(type(exam_id))
# print(exam_id)
# import pdb; pdb.set_trace()

base_path = sys.argv[0]
base_path = base_path.replace("/utility/main.py", "")
# コマンドライン引数をパース
exam_id = sys.argv[1]
# Hydraがargvを処理しないように、必要な引数だけ残して削除
sys.argv = sys.argv[:1]

@hydra.main(version_base=None,
    # config_path=f"/content/drive/MyDrive/signate/beginner_no52/exp/exp_{exam_id}/cfg_yamls/",
    config_path=f"{base_path}/exp/exp_{exam_id}/cfg_yamls/",
    config_name="params_config")

def main(cfg: DictConfig):
  global exam_id
  global base_path
  trgt_y = cfg.common.trgt_y
  base_path = Path(base_path)
#   base_path = Path('/content/drive/MyDrive/signate/beginner_no52')
  
  with open(base_path / f"data/cleaned_data/cleaned_tr_df_exam_{exam_id}.pickle", 'rb') as f:
    tr_df = pickle.load(f)
  with open(base_path / f"data/cleaned_data/cleaned_test_df_exam_{exam_id}.pickle", 'rb') as f:
    test_df = pickle.load(f)
    
    #   wandb_run = wandb.init(config=cfg)
  output_dir_obj = make_new_output_dir(base_path, exam_id)
  ens_regs = FitModels(tr_df, test_df, trgt_y, base_path, cfg, output_dir_obj, wandb_run=None)
  if cfg.common.use_single_model:
    tr_df, test_df, s_model_params, s_fit_params = ens_regs.main_exe()
  else:
    tr_df, test_df, l1_model_params, l1_fit_params, l2_model_params, l2_fit_params = ens_regs.main_exe()

#   trgt_ymd = ens_regs.trgt_ymd
#   exam_id = ens_regs.exam_id
#   save_dir = base_path / f"save/save_outputs/{trgt_ymd}/exam_no_{exam_id}"
#   save_dir = base_path / "output" / "exp" / f"exp_{exam_id}"
#   save_dir.mkdir(parents=True, exist_ok=True)


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


# def main(cfg: DictConfig):
# #   exam_id = cfg.common.exam_id
#   global exam_id
#   trgt_y = "y"
#   base_path = Path('/content/drive/MyDrive/signate/beginner_no52')
  
# #   with open(base_path / "data/cleaned_data/cleaned_tr_df.pickle", 'rb') as f:
# #     tr_df = pickle.load(f)
# #   with open(base_path / "data/cleaned_data/cleaned_test_df.pickle", 'rb') as f:
# #     test_df = pickle.load(f)

#   with open(base_path / f"data/cleaned_data/cleaned_tr_df_exam_{exam_id}.pickle", 'rb') as f:
#     tr_df = pickle.load(f)
#   with open(base_path / f"data/cleaned_data/cleaned_test_df_exam_{exam_id}.pickle", 'rb') as f:
#     test_df = pickle.load(f)
    
#     #   wandb_run = wandb.init(config=cfg)
#   output_dir_obj = make_new_output_dir(base_path, exam_id)
#   ens_clsfs = FitModels(tr_df, test_df, trgt_y, base_path, cfg, output_dir_obj, wandb_run=None)
#   meta_tr_df, meta_test_df, output_dir_obj = ens_clsfs.main_exe()
  
# #   trgt_ymd = ens_clsfs.trgt_ymd
# #   exam_id = ens_clsfs.exam_id
# #   save_dir = base_path / f"save/save_outputs/{trgt_ymd}/exam_no_{exam_id}"
# #   save_dir = base_path / "output" / "exp" / f"exp_{exam_id}"
# #   save_dir.mkdir(parents=True, exist_ok=True)
#   with open(output_dir_obj / "meta_tr_df.pickle", 'wb') as f:
#     pickle.dump(meta_tr_df, f)

#   with open(output_dir_obj / "meta_test_df.pickle", 'wb') as f:
#     pickle.dump(meta_test_df, f)

#   with open(output_dir_obj / "output_path.pickle", 'wb') as f:
#     pickle.dump(output_dir_obj, f)
  
#   OmegaConf.save(config=cfg, f=output_dir_obj / "params_config.yaml")
#   print("実行は成功しました！")
  
  
# if __name__ == "__main__":
#   main()