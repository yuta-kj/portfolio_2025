from train_exe import train_exe

def local_opt(trial, model_name, opt_model_params, opt_fit_params,
              tr_df, cv_idx_tuple, level_status, params_obj):
    trgt_y = params_obj.trgt_y
    phase = params_obj.phase
    acc_local_loss = []
    print(f"★ {level_status}_{model_name}_LOCAL_TRIAL_{trial.number + 1} strat...")
    
    loss = train_exe(
        model_name=model_name,
        opt_model_params = opt_model_params,
        opt_fit_params = opt_fit_params,
        phase=phase, tr_df=tr_df,
        cv_idx_tuple=cv_idx_tuple, params_obj=params_obj,
        trial=trial, acc_local_loss=acc_local_loss)

    return loss