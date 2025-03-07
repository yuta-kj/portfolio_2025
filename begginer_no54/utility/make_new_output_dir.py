from imports import (
    Path,
    datetime,
    re
)

# from pathlib import Path
# import datetime
# import re

def make_new_output_dir(base_path, exp_id):
    date = datetime.date.today()
    exam_year = date.year
    exam_month = date.month
    exam_day = date.day
    
    ptrn = re.compile(r'run_(\d{1,3})')
    max_id = 0

    output_path_obj = Path(base_path)/"output"/"exp"/f"exp_{exp_id}"
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True, exist_ok=True)
    
    for dir_obj in output_path_obj.iterdir():
        tmp_lst = ptrn.findall(str(dir_obj))
        if len(tmp_lst) == 0:
            continue;
        tmp_id = tmp_lst[-1]
        tmp_id = int(tmp_id)
        tmp_id = int(tmp_id)
        if tmp_id > max_id:
            max_id = tmp_id
    run_id = max_id + 1
    run_id = str(run_id)

    # output_path_obj = Path(base_path)/"output"/"exp"/f"exp_{exp_id}"
    # if not output_path_obj.exists():
    #     output_path_obj.mkdir(parents=True, exist_ok=True)

    new_output_path_obj = Path(base_path)/"output"/"exp"/f"exp_{exp_id}"/f"run_{run_id}"
    new_output_path_obj.mkdir(parents=True, exist_ok=True)

    return new_output_path_obj
    

