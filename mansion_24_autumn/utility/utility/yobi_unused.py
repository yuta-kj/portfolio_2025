#   def make_new_exam(self):
#     if self._make_new_exam:
#       self.trgt_ymd = f'{self.exam_year}_{self.exam_month}_{self.exam_day}'
#       save_dir = self.base_path / f'save/save_study/{self.trgt_ymd}'
#       if not save_dir.exists():
#         save_dir.mkdir(parents=True, exist_ok=True)
#       ptrn = re.compile(r'(\d{1,3})')
#       max_id = 0
#       for iter in save_dir.iterdir():
#         tmp_lst = ptrn.findall(str(iter))
#         tmp_id = tmp_lst[-1]
#         if not tmp_id.isdigit():
#           continue;
#         tmp_id = int(tmp_id)
#         if tmp_id > max_id:
#           max_id = tmp_id
#       exam_id = max_id + 1
#       self.exam_id = exam_id
#       study_exam_dir = save_dir / f'exam_no_{exam_id}'
#       study_exam_dir.mkdir(parents=True, exist_ok=True)
#       self.save_study_path = study_exam_dir
#       self._make_new_exam = False


#   def use_latest_exam(self):
#     if self._use_latest_exam:
#       save_study_dir = self.base_path / 'save/save_study'
#       tmp_lst = [child for child in save_study_dir.iterdir() if child.is_dir()]
#       tmp_lst = list(map(lambda x: x.parts[-1], tmp_lst))
#       tmp_lst = sorted(tmp_lst, key=lambda x: tuple(map(int, x.split('_'))), reverse=True)
#       trgt_ymd = tmp_lst[0]
#       self.trgt_ymd = trgt_ymd
#       latest_date_dir = save_study_dir / trgt_ymd
#       ptrn = re.compile(r'(\d{1,3})')
#       max_id = 0
#       for iter in latest_date_dir.iterdir():
#         tmp_lst = ptrn.findall(str(iter))
#         tmp_id = tmp_lst[-1]
#         if not tmp_id.isdigit():
#           continue;
#         tmp_id = int(tmp_id)
#         if tmp_id > max_id:
#           max_id = tmp_id
#       exam_id = max_id
#       self.exam_id = exam_id
#       self.save_study_path = latest_date_dir / f'exam_no_{exam_id}'
#       self._use_latest_exam = False