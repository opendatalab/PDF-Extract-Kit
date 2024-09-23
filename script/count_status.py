import os
from tqdm.auto import tqdm
with open("analysis/all_complete.filelist",'r') as f:
    all_complete = f.readlines()
    all_complete = [i.strip() for i in tqdm(all_complete)]

import json
with open("analysis/not_complete.filelist",'r') as f:
    not_complete = f.readlines()
    not_complete = [json.loads(i.strip())["file"] for i in tqdm(not_complete)]

with open("sci_index_files.finished.filelist",'r') as f:
    whole_names = f.readlines()
    whole_names = [i.strip() for i in whole_names]

print("all_complete:", len(all_complete))
print("not_complete:", len(not_complete))
print("whole_names:", len(whole_names))

remain = list(set(whole_names) - set(all_complete) - set(not_complete))
with open("analysis/no_status.filelist",'w') as f:
    for i in remain:
        f.write(i+'\n')