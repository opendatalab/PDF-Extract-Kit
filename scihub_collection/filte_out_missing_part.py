import os
from tqdm import tqdm
ROOT='scihub_collection/analysis'
ROOT='physics_collection/analysis'
print("collect already done jsonl name")
Already_Done = []
with open("scan_finished.missingpart", "r") as f:
    for line in f:
        line = line.strip().split('/')
        name = line[-1]
        Already_Done.append(name)
print(Already_Done[:2])
Already_Done = set(Already_Done)

print(f"collect should do jsonl name")
Should_do = {}
with open(f'{ROOT}/not_complete_pdf_page_id.pairlist.filelist','r') as f:
    for inputs_line in tqdm(f):
        splited_line = inputs_line.split()
        inputs_path = splited_line[0]
        line = inputs_path.strip().split('/')
        name = line[-1]

        Should_do[name]=inputs_line

Remain_jsonl_name = set(Should_do.keys()) - Already_Done
print("=================")
print(f"Remain_jsonl_name: {len(Remain_jsonl_name)}")
print(f"Already_Done: {len(Already_Done)}")
print(f"Should_do: {len(Should_do)}")
with open(f'{ROOT}/not_complete_pdf_page_id.pairlist.remain.filelist','w') as f:
    for name in Remain_jsonl_name:
        f.write(Should_do[name])