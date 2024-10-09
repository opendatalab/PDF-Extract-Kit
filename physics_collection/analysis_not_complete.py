import os,sys
import json
from tqdm.auto import tqdm
current_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(current_dir)
sys.path.append(current_dir)
from batch_running_task.get_data_utils import *
with open("physics_collection/analysis/not_complete_pdf_page_id.pairlist",'w') as file:
    with open("physics_collection/analysis/whole_layout_complete.filelist",'r') as f:
        alllines = f.readlines()
        for line in tqdm(alllines):
            line = line.strip().split()
            jsonl_path = line[0]
            bulk_status = json.loads(" ".join(line[1:]))
            # jsonl_path = data['file']
            # status = data['status']
            pdf_id_and_page_id_pair = []
            for track_id,pdf_status,page_status_list in bulk_status:
                for page_id, status in enumerate(page_status_list):
                    if status in {page_status.none, page_status.layout_complete_and_ocr_only_for_mfd}:
                        pdf_id_and_page_id_pair.append((track_id, page_id))
            if len(pdf_id_and_page_id_pair)>0:

                file.write(f"{jsonl_path} {json.dumps(pdf_id_and_page_id_pair)}"+'\n')
