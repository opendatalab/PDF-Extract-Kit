import os
import json
from tqdm.auto import tqdm
reason_code = { 
    "complete":   "P", #<--- layout + mfd + ocr_det  
    "only_have_15": "I", #<--- layout + mfd 
    "only_have_012467": "K", #<--- layout
    "no012467": "A",
    "none": "N"

}

 
with open("scihub_collection/analysis/not_complete_pdf_page_id.pairlist",'w') as file:
    with open("scihub_collection/analysis/not_complete.filelist",'r') as f:
        alllines = f.readlines()
        for line in tqdm(alllines):
            data = json.loads(line.strip())
            jsonl_path = data['file']
            status = data['status']
            pdf_id_and_page_id_pair = []
            for track_id,page_status in status:
                # pdf_id: int
                # page_status: List[int]
                for page_id, status in enumerate(page_status):
                    if status in {'N'}:
                        pdf_id_and_page_id_pair.append((track_id, page_id))
            if len(pdf_id_and_page_id_pair)>0:
                file.write(f"{jsonl_path} {json.dumps(pdf_id_and_page_id_pair)}"+'\n')
