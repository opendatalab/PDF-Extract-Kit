import os
import json
from tqdm.auto import tqdm
#simple_not_complete = []
with open("analysis/redo.pageid.filelist",'w') as fileobj:
    with open("analysis/not_complete.filelist",'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            this_line_missing = []
            data  = json.loads(line)
            file  = data['file']
            status= data['status']
            for pdf_id, page_status in enumerate(status):
                for page_id, status in enumerate(page_status):
                    if status != 'P':
                        this_line_missing.append(f"{pdf_id},{page_id}")
            this_line_missing = "|".join(this_line_missing)
            fileobj.write(f"{file} {this_line_missing}\n")
                        #fileobj.write(f"{file} {pdf_id} {page_id}\n")
                        #simple_not_complete.append([file, pdf_id, page_id])
# with open("analysis/redo.pageid.filelist",'w') as f:
#     for file, pdf_id, page_id in simple_not_complete:
#         f.write(f"{file} {pdf_id} {page_id}")
