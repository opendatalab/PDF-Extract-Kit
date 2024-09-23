import os
import json
add_on_metadata_list = []

with open("analysis2/better_addon.filelist",'r') as f:
    for line in f:
        line = line.strip()
        line = " ".join(line.split()[1:])
        line = json.loads(line)
        for metadata in line:
            add_on_metadata_list.append(metadata)

import numpy as np 
### split into chunk 1000 

add_on_metadata_chunk = np.array_split(add_on_metadata_list, 1)
for chunk_id, chunk in enumerate(add_on_metadata_chunk):
    with open(f"analysis2/add_on_metadata/metadata/{chunk_id}.jsonl",'w') as f:
        for metadata in chunk:
            f.write(json.dumps(metadata) + "\n")
