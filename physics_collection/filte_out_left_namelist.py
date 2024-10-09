import os
from typing import List
def obtain_pdfpath_from_jsonl(jsonlpath_list:List[str]):
    
    Already_Done = {}
    
    empty_filelist=[]
    if not isinstance(jsonlpath_list, list):    
        jsonlpath_list = [jsonlpath_list]
    for jsonlpath in jsonlpath_list:
        with open(jsonlpath, "r") as f:
            for line in f:
                line = line.strip().split()
                if len(line)<4:continue
                size = int(line[-2])
                filename = line[-1]
                if size == 0:
                    empty_filelist.append(NamePathMap[filename])
                    continue
                Already_Done[filename]=NamePathMap[filename]
    return Already_Done, empty_filelist
ROOT="finished"
NamePathMap={}
with open('physics_collection/physics.files.final.checklist.filelist','r') as f:
    for line in f:
        path = line.strip()
        name = os.path.basename(path)
        NamePathMap[name] = path

#Already_Done, empty_filelist = obtain_pdfpath_from_jsonl('physics_collection/finished.rec.filelist')
Already_Done, empty_filelist = obtain_pdfpath_from_jsonl(['physics_collection/physics.files.final.filelist',
                                                          ])

Already_Done_path = "physics_collection/sci_index_files.finished.filelist"
with open(Already_Done_path, "w") as f:
    for name, path in Already_Done.items():
        path = f"opendata:{path}" if not path.startswith("opendata:") else path
        f.write(f"{path}\n")
with open("physics_collection/sci_index_files.redo.filelist","w") as f:
    for name in empty_filelist:
        f.write(name+'\n')

Should_do = set(NamePathMap.keys()) - set(Already_Done.keys())
remain_file_path = "physics_collection/sci_index_files.remain.filelist"
print(remain_file_path)
with open(remain_file_path,'w') as f:
    for name in Should_do:
        f.write(f"{NamePathMap[name]}\n")