import os
Already_Done = []
with open("page_num_map.filelist", "r") as f:
    for line in f:
        line = line.strip().split()
        if len(line)<4:continue
        name = line[-1]
        Already_Done.append(name)

Already_Done = set(Already_Done)
print(f"read sci_index_files.namelist")
Should_do = []
with open('sci_index_files.namelist','r') as f:
    for line in f:
        name = line.strip()[:-1]
        if name in Already_Done:continue
        Should_do.append(name)
print(f"write to page_num_map.remain.filelist")
with open('page_num_map.remain.filelist','w') as f:
    for name in Should_do:
        f.write("opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub/"+name+'l\n')