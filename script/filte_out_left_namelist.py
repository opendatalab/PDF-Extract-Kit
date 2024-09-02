import os
ROOT="finished"
Already_Done = {}
empty_filelist=[]
for name in os.listdir(ROOT):
    print(f"read {name}")
    version = name.split(".")[1]
    if not name.endswith(".filelist"):continue
    with open(os.path.join(ROOT, name), "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line)<4:continue
            size = int(line[-2])
            filename = line[-1]
            abspath  = f"s3://llm-pdf-text/pdf_gpu_output/scihub_shared/{version}/result/{filename}"
            if size == 0:
                empty_filelist.append(f"opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub/"+filename)
                continue
            
            
            Already_Done[filename]=abspath

Already_Done_path = "sci_index_files.finished.filelist"
with open(Already_Done_path, "w") as f:
    for name, path in Already_Done.items():
        f.write(f"opendata:{path}\n")
with open("sci_index_files.redo.filelist","w") as f:
    for name in empty_filelist:
        f.write(name+'\n')

Should_do = []
with open('sci_index_files.namelist','r') as f:
    for line in f:
        name = line.strip()
        if name in Already_Done:
            continue
        Should_do.append(name)
print(f"write to sci_index_files.remain.filelist")
with open('sci_index_files.remain.filelist','w') as f:
    for name in Should_do:
        f.write("opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub/"+name+'\n')