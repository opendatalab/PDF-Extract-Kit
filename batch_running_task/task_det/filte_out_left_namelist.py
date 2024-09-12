import os
Already_Done={}
empty_filelist = []
with open("finished.det_patch.filelist", "r") as f:
    for line in f:
        line = line.strip().split()
        if len(line)<4:continue
        size = int(line[-2])
        filename = line[-1]
        abspath  = f"opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/layoutV1/result/{filename}"
        if size == 0:
            empty_filelist.append(filename)
            continue
        Already_Done[filename]=abspath

Should_do = []
with open('scihub_collection/sci_hub.need_det.filelist','r') as f:
    for line in f:
        name = line.strip().split("/")[-1]
        if name in Already_Done:continue
        Should_do.append(name)
print(f"write to sci_index_files.remain.filelist")
with open('scihub_collection/sci_hub.need_det.remain.filelist','w') as f:
    for name in Should_do:
        f.write("opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/layoutV1/result/"+name+'\n')