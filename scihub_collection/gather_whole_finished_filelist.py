import os
filepathlist= [
"layoutV1",
"layoutV2",
"layoutV3",
"layoutV5",
"layoutV6",
]
too_small_files = {}
with open("scihub_collection/sci_hub.finished.filelist",'w') as file:
    for filepart in filepathlist:
        filepath = f"scihub_collection/sci_hub.finished.{filepart}.filelist"
        with open(filepath, 'r') as f:
            for line in f:
                date, time, size, name = line.strip().split()
                abspath = f"s3://llm-pdf-text/pdf_gpu_output/scihub_shared/{filepart}/result/{name}"
                if int(size) < 1000:
                    too_small_files[name] = abspath
                    continue
                if name in too_small_files:
                    ##remove the file from the set
                    del too_small_files[name]
                file.write(abspath+'\n')
print(f"Too small files num = {len(too_small_files)}=>{too_small_files.values()}")