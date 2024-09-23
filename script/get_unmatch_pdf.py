import os

ROOT="analysis/unmatch_pdf_number.filelist.split"
BetterFullrestart = []
BetterAddon = []


for filename in os.listdir(ROOT):
    filepath = os.path.join(ROOT, filename)
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            metadata_name = line[0]
            current_num   = int(line[9].strip(','))
            should_num    = int(line[-1])
            if should_num - current_num < 500:
                BetterAddon.append(metadata_name)
            else:
                BetterFullrestart.append(metadata_name)
with open("sci_index_files.redo2.filelist",'w') as f:
    for i in BetterFullrestart:
        f.write(i+'\n')

with open("sci_index_files.addon3.filelist",'w') as f:
    for i in BetterAddon:
        f.write(i+'\n')
