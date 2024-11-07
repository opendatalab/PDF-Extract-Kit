from magic_pdf.pipe.UNIPipe import UNIPipe
import json
from magic_pdf.rw.S3ReaderWriter import S3ReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from batch_running_task.get_data_utils import *
from batch_running_task.batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig,dataclass
import json
from tqdm.auto import tqdm
from simple_parsing import ArgumentParser
import time
import subprocess

#model_list = reformat_to_minerU_input(pdf_info) #pdf_info['doc_layout_result']
FILEROOT="output/2301.01531"
filename= "2301.01531.json"
pdf_path= "Papers/mendeley_backup/2301.01531.pdf"

with open(f"{FILEROOT}/{filename}",'r') as f:
    model_list = json.load(f)
img_save_dir = f"{FILEROOT}/images"
image_writer = DiskReaderWriter(img_save_dir)
pdf_bytes = open(pdf_path,"rb").read()
pip = UNIPipe(pdf_bytes, {"_pdf_type":"", "model_list":model_list}, image_writer=image_writer)
pip.pdf_type = pip.PIP_OCR
pip.pipe_parse()
md = pip.pipe_mk_markdown(img_save_dir,drop_mode="none")
with open(f"{FILEROOT}/markdown.md",'w') as f:
    f.write(md)
print(md)