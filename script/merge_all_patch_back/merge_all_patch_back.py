import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from batch_running_task.get_data_utils import *
from batch_running_task.batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig,dataclass
import json
from tqdm.auto import tqdm
from simple_parsing import ArgumentParser
import time
import subprocess
client = build_client()
OriginDATAROOT="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"

from batch_running_task.utils import convert_boxes
output_width =1472 #pdf_metadata['width']#1472
output_height=1920 #pdf_metadata['height']#1920


client = build_client()
def process_file(result_path, args):
    if result_path.startswith("s3:"):
        result_path = "opendata:"+result_path
    filename = os.path.basename(result_path)
    target_file_path = os.path.join(os.path.dirname(os.path.dirname(result_path)),"final_20240923",filename)
    if not args.redo and check_path_exists(target_file_path,client):
        tqdm.write(f"skip {target_file_path}")
        return 
    #target_file_path = "test.jsonl"
    result = read_data_with_version(result_path,client)
    tqdm.write(f"read {result_path} to {target_file_path}")
    
    write_jsonl_to_path(result,target_file_path ,client)
    #return pdf_path_map_to_page_num

def process_one_file_wrapper(args):
    arxiv_path, args = args
    return    process_file(arxiv_path,args)
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(BatchModeConfig, dest="config")
    args = parser.parse_args()
    args = args.config   
    args.task_name = "scan"
    alread_processing_file_list = obtain_processed_filelist(args)
    results = process_files(process_one_file_wrapper, alread_processing_file_list, args)
  

    