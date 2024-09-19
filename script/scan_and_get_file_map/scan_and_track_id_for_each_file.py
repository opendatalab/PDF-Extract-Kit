import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_data_utils import *
import json
from tqdm.auto import tqdm
from simple_parsing import ArgumentParser
from batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
import time
import subprocess
PageInformationROOT="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/page_num_map"
OriginDATAROOT="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"

client = build_client()
def process_file(metadata_file, args:BatchModeConfig):
    pdf_path_map_to_page_num = []
    if metadata_file.startswith("s3:"):
        metadata_file = "opendata:"+metadata_file
    metadata_file_name = os.path.basename(metadata_file)
    metadata_list      = read_json_from_path(metadata_file,client)
    track_id_list = [metadata["track_id"] for metadata in metadata_list]
    return metadata_file + " " + ",".join(track_id_list)

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
    fold   = os.path.join(args.savepath,f"track_id_for_each_file.split")
    os.makedirs(fold,exist_ok=True)
    savepath = os.path.join(fold,f"{args.start_index:07d}-{args.end_index:07d}")
    with open(savepath,'w') as f:
        for result in results:
            f.write(result+'\n')
    