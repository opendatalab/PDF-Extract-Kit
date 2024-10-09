import os
import json
## redirect package root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm.auto import tqdm
from batch_running_task.get_data_utils import *
from batch_running_task.batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
from simple_parsing import ArgumentParser
ROOT="physics_collection/physics_collection.metadata.split"
#SAVE="physics_collection/physics_collection.metadata.minibatch.split"
SAVE="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/physics_part/result/"
client = build_client()
#client = None
def process_file(filename, args:BatchModeConfig):
    chunk = 1000
    metadata_file= os.path.join(ROOT, filename)
    filename = filename.replace(".jsonl","")
    metadata = read_json_from_path(metadata_file,client)
    chunk_num= int(np.ceil(len(metadata)/chunk))
    for i in tqdm(range(chunk_num),position=1, leave=False):
        start = i*chunk
        end   = min((i+1)*chunk, len(metadata))
        save_path = os.path.join(SAVE, f"{filename}.{start:05d}_{end:05d}.jsonl")
        if os.path.exists(save_path):
            continue
        
        minibatch = metadata[start:end]
        
        write_jsonl_to_path(minibatch, save_path, client)

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
    