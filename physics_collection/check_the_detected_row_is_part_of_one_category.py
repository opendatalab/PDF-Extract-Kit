import os
import json
from tqdm.auto import tqdm
from batch_running_task.get_data_utils import *
import json
from tqdm.auto import tqdm

from simple_parsing import ArgumentParser
from batch_running_task.batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
PageInformationROOT="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/page_num_map"
OriginDATAROOT="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"

### form all physics doi set
with open("physics_collection/must_be_physics.doilist",'r') as f:
    physics_doilist = f.readlines()
    physics_doilist = [x.strip() for x in physics_doilist]
physics_doilist = set(physics_doilist)

client = build_client()
def process_file(metadata_file, args:BatchModeConfig):
    physics_collection = []
    metadata_list   = read_json_from_path(metadata_file,client)
    for metadata in metadata_list:
        doi = metadata['remark']['original_file_id']
        if doi in physics_doilist:
            physics_collection.append(metadata)
    return physics_collection

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
    whole_physics_collection = []
    for result in results:
        whole_physics_collection.extend(result)
    fold   = os.path.join(args.savepath,f"physics_collection.metadata.split")
    os.makedirs(fold,exist_ok=True)
    savepath = os.path.join(fold,f"{args.start_index:07d}-{args.end_index:07d}")
    with open(savepath,'w') as f:
        for metadata in whole_physics_collection:
            f.write(json.dumps(metadata)+'\n')

    