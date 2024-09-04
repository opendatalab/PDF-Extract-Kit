import os
import json
from tqdm.auto import tqdm
from simple_parsing import ArgumentParser
from batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
from get_data_utils import *
def clean_pdf_path(pdf_path):
    return pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path
def load_partition_pdf_mapping():

    trace_id_to_original_partition = {}
    original_partition_to_trace_id = {}
    with open("analysis/better_addon.filelist",'r') as f:
        lines = f.readlines()
        for line in tqdm(lines,desc="read better_addon.filelist"):
            line = line.strip().split()
            original_partition = line[0]
            if original_partition not in original_partition_to_trace_id: 
                original_partition_to_trace_id[original_partition] = []
            line = " ".join(line[1:])
            line = json.loads(line)
            for metadata in line:
                pdf_path = clean_pdf_path(metadata["path"])
                
                original_partition_to_trace_id[original_partition].append(pdf_path)
                trace_id_to_original_partition[pdf_path] = original_partition
            
    return original_partition_to_trace_id, trace_id_to_original_partition

def load_pdf_results_map():
    ## then we read whole result into memory
    pdf_path_to_result = {}
    RESULT_ROOT_PATH = "analysis/add_on_metadata/metadata/layoutV6/result"
    for result_file_name in tqdm(os.listdir(RESULT_ROOT_PATH),desc="read result"):
        result_file_path = os.path.join(RESULT_ROOT_PATH,result_file_name)
        
        with open(result_file_path,'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"read {result_file_name}",leave=False):
                line = line.strip()
                line = json.loads(line)
                pdf_path = clean_pdf_path(line["path"])
                pdf_path_to_result[pdf_path] = line
        # break
    return pdf_path_to_result

pdf_path_to_result = load_pdf_results_map()
client = build_client()
def process_file(partition, pdf_path_list):
    original_metadata_list = read_json_from_path(partition,client)
    original_metadata_map  = {clean_pdf_path(metadata["path"]):metadata for metadata in original_metadata_list}
    do_we_add_Q = False
    for pdf_path in pdf_path_list:
        if pdf_path not in pdf_path_to_result:
            #tqdm.write(f"pdf_path={pdf_path} not in result")
            continue
        if pdf_path in original_metadata_map:
            continue
        original_metadata_list.append(pdf_path_to_result[pdf_path])
        do_we_add_Q = True
    if do_we_add_Q:write_jsonl_to_path(original_metadata_list,partition,client)

def process_one_file_wrapper(args):
    (partition, pdf_path_list), args = args
    return    process_file(partition, pdf_path_list)
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(BatchModeConfig, dest="config")
    args = parser.parse_args()
    args = args.config   
    args.task_name = "scan"
    original_partition_to_trace_id, trace_id_to_original_partition = load_partition_pdf_mapping()
    alread_processing_file_list = list(original_partition_to_trace_id.items())
    alread_processing_file_list = obtain_processed_filelist(args, alread_processing_file_list)
    results = process_files(process_one_file_wrapper, alread_processing_file_list, args)


