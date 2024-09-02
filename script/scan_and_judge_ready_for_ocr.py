from get_data_utils import *
import json
from tqdm.auto import tqdm

from simple_parsing import ArgumentParser
from batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
import time
import subprocess
PageInformationROOT="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/page_num_map"
OriginDATAROOT="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"
reason_code = { 
    "complete":   "P",
    "only_have_15": "I",
    "only_have_012467": "K",
    "no012467": "A",
    "none": "N"

}
def clean_pdf_path(pdf_path):
    return pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path
def get_page_info_path(metadata_file):
    metadata_file_name = os.path.basename(metadata_file)
    if metadata_file.startswith("opendata:") or metadata_file.startswith("s3:"):
        page_information_file_path= os.path.join(PageInformationROOT, metadata_file_name.replace(".jsonl",".json"))
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(metadata_file))))
        page_information_file_path= os.path.join(root, "page_num_map", metadata_file_name.replace(".jsonl",".json"))
        #print(page_information_file_path)
    return page_information_file_path

def get_origin_path(metadata_file):
    metadata_file_name = os.path.basename(metadata_file)
    if metadata_file.startswith("opendata:") or metadata_file.startswith("s3:"):
        origin_path= os.path.join(OriginDATAROOT, metadata_file_name)
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(metadata_file))))
        origin_path= os.path.join(root, "metadata", metadata_file_name)
    return origin_path

client = None
def process_file(metadata_file, args:BatchModeConfig):
    pdf_path_map_to_page_num = []
    metadata_file_name = os.path.basename(metadata_file)
    page_information_file_path= get_page_info_path(metadata_file)
    metadata_list   = read_json_from_path(metadata_file,client)
    page_information= read_json_from_path(page_information_file_path,client)
    if len(metadata_list) != len(page_information) and len(page_information) - len(metadata_list) > 500:
        exit_reason = "better_redo"
        return exit_reason, metadata_file, [metadata_file,"fail due to unmatch pdf number origin is {}, page information is {}".format(len(metadata_list), len(page_information))]

            
    page_information_map = {}
    for pdf_path, pdf_page_num in page_information:
        if pdf_page_num <= 0:
            continue
        pdf_path          = pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path
        page_information_map[pdf_path] = pdf_page_num
    
    if len(metadata_list) != len(page_information_map) and len(page_information_map) - len(metadata_list) <= 500:
        if len(metadata_list) < len(page_information_map):
            exit_reason = "better_addon"
            missing_part= []
            origin_data_path = get_origin_path(metadata_file)
            print(f"pdf num in page information is {len(page_information_map)} from {len(page_information)}")   
            print(f"pdf num in result is {len(metadata_list)}")
            unique_pdf_path = set([clean_pdf_path(metadata['path']) for metadata in metadata_list])
            print(f"pdf num in origin unique is {len(unique_pdf_path)}")
            not_in_page_information = set(page_information_map.keys()) - unique_pdf_path
            print(f"the missing pdf is {not_in_page_information}")
            for pdf_path in not_in_page_information:
                print(f"pdf_path is {pdf_path}=>{page_information_map[pdf_path]}")

            metadata_list=read_json_from_path(origin_data_path,client)
            print(f"pdf num in origin is {len(metadata_list)}")
            
            for pdf_id, metadata in enumerate(metadata_list):
                origin_path_path  = clean_pdf_path(metadata['path'])
                if origin_path_path in not_in_page_information:
                    missing_part.append(metadata)
            return exit_reason, metadata_file, [metadata_file, json.dumps(missing_part)]
        else:
            exit_reason = "check_the_page_information"
            print(f"fail due to unmatch pdf number origin is {len(metadata_list)}, page information is {len(page_information_map)} from {len(page_information)}")
            return exit_reason, metadata_file, [metadata_file,f"fail due to unmatch pdf number origin is {len(metadata_list)}, page information is {len(page_information_map)} from {len(page_information)}"]

    
    status_all_pdf = []
    for pdf_id, metadata in enumerate(metadata_list):
        doc_layout_result = metadata['doc_layout_result']
        origin_path_path  = metadata['path']
        origin_path_path  = origin_path_path[len("opendata:"):] if origin_path_path.startswith("opendata:") else origin_path_path
        assert origin_path_path in page_information_map, f"pdf_id={pdf_id} origin_path_path {origin_path_path} not in page_information_map"
        pdf_page_num = page_information_map[origin_path_path]
        
        status_for_this_pdf= {t:reason_code["none"] for t in range(pdf_page_num)}
        for page_meta in doc_layout_result:
            page_id = page_meta['page_id']
            ### now do parser check 
            page_id     =  page_meta["page_id"]
            layout_dets =  page_meta["layout_dets"]
            has_category_012467= False 
            has_category_15    = False
            for box in layout_dets:
                category_id = box['category_id']
                if category_id == 15:
                    has_category_15=True
                if category_id in [0,1,2,4,6,7]:
                    has_category_012467=True
                if has_category_15 and has_category_012467:
                    break
            if has_category_15 and has_category_012467:
                status_for_this_pdf[page_id] = reason_code["complete"]
            elif has_category_15:
                status_for_this_pdf[page_id] = reason_code["only_have_15"] ## this is impossible
            elif has_category_012467:
                status_for_this_pdf[page_id] = reason_code["only_have_012467"] ## this is mean ocr line det fail better to refine it later
            else:
                status_for_this_pdf[page_id] = reason_code["complete"] ### this is mean the whole page is not a string available page such as figure or table
        status_for_this_pdf_list_format = [status_for_this_pdf[page_id] for page_id in range(pdf_page_num)]
        status_all_pdf.append(status_for_this_pdf_list_format)
    
    is_whole_complete = True
    for status_for_this_pdf_list_format in status_all_pdf:
        for status in status_for_this_pdf_list_format:
            if status != reason_code["complete"]:
                is_whole_complete = False
                break
        if not is_whole_complete:
            break
    
    if is_whole_complete:
        exit_reason = "all_complete"
        return exit_reason, metadata_file,status_all_pdf
    else:
        exit_reason = "not_complete"
        return exit_reason, metadata_file,status_all_pdf
  
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
    analysis = {}
    for exit_reason, metadata_file, status_all_pdf in results:
        if exit_reason not in analysis: analysis[exit_reason] = []
        if exit_reason == "all_complete":
            analysis[exit_reason].append(metadata_file)
        elif exit_reason == "not_complete":
            analysis[exit_reason].append({'file':metadata_file, "status":status_all_pdf})
        else:
            analysis[exit_reason].append(status_all_pdf)

    for key, val in analysis.items():
        print(f"{key}=>{len(val)}")
        fold = os.path.join(args.logpath,f"{key.lower()}.filelist.split")
        logpath = os.path.join(fold,f"{args.start_index}-{args.end_index}")
        os.makedirs(fold, exist_ok=True)
        if key == "all_complete":
            with open(logpath, 'w') as f:
                for line in val:
                    f.write(line+'\n')
        elif key == "not_complete":
            with open(logpath, 'w') as f:
                for line in val:
                    f.write(json.dumps(line)+'\n')
        else:
            with open(logpath, 'w') as f:
                for metadata_file, astring in val:
                    f.write(f"{metadata_file} {astring}"+'\n')
                

    