import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_data_utils import *
import json
from tqdm.auto import tqdm
from simple_parsing import ArgumentParser
from batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig, dataclass
import time
import subprocess
from typing import Dict, Any



@dataclass
class StatusCheckConfig(BatchModeConfig):
    PageInformationROOT:str="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/page_num_map"
    OriginDATAROOT:str="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"
    use_patch:bool=False
    use_candidate:bool=False

def clean_pdf_path(pdf_path):
    return pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path

def get_page_info_path(metadata_file,args):
    metadata_file_name = os.path.basename(metadata_file)
    
    if metadata_file.startswith("s3:"):metadata_file="opendata:"+metadata_file
    if metadata_file.startswith("opendata:"):
        page_information_file_path= os.path.join(args.PageInformationROOT, metadata_file_name.replace(".jsonl",".json"))
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(metadata_file))))
        page_information_file_path= os.path.join(root, "page_num_map", metadata_file_name.replace(".jsonl",".json"))
        if not os.path.exists(page_information_file_path):
            page_information_file_path = os.path.join(args.PageInformationROOT, metadata_file_name.replace(".jsonl",".json"))
        #print(page_information_file_path)
    return page_information_file_path

def get_origin_path(metadata_file,args):
    metadata_file_name = os.path.basename(metadata_file)
    if metadata_file.startswith("opendata:") or metadata_file.startswith("s3:"):
        origin_path= os.path.join(args.OriginDATAROOT, metadata_file_name)
    else:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(metadata_file))))
        origin_path= os.path.join(root, "metadata", metadata_file_name)
    return origin_path

def get_box_status(box:Dict[str,Any]):
    category_id = box['category_id']
    if category_id == 15:
        if box.get('text',"") != "" or 'sub_boxes' in box:
            return boxstatus.has_category_rec_and_get_rec
        else:
            return boxstatus.has_category_rec_without_rec
    elif category_id in {0,1,2,4,6,7}:
        return boxstatus.has_category_layout
    elif category_id in {13, 14}:
        if box.get('latex',"") != "":
            return boxstatus.has_category_mfd_and_get_mfr
        else:
            return boxstatus.has_category_mfd_without_mfr

def judge_one_page_status_via_box_status(box_status_list):
    box_status_list = set(box_status_list)
    #assert boxstatus.has_category_layout in box_status_list, f"box_status_list={box_status_list}"
    if boxstatus.has_category_layout not in box_status_list:
        return page_status.none
    ## not mfd and rec 
    if (boxstatus.has_category_mfd_and_get_mfr not in box_status_list and boxstatus.has_category_rec_and_get_rec not in box_status_list 
    and boxstatus.has_category_mfd_without_mfr not in box_status_list and boxstatus.has_category_rec_without_rec not in box_status_list):
        return page_status.only_have_layout
    ## has mfd or rec but without get mfr or rec
    elif (boxstatus.has_category_mfd_and_get_mfr not in box_status_list and boxstatus.has_category_rec_and_get_rec not in box_status_list):
        return page_status.layout_complete
    elif ( (boxstatus.has_category_mfd_and_get_mfr in box_status_list and boxstatus.has_category_rec_and_get_rec in box_status_list)
      or (boxstatus.has_category_mfd_and_get_mfr in box_status_list and boxstatus.has_category_rec_without_rec not in box_status_list)
      or (boxstatus.has_category_rec_and_get_rec in box_status_list and boxstatus.has_category_mfd_without_mfr not in box_status_list)):  
        return page_status.layout_complete_and_ocr_finished
    elif (boxstatus.has_category_mfd_and_get_mfr in box_status_list and boxstatus.has_category_rec_without_rec in box_status_list):
        return page_status.layout_complete_and_ocr_only_for_rec
    elif (boxstatus.has_category_rec_and_get_rec in box_status_list and boxstatus.has_category_mfd_without_mfr in box_status_list):
        return page_status.layout_complete_and_ocr_only_for_mfd

def judge_one_pdf_status_via_page_status(page_status_list):
    page_status_list = set(page_status_list)
    if page_status.only_have_layout in page_status_list:
        return pdf_status.layout_not_complete
    elif page_status.layout_complete in page_status_list:
        return pdf_status.layout_has_complete
    elif page_status.layout_complete_and_ocr_finished in page_status_list:
        return pdf_status.layout_complete_and_ocr_finished
    elif page_status.layout_complete_and_ocr_only_for_rec in page_status_list:
        return pdf_status.layout_complete_without_ocr
    elif page_status.layout_complete_and_ocr_only_for_mfd in page_status_list:
        return pdf_status.layout_complete_without_ocr
    else:
        return pdf_status.layout_not_complete

def judge_package_status_via_pdf_status(pdf_status_list):
    pdf_status_list = set(pdf_status_list)
    # if pdf_status.layout_not_complete in pdf_status_list:
    #     return packstatus.layout_not_complete
    if pdf_status.layout_has_complete in pdf_status_list:
        return packstatus.whole_layout_complete
    elif pdf_status.layout_complete_and_ocr_finished in pdf_status_list:
        return packstatus.whole_ocr_complete
    else:
        return packstatus.better_redo



client = build_client()
def process_file(metadata_file, args:StatusCheckConfig):
    pdf_path_map_to_page_num = []
    if metadata_file.startswith("s3:"):
        metadata_file = "opendata:"+metadata_file
    metadata_file_name = os.path.basename(metadata_file)
    page_information_file_path= get_page_info_path(metadata_file,args)
    if args.use_candidate:
        new_path = metadata_file.replace('result/','final_layout/')
        if check_path_exists(new_path,client):
            metadata_file = new_path
    if args.use_patch:
        metadata_list   = read_data_with_patch(metadata_file,client)
    else:
        metadata_list   = read_json_from_path(metadata_file,client)
    page_information= read_json_from_path(page_information_file_path,client)
    if len(metadata_list) != len(page_information) and len(page_information) - len(metadata_list) > 500:
        exit_reason = packstatus.better_redo
        return exit_reason, metadata_file, [metadata_file,"fail due to unmatch pdf number origin is {}, page information is {}".format(len(metadata_list), len(page_information))]

            
    page_information_map = {}
    for pdf_path, pdf_page_num in page_information:
        if pdf_page_num <= 0:
            continue
        pdf_path          = pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path
        page_information_map[pdf_path] = pdf_page_num
    
    if len(metadata_list) != len(page_information_map) and len(page_information_map) - len(metadata_list) <= 500:
        if len(metadata_list) < len(page_information_map):
            exit_reason = packstatus.better_addon
            missing_part= []
            origin_data_path = get_origin_path(metadata_file,args)
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
            exit_reason = packstatus.check_the_page_information
            print(f"fail due to unmatch pdf number origin is {len(metadata_list)}, page information is {len(page_information_map)} from {len(page_information)}")
            return exit_reason, metadata_file, [metadata_file,f"fail due to unmatch pdf number origin is {len(metadata_list)}, page information is {len(page_information_map)} from {len(page_information)}"]

    
    status_all_pdf = []
    for pdf_id, metadata in enumerate(metadata_list):
        doc_layout_result = metadata['doc_layout_result']
        origin_path_path  = metadata['path']
        origin_path_path  = origin_path_path[len("opendata:"):] if origin_path_path.startswith("opendata:") else origin_path_path
        assert origin_path_path in page_information_map, f"pdf_id={pdf_id} origin_path_path {origin_path_path} not in page_information_map"
        pdf_page_num = page_information_map[origin_path_path]
        track_id = metadata['track_id']
        status_for_this_pdf= {t:page_status.none for t in range(pdf_page_num)}
        for page_meta in doc_layout_result:
            page_id = page_meta['page_id']
            ### now do parser check 
            page_id     =  page_meta["page_id"]
            layout_dets =  page_meta["layout_dets"]
            if len(layout_dets)==0:
                continue
                #raise ValueError(f"pdf_id={pdf_id} page_id={page_id} page_meta={page_meta} is empty")
            box_status_list = [get_box_status(box) for box in layout_dets]
            status_for_this_pdf[page_id] = judge_one_page_status_via_box_status(box_status_list)

        status_for_this_pdf_list_format = [status_for_this_pdf[page_id] for page_id in range(pdf_page_num)]
        pdf_status_for_this_pdf = judge_one_pdf_status_via_page_status(status_for_this_pdf_list_format)
        status_all_pdf.append([track_id, pdf_status_for_this_pdf, status_for_this_pdf_list_format])
    
    exit_reason = judge_package_status_via_pdf_status([pdf_status for track_id, pdf_status, status_for_this_pdf_list_format in status_all_pdf])
    
    return exit_reason, metadata_file,status_all_pdf

def process_one_file_wrapper(args):
    arxiv_path, args = args
    return    process_file(arxiv_path,args)
 


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(StatusCheckConfig, dest="config")
    args = parser.parse_args()
    args = args.config   
    args.task_name = "scan"
    alread_processing_file_list = obtain_processed_filelist(args)
    results = process_files(process_one_file_wrapper, alread_processing_file_list, args)
    analysis = {}
    for exit_reason, metadata_file, status_all_pdf in results:
        if exit_reason not in analysis: analysis[exit_reason] = []
        analysis[exit_reason].append([metadata_file, status_all_pdf])

        # if exit_reason not in analysis: analysis[exit_reason] = []
        # if exit_reason == "all_complete":
        #     analysis[exit_reason].append(metadata_file)
        # elif exit_reason == "not_complete":
        #     analysis[exit_reason].append({'file':metadata_file, "status":status_all_pdf})
        # else:
        #     analysis[exit_reason].append(status_all_pdf)

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
                for metadata_file, status_all_pdf in val:
                    f.write(f"{metadata_file} "+ json.dumps(status_all_pdf) +'\n')
                

    