import os
import json
from tqdm.auto import tqdm
from get_data_utils import *
import json
from tqdm.auto import tqdm

from simple_parsing import ArgumentParser
from batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
PageInformationROOT="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/page_num_map"
OriginDATAROOT="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"

### form all physics doi set
with open("physics_collection/must_be_physics.doilist",'r') as f:
    physics_doilist = f.readlines()
    physics_doilist = [x.strip() for x in physics_doilist]
physics_doilist = set(physics_doilist)
output_width =1472 #pdf_metadata['width']#1472
output_height=1920 #pdf_metadata['height']#1920

def build_dict(pdf_metadata_list):
    pdf_metadata_dict = {}
    for pdf_metadata in pdf_metadata_list:
        track_id = pdf_metadata['track_id']
        height   = pdf_metadata['height']# 1920,
        width    = pdf_metadata['width']#1472
        if height == output_height and width == output_width:
            pass
        else:
            ### lets do the bbox convertion
            doc_layout_result=pdf_metadata['doc_layout_result']
            for pdf_page_metadata in doc_layout_result:
                page_id = pdf_page_metadata['page_id']
                layout_dets = []
                for res in pdf_page_metadata["layout_dets"]:
                    new_res = res.copy()
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    bbox= [xmin, ymin, xmax, ymax]
                    bbox= convert_boxes([bbox], pdf_metadata['width'], pdf_metadata['height'], output_width, output_height)[0]
                    poly= [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
                    res['poly'] = poly
        page_id_to_metadata = {pdf_page_metadata['page_id']: pdf_page_metadata for pdf_page_metadata in pdf_metadata['doc_layout_result']}
        pdf_metadata_dict[track_id] = page_id_to_metadata
            
    return pdf_metadata_dict

def read_data_with_patch(result_path, client):
    if result_path.startswith("s3:"):
        result_path = "opendata:"+result_path
    pdf_path_map_to_page_num = []
    assert "layoutV" in result_path
    filename   = os.path.basename(result_path)
    patch_path = os.path.join(os.path.dirname(os.path.dirname(result_path)),"det_patch_good",filename)
    missingpath= os.path.join(os.path.dirname(os.path.dirname(result_path)),"fix_missing_page_version1",filename)
    target_file_path = os.path.join(os.path.dirname(os.path.dirname(result_path)),"final",filename)
    if not args.redo and check_path_exists(target_file_path,client):
        tqdm.write(f"{target_file_path} already exists, skip")
        return
    tqdm.write(f"processing {result_path} to {target_file_path}")

    assert check_path_exists(result_path,client)
    tqdm.write("reading result")
    result  = read_json_from_path(result_path,client)
    result_dict      = build_dict(result)
    if check_path_exists(patch_path,client):
        tqdm.write("reading patch")
        patch   = read_json_from_path(patch_path,client)
        patch_add_dict   = build_dict(patch)
    else:
        patch_add_dict = {}
    if check_path_exists(missingpath,client):
        tqdm.write("reading missing")
        missing = read_json_from_path(missingpath,client)
        missing_dict     = build_dict(missing)
    else:
        missing_dict = {}

    tqdm.write("reading done")
    if len(patch_add_dict) == 0 and len(missing_dict) == 0:
        tqdm.write(f"no patch and missing for {result_path}")
        
    else:
        tqdm.write("adding patch and missing")
        for track_id, pdf_metadata in result_dict.items():
            for patch_dict in [patch_add_dict, missing_dict]:
                if track_id in patch_dict:
                    patch_pdf_metadata = patch_dict[track_id]
                    for page_id, pdf_page_metadata in patch_pdf_metadata.items():
                        if page_id in pdf_metadata:
                            ## then merge page result
                            pdf_metadata[page_id]["layout_dets"].extend(pdf_page_metadata["layout_dets"])
                        else:
                            pdf_metadata[page_id] = pdf_page_metadata
 
        for pdf_metadata in result:
            track_id = pdf_metadata['track_id']
            pdf_metadata['height'] = output_height
            pdf_metadata['width'] = output_width
            doc_layout_result = []
            for page_id, pdf_page_metadata in result_dict[track_id].items():
                doc_layout_result.append(pdf_page_metadata)
            pdf_metadata['doc_layout_result'] = doc_layout_result       
    return result

client = build_client()  
def process_file(metadata_file, args:BatchModeConfig):
    physics_collection = []
    metadata_list   = read_data_with_patch(metadata_file,client)
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

    