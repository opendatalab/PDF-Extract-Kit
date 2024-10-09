from magic_pdf.pipe.UNIPipe import UNIPipe
import json
from magic_pdf.rw.S3ReaderWriter import S3ReaderWriter
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
from batch_running_task.utils import convert_boxes
# set logging level to Error
from loguru import logger

# Remove the default logger
logger.remove()

# Add a new logger with the desired level
logger.add(sys.stderr, level="ERROR")

@dataclass
class FormatIntoMarkdownConfig(BatchModeConfig):
    savepath: str = "opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared/physics_part/markdown"
    saveimageQ: bool = False


def reformat_to_minerU_input(pool):
    """
    our directly formated input is a little different from the format that minerU required.
    we need convert


    MinerU input format:
    {
        ....
        "doc_layout_result": [
            {
            "layout_dets": [...],
            "page_info": {}
        ]
    }
    """
    height = pool["height"]
    width  = pool["width"]
    old_doc_layout_result = pool['doc_layout_result']
    new_doc_layout_result = []
    for page_res_information in old_doc_layout_result:
        page_num = page_res_information['page_id']

        old_page_layout_dets = page_res_information['layout_dets']
        new_page_layout_dets = []
        
        for bbox_information in old_page_layout_dets:
            if bbox_information.get('category_id',"")!=15:
                new_page_layout_dets.append(bbox_information)
            elif "sub_boxes" not in bbox_information :
                if "text" not in bbox_information:
                    bbox_information['text'] = "[Missing]"
                    bbox_information['score']= 1
                new_page_layout_dets.append(bbox_information)
            elif len(bbox_information['sub_boxes']) == 0:
                new_bbox_information = {
                    "category_id": bbox_information['category_id'],
                    "poly": bbox_information['poly'],
                    "text": bbox_information['text'],
                    "score": bbox_information['score']
                }
                new_page_layout_dets.append(new_bbox_information)
            else:
                current_bbox_catagory = bbox_information['category_id']
                for sub_bbox_information in bbox_information['sub_boxes']:
                    new_bbox_information = {
                        "category_id": current_bbox_catagory,
                        "poly": sub_bbox_information['poly'],
                        "text": sub_bbox_information['text'],
                        "score": sub_bbox_information['score']
                    }
                    new_page_layout_dets.append(new_bbox_information)
    
        new_page = {
            "layout_dets": new_page_layout_dets,
            "page_info": {"page_no": page_num, "height": height, "width": width}
        }
        new_doc_layout_result.append(new_page)
    return new_doc_layout_result

def process_file(jsonl_path, args):
    filename  = os.path.basename(jsonl_path)
    saveroot  = os.path.dirname(os.path.dirname(jsonl_path))
    targetpath= os.path.join(saveroot,"markdown",filename)
    image_save_root = os.path.join(saveroot,"images_per_pdf",filename[:-len(".jsonl")])
    if not args.redo and check_path_exists(targetpath,client):
        tqdm.write(f"skip {targetpath}")
        return
    jsonl_data = read_json_from_path(jsonl_path,client)
    markdown_for_this_bulk = []
    for pdf_info in tqdm(jsonl_data,desc=f"process {filename}",leave=False,position=1):
        try:
            model_list = reformat_to_minerU_input(pdf_info) #pdf_info['doc_layout_result']
            track_id   = pdf_info['track_id']
            img_save_dir = os.path.join(image_save_root, track_id)
            if img_save_dir.startswith("opendata:"):
                img_save_dir = img_save_dir[len("opendata:"):]
            # if img_save_dir.startswith("s3://"):
            #     img_save_dir = img_save_dir[len("s3://"):]
            image_writer = S3ReaderWriter(ak="ZSLIM2EYKENEX5B4AYBI", 
                                        sk="2ke199F35V9Orwcu8XJyGUcaJzeDz4LzvMP5yEFD", 
                                        endpoint_url='http://p-ceph-norm-outside.pjlab.org.cn',
                                        parent_path=img_save_dir)
            pdf_path = pdf_info["path"]
            if pdf_path.startswith("s3:"):
                pdf_path = "opendata:"+pdf_path
            pdf_bytes = client.get(pdf_path)#read_pdf_from_path(pdf_path,client)
            pip = UNIPipe(pdf_bytes, {"_pdf_type":"", "model_list":model_list}, image_writer=image_writer)
            pip.pdf_type = pip.PIP_OCR
            pip.pipe_parse()
            md = pip.pipe_mk_markdown(img_save_dir,drop_mode="none")
        except:
            md = "Fail to Parser"
        
        markdown_for_this_bulk.append({"track_id":track_id, "path":pdf_path, "markdown":md})
    write_jsonl_to_path(markdown_for_this_bulk,targetpath,client)


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
  

    
    





