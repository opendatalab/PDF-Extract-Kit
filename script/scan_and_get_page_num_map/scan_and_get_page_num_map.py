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

@dataclass
class PageNumConfig(BatchModeConfig):
    savepath: str = "opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared"

client = build_client()
def process_file(metadata_file, args:PageNumConfig):
    pdf_path_map_to_page_num = []
    if "layoutV" in metadata_file:
        filename = os.path.basename(metadata_file)
        metadata_file = os.path.join(OriginDATAROOT,filename)
    metadata_file_name = metadata_file.split("/")[-1].replace('.jsonl','.json')
    target_file_path   = os.path.join(args.savepath, f"page_num_map/{metadata_file_name}")
    if not args.redo and check_path_exists(target_file_path, client):
        tqdm.write(f"already processed {metadata_file}, we pass")
        return
    metadatalist = read_json_from_path(metadata_file, client)
    tqdm.write(f"save to {target_file_path}")
    iterater  = tqdm(metadatalist,position=1,leave=False) if args.batch_num==0 else metadatalist
    for metadata in iterater:
        pdfpath  = metadata['path']
        if pdfpath.startswith("s3:"): pdfpath = "opendata:"+ pdfpath
        if not check_path_exists(pdfpath, client):
            pdf_path_map_to_page_num.append([pdfpath, 0])
            continue
        try:
            pdf_buffer = read_pdf_from_path(pdfpath, client)
            page = pdf_buffer.load_page(0)
            dpi = 200
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        except Exception as e:
            tqdm.write(f"""error in loading pdf {pdfpath}, we pass""")
            pdf_path_map_to_page_num.append([pdfpath, -1])
            print(e)
            continue
        pdf_path_map_to_page_num.append([pdfpath, len(pdf_buffer)])
    
    write_json_to_path(pdf_path_map_to_page_num,target_file_path ,client)
    #return pdf_path_map_to_page_num

def process_one_file_wrapper(args):
    arxiv_path, args = args
    return    process_file(arxiv_path,args)
 
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(PageNumConfig, dest="config")
    args = parser.parse_args()
    args = args.config   
    args.task_name = "scan"
    alread_processing_file_list = obtain_processed_filelist(args)
    results = process_files(process_one_file_wrapper, alread_processing_file_list, args)
    # pdf_path_map_to_page_num_total = {}
    # for pdf_path_map_to_page_num in results:
    #     pdf_path_map_to_page_num_total.update(pdf_path_map_to_page_num)
    # with open(f"page_num_map/pdf_path_map_to_page_num.{args.index_part}.{args.num_parts}.json",'w') as f:
    #     json.dump(pdf_path_map_to_page_num_total,f)

    

    