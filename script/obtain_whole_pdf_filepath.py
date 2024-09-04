from get_data_utils import *

from pathlib import Path
import sys, os
from batch_run_utils import BatchModeConfig, obtain_processed_filelist, process_files,dataclass, save_analysis
from simple_parsing import ArgumentParser

from petrel_client.client import Client  # 安装完成后才可导入 
client = Client(conf_path="~/petreloss.conf") # 实例化Petrel Client，然后就可以调用下面的APIs  
@dataclass
class GetWholePDFConfig(BatchModeConfig):
    task_name = 'get_whole_PDF_files'

def check_one_path_wrapper(args):
    metadata_json_path, args = args
    data = read_json_from_path(metadata_json_path,client)
    return data
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(GetWholePDFConfig, dest="config")
    args = parser.parse_args()
    args = args.config

    
    #if args.mode == 'analysis':
    alread_processing_file_list = obtain_processed_filelist(args)
    results = process_files(check_one_path_wrapper, alread_processing_file_list, args)
    whole_path_list = []
    for pathlist in results:
        whole_path_list.extend(whole_path_list)
    filename = os.path.basename(args.root_path).replace('.jsonl','.filelist')
    with open(f"pdf_path_collections/{filename}",'w') as f:
        f.write('\n'.join(whole_path_list))
    # #print(results)
    # analysis= {}
    # for arxivid, _type in results:
    #     if _type not in analysis:
    #         analysis[_type] = []
    #     analysis[_type].append(arxivid)

    # totally_paper_num = len(alread_processing_file_list)
    # save_analysis(analysis, totally_paper_num==1, args)
