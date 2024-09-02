from tqdm.auto import tqdm
from multiprocessing import Pool
import numpy as np
import argparse
import os
from dataclasses import dataclass
from typing import List
@dataclass
class BatchModeConfig:
    task_name = 'temp'
    root_path : str
    
    index_part : int = 0
    num_parts : int = 1
    datapath : str = None
    savepath : str = None
    logpath : str = "analysis"
    batch_num : int = 0
    redo : bool = False
    shuffle: bool = False
    debug:bool=False
    verbose: bool = False
    ray_nodes: List[int] = None
    debug: bool = False

    def from_dict(kargs):
        return BatchModeConfig(**kargs)
    def to_dict(self):
        return self.__dict__
    


def process_files(func, file_list, args:BatchModeConfig):
    
    num_processes = args.batch_num
    if num_processes == 0:
        results = []
        for arxivpath in tqdm(file_list, desc="Main Loop:"):
            results.append(func((arxivpath, args)))
        return results
    else:
        with Pool(processes=num_processes) as pool:
            args_list = [(file, args) for file in file_list]
            results = list(tqdm(pool.imap(func, args_list), total=len(file_list), desc="Main Loop:"))
    return results

import json
def obtain_processed_filelist(args:BatchModeConfig,alread_processing_file_list=None):
    ROOT_PATH = args.root_path
    index_part= args.index_part
    num_parts = args.num_parts
    if alread_processing_file_list is None:
        if ROOT_PATH.endswith('.json'):
            with open(ROOT_PATH,'r') as f:
                alread_processing_file_list = json.load(f)
        elif os.path.isfile(ROOT_PATH):
            if ROOT_PATH.endswith('.filelist'):
                with open(ROOT_PATH,'r') as f:
                    alread_processing_file_list = [t.strip() for t in f.readlines()]
            elif ROOT_PATH.endswith('.arxivids'):
                with open(ROOT_PATH,'r') as f:
                    alread_processing_file_list = [os.path.join(args.datapath, t.strip()) for t in f.readlines()]
            else:
                alread_processing_file_list = [ROOT_PATH]
        elif os.path.isdir(ROOT_PATH):
            ### this means we will do the whole subfiles under this folder
            alread_processing_file_list = os.listdir(ROOT_PATH)
            alread_processing_file_list = [os.path.join(ROOT_PATH,t) for t in alread_processing_file_list]
        else:
            ### directly use the arxivid as input
            alread_processing_file_list = [os.path.join(args.datapath, ROOT_PATH)]

    totally_paper_num = len(alread_processing_file_list)
    if totally_paper_num > 1:
        divided_nums = np.linspace(0, totally_paper_num , num_parts+1)
        divided_nums = [int(s) for s in divided_nums]
        start_index = divided_nums[index_part]
        end_index   = divided_nums[index_part + 1]
    else:
        start_index = 0
        end_index   = 1
    args.start_index = start_index
    args.end_index   = end_index
    if args.shuffle:
        np.random.shuffle(alread_processing_file_list)
    alread_processing_file_list = alread_processing_file_list[start_index:end_index]
    
    return alread_processing_file_list


def save_analysis(analysis, debug, args):
    
    logpath = os.path.join(args.logpath,args.task_name)
    print(logpath)
    os.makedirs(logpath, exist_ok=True)
    if args.num_parts > 1:
        for key, val in analysis.items():
            print(f"{key}=>{len(val)}")
            fold = os.path.join(logpath,f"{key.lower()}.filelist.split")
            os.makedirs(fold, exist_ok=True)
            with open(os.path.join(fold,f"{args.start_index}-{args.end_index}"), 'w') as f:
                for line in val:
                    f.write(line+'\n')
    else:
        #print(analysis)
        for key, val in analysis.items():
            print(f"{key}=>{len(val)}")
            if debug:
                print(val)
            else:
                with open(os.path.join(logpath,f"{key.lower()}.filelist"), 'w') as f:
                    for line in val:
                        f.write(line+'\n')
    