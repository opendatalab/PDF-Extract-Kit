#!/usr/bin/python
# -*- coding: UTF-8 -*-

import io
import os
import yaml
import json
import argparse
from tqdm import tqdm
from modules.s3_utils import *


def get_s3_file_list(s3_path, post_fix='.jsonl'):
    s3_cfg = get_s3_cfg_by_bucket(s3_path)
    s3_client = get_s3_client('', s3_cfg)
    # file_list = [os.path.basename(item) for item in list_s3_objects(s3_client, s3_path) if item.endswith(post_fix)]
    file_list = [item.replace(s3_path, "") for item in list_s3_objects(s3_client, s3_path, recursive=True) if item.endswith(post_fix)]
    return file_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--count', action='store_true')
    args = parser.parse_args()
    
    with open('global_args.yaml') as f:
        global_args = yaml.load(f, Loader=yaml.FullLoader)
    input_list_file = global_args['run_args']['input_list_file']
    output_list_file = global_args['run_args']['output_list_file']
    input_s3_dir = global_args['s3_args']['input_s3_dir']
    output_s3_dir = global_args['s3_args']['output_s3_dir']
        
    input_list = get_s3_file_list(input_s3_dir)
    print('=> total input list:', len(input_list))
    print('=> top 5 names:')
    for name in input_list[0:5]:
        print('  => :', name)
            
    output_list = get_s3_file_list(output_s3_dir)
    print('total output list:', len(output_list))
    print('=> top 5 names:')
    for name in output_list[0:5]:
        print('  => :', name)

    if args.update:
        # update参数为True的时候，会同步更新列表文件，否则只会展示出来输入和输出的数量。
        try:
            os.remove(input_list_file)
            os.remove(output_list_file)
        except:
            pass
        
        with open(input_list_file, 'a') as f:
            for name in input_list:
                f.write(name+'\n')
        with open(output_list_file, 'a') as f:
            for name in output_list:
                f.write(name+'\n')
        print("=> input and output list updated.")
        
    if args.count:
        print("=> counting valid rate in output files ...")
        s3_cfg = get_s3_cfg_by_bucket(output_s3_dir)
        s3_client = get_s3_client('', s3_cfg)
        total = 0
        res = 0
        for idx, jsonl_file in tqdm(enumerate(output_list)):
            content = read_s3_object_content(s3_client, os.path.join(output_s3_dir, jsonl_file))
            all_datas = io.BytesIO(content)
            bad = 0
            good = 0
            for line in all_datas:
                if not line:
                    continue
                pdf_info = json.loads(line)
                if pdf_info.get("doc_layout_result", False):
                    good += 1
                else:
                    bad += 1
                total += 1
            res += good
            # if good == 0:
                # print(f"  =>  got a totally bad jsonl: {jsonl_file} \n")
            if idx % 200 == 0 and idx > 0:
                print(f"=> total valid rate at {idx}/{len(output_list)}: {res}, {total}, {round(res/total, 2)} \n")
        print(f"\n=> final valid rate: {res}, {total}, {round(res/total, 2)} \n")
        
if __name__ == '__main__':
    main()