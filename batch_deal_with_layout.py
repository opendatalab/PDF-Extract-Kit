
from rough_layout import *
from get_data_utils import *
RESULT_SAVE_PATH="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared"
#RESULT_SAVE_PATH="tianning:s3://temp/debug"
INPUT_LOAD_PATH="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"
LOCKSERVER="http://10.140.52.123:8000"
from datetime import datetime,timedelta
if __name__ == '__main__':
    import argparse, logging, os
    import numpy as np
    from tqdm.auto import tqdm
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--index_part", type=int, default=0)
    parser.add_argument('--num_parts', type=int, default=1)

    parser.add_argument('--verbose', '-v', action='store_true', help='', default=False)
    parser.add_argument('--redo',  action='store_true', help='', default=False)
    parser.add_argument('--do_not_det',  action='store_true', help='', default=False)
    parser.add_argument('--do_rec',  action='store_true', help='', default=False)
    parser.add_argument('--shuffle',  action='store_true', help='', default=False)
    parser.add_argument('--inner_batch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--result_save_path', type=str, default=RESULT_SAVE_PATH)
    args = parser.parse_args()
    root_path = args.root_path
    if os.path.isdir(root_path):
        ###### do not let the program scan the dir ########
        ##### thus the only dir case is that use a dir path like data/archive_json/quant-ph_0004055
        raise NotImplementedError
        all_file_list = [root_path]
    elif os.path.isfile(root_path):
        if root_path.endswith('.jsonl'):
            all_file_list = [root_path]
        else:
            with open(root_path,'r') as f:
                all_file_list = [t.strip() for t in f.readlines()]
    else:
        raise NotImplementedError
    index_part= args.index_part
    num_parts = args.num_parts 
    totally_paper_num = len(all_file_list)
    if totally_paper_num > 1:
        divided_nums = np.linspace(0, totally_paper_num - 1, num_parts+1)
        divided_nums = [int(s) for s in divided_nums]
        start_index = divided_nums[index_part]
        end_index   = divided_nums[index_part + 1]
    else:
        start_index = 0
        end_index   = 1
        verbose = True
    if args.shuffle:
        np.random.shuffle(all_file_list)

    all_file_list = all_file_list[start_index: end_index]
    
    if len(all_file_list)==0:
        print(f"Index {index_part} has no file to process")
        exit()
    
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)

    img_size  = model_configs['model_args']['img_size']
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device    = model_configs['model_args']['device']
    dpi       = model_configs['model_args']['pdf_dpi']

    task_name = "layoutV2"
    layout_model = None
    mfd_model    = None
    client = None
    ocrmodel = None
    for inputs_path in tqdm(all_file_list, leave=False, position=1):
        filename    = os.path.basename(inputs_path)
        result_path = os.path.join(args.result_save_path, task_name, "result", filename)
        if inputs_path.startswith('s3'):
            inputs_path = "opendata:"+inputs_path
        if result_path.startswith('s3'):
            result_path = "opendata:"+result_path
        # assert inputs_path.startswith('opendata:s3')
        # assert result_path.startswith('opendata:s3')
        if client is None:
            client = build_client()
        
        
        skip = False
        for result_old_path in [
            os.path.join(args.result_save_path, "layoutV2", "result", filename),
            os.path.join("opendata:s3://llm-pdf-text/pdf_gpu_output/ebook_index_v4/scihub/v001/scihub/", filename),
            os.path.join(args.result_save_path, "layoutV1", "result", filename),
        ]:
            
            if check_path_exists(result_old_path,client) and not args.redo:
                tqdm.write(f"[Skip]: existed {result_old_path} ")
                skip = True
                break
        if skip:continue

        if not check_path_exists(inputs_path,client):
            tqdm.write(f"[Skip]: no {inputs_path} ")
            continue
        
        lock_path = os.path.join(LOCKSERVER, "checklocktime", filename)
        last_start_time = check_lock_and_last_start_time(lock_path,client)
        if last_start_time and not args.redo:
            date_string = last_start_time
            date_format = "%Y-%m-%d %H:%M:%S"
            date = datetime.strptime(date_string, date_format)
            deltatime = datetime.now() - date
            if deltatime < timedelta(hours=1):
                tqdm.write(f"[Skip]: {filename} is locked by {date_string} created at {last_start_time} [now is {deltatime}]")
                continue
        
        create_last_start_time_lock(os.path.join(LOCKSERVER,"createlocktime", filename),client)

        if layout_model is None:layout_model = get_layout_model(model_configs)
        if mfd_model    is None:mfd_model    = get_batch_YOLO_model(model_configs,batch_size=args.inner_batch_size) 
        if ocrmodel is None:ocrmodel = ModifiedPaddleOCR(show_log=True)
        print(f"now we deal with {inputs_path} to {result_path}")
        try:
            deal_with_one_dataset(inputs_path, result_path, 
                                    layout_model, mfd_model,  ocrmodel=ocrmodel, 
                                    inner_batch_size=args.inner_batch_size, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    do_text_det = not args.do_not_det,
                                    do_text_rec = args.do_rec)
            print(f"finish dealing with {inputs_path} to {result_path}")
        except:
            traceback.print_exc()
            tqdm.write(f"[Error]: {inputs_path} to {result_path}")
