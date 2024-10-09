
from batch_deal_with_layout import *

if __name__ == '__main__':

    from tqdm.auto import tqdm
    import traceback
    parser = ArgumentParser()
    parser.add_arguments(BatchLayoutConfig, dest="config")
    args = parser.parse_args()
    args = args.config   
    #args.check_lock = hostname.startswith('SH')
    assert not args.async_mode, "async_mode is not safe, please disable it"
    all_file_list = obtain_processed_filelist(args)
    
    if len(all_file_list)==0:
        exit()
    
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)

    img_size  = model_configs['model_args']['img_size']
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device    = model_configs['model_args']['device']
    dpi       = model_configs['model_args']['pdf_dpi']


    version = "fix_missing_page_version3"
    layout_model = None
    mfd_model    = None
    client = None
    ocrmodel = None
    page_num_map_whole = None #get_page_num_map_whole()
    for inputs_line in tqdm(all_file_list, leave=False, position=1):

        splited_line = inputs_line.strip().split()
        inputs_path = splited_line[0]
        json_str = " ".join(splited_line[1:])
   
        page_num_for_name = json.loads(json_str)

        if os.path.exists(CURRENT_END_SIGN):
            break
        filename    = os.path.basename(inputs_path)
        #assert "layoutV" in inputs_path
        result_save_root = os.path.join(os.path.dirname(os.path.dirname(inputs_path)),version)
        #inputs_path = os.path.join(INPUT_LOAD_PATH,filename)

        if inputs_path.startswith('s3'):
            inputs_path = "opendata:"+inputs_path
        # assert inputs_path.startswith('opendata:s3')
        # assert result_path.startswith('opendata:s3')
        if client is None:
            client = build_client()
        
        if not check_path_exists(inputs_path,client):
            tqdm.write(f"[Skip]: no {inputs_path} ")
            continue

        POSSIABLE_RESULT_SAVE_DIR_LIST=[
            result_save_root,
            os.path.join("opendata:s3://llm-pdf-text/pdf_gpu_output/ebook_index_v4/scihub/v001/scihub/"),
        ]

        skip = False
        for result_old_dir in POSSIABLE_RESULT_SAVE_DIR_LIST:
            result_old_path = os.path.join(result_old_dir, filename)
            if check_path_exists(result_old_path,client) and not args.redo:
                tqdm.write(f"[Skip]: existed {result_old_path} ")
                skip = True
                break
        if skip:continue

        
        
        partion_num = 1
        for partion_idx in range(partion_num):
            if partion_num > 1:
                filename_with_partion = f"{filename.replace('.jsonl','')}.{partion_idx}_{partion_num}.jsonl"
            else:
                filename_with_partion = filename

            skip = False
            for result_old_dir in POSSIABLE_RESULT_SAVE_DIR_LIST:
                result_old_path = os.path.join(result_old_dir, filename_with_partion)
                if not args.redo and check_path_exists(result_old_path,client):
                    tqdm.write(f"[Skip]: existed {result_old_path} ")
                    skip = True
                    break
            if skip:continue

            
            result_path = os.path.join(result_save_root, filename_with_partion)
            if args.use_lock:
                lock_path = os.path.join(LOCKSERVER, "checklocktime", filename_with_partion)
                last_start_time = check_lock_and_last_start_time(lock_path,client)
                if last_start_time and not args.redo:
                    date_string = last_start_time
                    date_format = "%Y-%m-%d %H:%M:%S"
                    date = datetime.strptime(date_string, date_format)
                    deltatime = datetime.now() - date
                    if deltatime < timedelta(hours=0.1):
                        tqdm.write(f"[Skip]: {filename_with_partion} is locked by {date_string} created at {last_start_time} [now is {deltatime}]")
                        continue
                
                create_last_start_time_lock(os.path.join(LOCKSERVER,"createlocktime", filename_with_partion),client)

            print(f"now we deal with {inputs_path} to {result_path}")
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            if args.debug:raise
            if layout_model is None:layout_model = get_layout_model(model_configs,args.accelerated_layout)
            if mfd_model    is None:mfd_model    = get_batch_YOLO_model(model_configs,batch_size=args.inner_batch_size,use_tensorRT=args.accelerated_mfd)
            if ocrmodel is None:ocrmodel = ModifiedPaddleOCR(show_log=True)
            
            try:
                deal_with_page_info_dataset_for_missing_page(inputs_path, result_path, 
                                        layout_model, mfd_model,  ocrmodel=ocrmodel, 
                                        inner_batch_size=args.inner_batch_size, 
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        do_text_det = not args.do_not_det,
                                        do_text_rec = args.do_rec,
                                        partion_num = partion_num,
                                        partion_idx = partion_idx,page_num_for_name=page_num_for_name
                                        )
                print(f"""
=========================================
finish dealing with {result_path}
=========================================
                      """)
            except:
                traceback.print_exc()
                tqdm.write(f"[Error]: {filename_with_partion} failed")
            finally:
                pass