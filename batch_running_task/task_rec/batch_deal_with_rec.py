
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from rough_rec import *
import yaml
# from rough_layout_with_aync import * ## async is not safe, lets disable it 
from get_data_utils import *
RESULT_SAVE_PATH="opendata:s3://llm-pdf-text/pdf_gpu_output/scihub_shared"
#RESULT_SAVE_PATH="tianning:s3://temp/debug"
INPUT_LOAD_PATH="opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub"
import socket   
hostname= socket.gethostname()
LOCKSERVER="http://10.140.52.123:8000" if hostname.startswith('SH') else "http://paraai-n32-h-01-ccs-master-2:32453"
from datetime import datetime,timedelta
import socket   
hostname= socket.gethostname()
from batch_run_utils import BatchModeConfig, process_files,dataclass,obtain_processed_filelist
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm
import traceback

@dataclass
class BatchRECConfig(BatchModeConfig):
    image_batch_size: int = 256
    pdf_batch_size: int = 32
    num_workers: int = 4
    result_save_path: str=RESULT_SAVE_PATH
    check_lock: bool = True
    update_origin: bool = False
    compile: bool = False
    replace:bool=False
if __name__ == '__main__':
    task_name = "physics_part"
    version   = "mfr_patch_bf16"
    
    parser = ArgumentParser()
    parser.add_arguments(BatchRECConfig, dest="config")
    args = parser.parse_args()
    args = args.config   
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

    task_name = "physics_part"
    version   = "rec_fixed_final"
    layout_model = None
    mfd_model    = None
    client = None
    ocrmodel = None
    page_num_map_whole = None #get_page_num_map_whole()
    for inputs_path in tqdm(all_file_list, leave=False, position=1):
        filename    = os.path.basename(inputs_path)
        
        if args.replace:
            origin_root = os.path.dirname(inputs_path).split('/')
            task_name = origin_root[-2]
            version   = origin_root[-1]
            args.result_save_path = os.path.dirname(os.path.dirname(os.path.dirname(inputs_path)))
            args.redo = True
            args.update_origin = True
        result_save_root = os.path.join(args.result_save_path, task_name, version)
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
            os.path.join(args.result_save_path, task_name, version),
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
                filename_with_partion = f"{filename.replace('.jsonl','')}.{partion_idx:02d}_{partion_num:02d}.jsonl"
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
            if args.check_lock:
                lock_path = os.path.join(LOCKSERVER, "checklocktime", filename_with_partion)
                last_start_time = check_lock_and_last_start_time(lock_path,client)
                if last_start_time and not args.redo:
                    date_string = last_start_time
                    date_format = "%Y-%m-%d %H:%M:%S"
                    date = datetime.strptime(date_string, date_format)
                    deltatime = datetime.now() - date
                    if deltatime < timedelta(hours=0.5):
                        tqdm.write(f"[Skip]: {filename_with_partion} is locked by {date_string} created at {last_start_time} [now is {deltatime}]")
                        continue
                
                create_last_start_time_lock(os.path.join(LOCKSERVER,"createlocktime", filename_with_partion),client)

            print(f"now we deal with {inputs_path} to {result_path}")
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            
            if ocrmodel is None:
                ocrmodel = TextRecognizer(rec_args)
                if args.compile:
                    ocrmodel.net.backbone = torch.compile(ocrmodel.net.backbone)

            
            try:
                deal_with_one_dataset(inputs_path, result_path, ocrmodel,
                          #batch_size  = args.batch_size,
                          pdf_batch_size=args.pdf_batch_size, image_batch_size=args.image_batch_size,
                          num_workers = args.num_workers,
                          partion_num = partion_num,
                          partion_idx = partion_idx,update_origin=args.update_origin)
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