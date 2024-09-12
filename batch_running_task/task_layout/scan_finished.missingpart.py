
from batch_deal_with_layout import *
version = "fix_missing_page_version1"
client=None
finished_file_list = []
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
    
    for inputs_line in tqdm(all_file_list, leave=False, position=1):

        splited_line = inputs_line.split()
        inputs_path = splited_line[0]
        filename    = os.path.basename(inputs_path)
        assert "layoutV" in inputs_path
        result_save_root = os.path.join(os.path.dirname(os.path.dirname(inputs_path)),version)
        inputs_path      = os.path.join(INPUT_LOAD_PATH,filename)

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
                #tqdm.write(f"[Skip]: existed {result_old_path} ")
                skip = True
                break
        if skip:
            finished_file_list.append(inputs_path)
    with open('scan_finished.missingpart','w') as f:
        f.write('\n'.join(finished_file_list))