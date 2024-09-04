

# import os
# import json
# import sqlite3
# from tqdm.auto import tqdm
# from simple_parsing import ArgumentParser
# from batch_run_utils import obtain_processed_filelist, process_files,save_analysis, BatchModeConfig
# # Path to the directory containing JSON files
# FILEROOT = "page_num_map"

# # Connect to or create the SQLite database
# conn = sqlite3.connect('page_num_map.db')
# cursor = conn.cursor()

# # Create a table to store the data
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS page_num_map (
#     key TEXT PRIMARY KEY,
#     value INTEGER
# )
# ''')
# conn.commit()
# conn.close()

# # Function to insert or update data in the database
# def insert_or_update_data(data):
#     conn = sqlite3.connect('page_num_map.db')
#     cursor = conn.cursor()
#     for key, value in data.items():
#         cursor.execute('''
#         INSERT OR REPLACE INTO page_num_map (key, value) VALUES (?, ?)
#         ''', (key, value))
#     conn.commit()
#     conn.close()
# # Iterate over each JSON file and insert data into the database


# def process_file(metadata_file, args:BatchModeConfig):
#     metadata_file = os.path.join(FILEROOT, metadata_file)
#     with open(metadata_file) as f:
#         data = json.load(f)
#         insert_or_update_data(data)

# def process_one_file_wrapper(args):
#     arxiv_path, args = args
#     return  process_file(arxiv_path,args)

# parser = ArgumentParser()
# parser.add_arguments(BatchModeConfig, dest="config")
# args = parser.parse_args()
# args = args.config   
# args.task_name = "scan"
# alread_processing_file_list = obtain_processed_filelist(args)

# results = process_files(process_one_file_wrapper, alread_processing_file_list, args)


