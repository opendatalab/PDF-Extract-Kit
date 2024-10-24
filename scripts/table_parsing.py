import os
import sys
import os.path as osp
import argparse

sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
import pdf_extract_kit.tasks

TASK_NAME = 'table_parsing'


def parse_args():
    parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    return parser.parse_args()

def main(config_path):
    config = load_config(config_path)
    task_instances = initialize_tasks_and_models(config)

    # get input and output path from config
    input_data = config.get('inputs', None)
    result_path = config.get('outputs', 'outputs'+'/'+TASK_NAME)

    # table_parsing_task
    model_table_parsing = task_instances[TASK_NAME]

    # for image detection
    parsing_results = model_table_parsing.predict(input_data, result_path)


    print('Table Parsing results are as follows:')
    for id, result in enumerate(parsing_results):
        print(str(id+1)+':\n', result)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
