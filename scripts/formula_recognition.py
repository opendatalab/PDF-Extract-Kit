import os
import sys
import os.path as osp
import argparse

sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
import pdf_extract_kit.tasks

TASK_NAME = 'formula_recognition'


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

    # formula_detection_task
    model_formula_recognition = task_instances[TASK_NAME]

    # for image detection
    recognition_results = model_formula_recognition.predict(input_data, result_path)


    print('Recognition results are as follows:')
    for id, math in enumerate(recognition_results):
        print(str(id+1)+': ', math)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
