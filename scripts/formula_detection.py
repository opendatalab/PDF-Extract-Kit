import os
import sys
import os.path as osp
import argparse

sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
import pdf_extract_kit.tasks

TASK_NAME = 'formula_detection'


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
    model_formula_detection = task_instances[TASK_NAME]

    # for image detection
    detection_results = model_formula_detection.predict_images(input_data, result_path)

    # for pdf detection
    # detection_results = model_formula_detection.predict_pdfs(input_data, result_path)

    # print(detection_results)
    print(f'The predicted results can be found at {result_path}')


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
