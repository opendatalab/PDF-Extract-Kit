import os
import sys
import os.path as osp
import argparse
from pdf2markdown import PDF2MARKDOWN

sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
from pdf_extract_kit.registry.registry import TASK_REGISTRY


TASK_NAME = 'pdf2markdown'

def parse_args():
    parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    return parser.parse_args()

def main(config_path):
    config = load_config(config_path)
    task_instances = initialize_tasks_and_models(config)

    # get input and output path from config
    input_data = config.get('inputs', None)
    result_path = config.get('outputs', 'outputs/pdf_extract')
    visualize = config.get('visualize', False)
    merge2markdown = config.get('merge2markdown', False)

    layout_model = task_instances['layout_detection'].model if 'layout_detection' in task_instances else None
    mfd_model = task_instances['formula_detection'].model if 'formula_detection' in task_instances else None
    mfr_model = task_instances['formula_recognition'].model if 'formula_recognition' in task_instances else None
    ocr_model = task_instances['ocr'].model if 'ocr' in task_instances else None
    
    pdf_extract_task = TASK_REGISTRY.get(TASK_NAME)(layout_model, mfd_model, mfr_model, ocr_model)
    extract_results = pdf_extract_task.process(input_data, save_dir=result_path, visualize=visualize, merge2markdown=merge2markdown)

    print(f'Task done, results can be found at {result_path}')

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
