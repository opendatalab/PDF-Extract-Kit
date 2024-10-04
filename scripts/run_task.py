import os
import sys
import os.path as osp
import argparse

sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
import pdf_extract_kit.tasks  # 确保所有任务模块被导入


def parse_args():
    parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    return parser.parse_args()

def main(config_path):
    config = load_config(config_path)
    task_instances = initialize_tasks_and_models(config)

    # 从配置文件中获取输入数据路径
    input_data = config.get('inputs', None)
    result_path = config.get('outputs', 'outputs')

    # formula_detection_task
    model_formula_detection = task_instances['formula_detection']
    detection_results = model_formula_detection.predict(input_data, result_path)
    print(detection_results)

    # formula_recognition_task
    # model_formula_recognition = task_instances['formula_recognition']
    # recognition_results = model_formula_recognition.predict(input_data, result_path)

    # for id, math in enumerate(recognition_results):
    #     print(str(id+1)+': ', math)

    # results = task_instance.run(input_data)
    # print(results)

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
