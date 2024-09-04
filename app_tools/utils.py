import os
import json


def save_file(output_dir, single_pdf, doc_layout_result):
    """
    This function saves the document layout result as a JSON file in a specified output directory.

    Parameters:
    - output_dir (str): The directory where the JSON file will be saved.
    - single_pdf (str): The path to the single PDF file.
    - doc_layout_result (dict): The document layout result that will be saved as a JSON file.

    Returns:
    - basename (str): The basename of the single PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(single_pdf)[0:-4]
    with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
        json.dump(doc_layout_result, f)
    return basename
