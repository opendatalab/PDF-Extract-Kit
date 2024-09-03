import os
import json


def save_file(output_dir, single_pdf, doc_layout_result):
    """
    Save the document layout result as a JSON file in the specified output directory.

    :param output_dir: The directory where the JSON file should be saved.
    :param single_pdf: The path of the single PDF file.
    :param doc_layout_result: The document layout result to be saved.
    :return: The base name of the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(single_pdf)[0:-4]
    with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
        json.dump(doc_layout_result, f)
    return basename
