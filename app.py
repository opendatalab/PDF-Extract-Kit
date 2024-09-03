# refactoring pdf_extract.py

import os
import json
import time

from modules.self_modify import ModifiedPaddleOCR

from utils.pdf_tools import PDFProcessor
from utils.config import setup_logging

from utils.model_tools import mfd_model_init
from utils.model_tools import mfr_model_init
from utils.model_tools import layout_model_init
from utils.model_tools import tr_model_init

from utils.recognition import formula_recognition
from utils.recognition import ocr_recognition, table_recognition
from utils.visualize import get_visualize

from utils.detection import layout_detection, formula_detection

# Apply the logging configuration
logger = setup_logging('app')


if __name__ == '__main__':

    # Params
    pdf_path: str = '1706.03762.pdf'
    output_dir: str = 'output'
    batch_size: int = 128
    vis: bool = False
    render: bool = False

    logger.info('Started!')
    start_0 = time.time()
    ## ======== model init ========##
    mfd_model = mfd_model_init()
    mfr_model, mfr_transform = mfr_model_init()
    tr_model = tr_model_init()
    layout_model = layout_model_init()
    ocr_model = ModifiedPaddleOCR(show_log=True)
    logger.info(f'Model init done in {int(time.time() - start_0)}s!')
    ## ======== model init ========##

    start_0 = time.time()

    pdf_processor = PDFProcessor()
    all_pdfs = pdf_processor.check_pdf(pdf_path)

    for idx, single_pdf, img_list in pdf_processor.process_all_pdfs(all_pdfs):

        # layout detection and formula detection
        doc_layout_result = layout_detection(img_list, layout_model)
        doc_layout_result, latex_filling_list, mf_image_list = formula_detection(img_list, doc_layout_result, mfd_model)

        # Formula recognition, collect all formula images in whole pdf file, then batch infer them.
        formula_recognition(mf_image_list, latex_filling_list, mfr_model, mfr_transform, batch_size)

        # ocr and table recognition
        doc_layout_result = ocr_recognition(img_list, doc_layout_result, ocr_model)
        doc_layout_result = table_recognition(img_list, doc_layout_result, tr_model)


        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(single_pdf)[0:-4]
        logger.debug(f'Save file: {basename}.json')
        with open(os.path.join(output_dir, f'{basename}.json'), 'w') as f:
            json.dump(doc_layout_result, f)

        if vis:
            get_visualize(img_list, doc_layout_result, render, output_dir, basename)

    logger.info(f'Finished! time cost: {int(time.time() - start_0)} s')
    logger.info('----------------------------------------')