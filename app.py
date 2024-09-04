# refactoring pdf_extract.py
import time
import argparse

from app_tools.config import setup_logging
from app_tools.pdf import PDFProcessor
from app_tools.layout_analysis import LayoutAnalyzer
from app_tools.formula_analysis import FormulaProcessor
from app_tools.ocr_analysis import OCRProcessor
from app_tools.table_analysis import TableProcessor
from app_tools.visualize import get_visualize
from app_tools.utils import save_file

logger = setup_logging('app')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process PDF files and render output images.")
    parser.add_argument('--pdf', type=str, required=True, help="Path to the input PDF file")
    parser.add_argument('--output', type=str, default="output", help="Output directory or filename prefix (default: 'output')")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for processing (default: 128)")
    parser.add_argument('--vis', action='store_true', help="Enable visualization mode")
    parser.add_argument('--render', action='store_true', help="Enable rendering mode")
    args = parser.parse_args()
    logger.info("Arguments: %s", args)

    logger.info('Started!')
    start = time.time()
    ## ======== model init ========##
    analyzer = LayoutAnalyzer()
    formulas = FormulaProcessor()
    ocr_processor = OCRProcessor(show_log=True)
    table_processor = TableProcessor()
    logger.info(f'Model init done in {int(time.time() - start)}s!')
    ## ======== model init ========##

    start = time.time()
    pdf_processor = PDFProcessor()
    all_pdfs = pdf_processor.check_pdf(args.pdf)

    for idx, single_pdf, img_list in pdf_processor.process_all_pdfs(all_pdfs):

        doc_layout_result = analyzer.detect_layout(img_list)
        doc_layout_result = formulas.detect_recognize_formulas(img_list, doc_layout_result, args.batch_size)
        doc_layout_result = ocr_processor.recognize_ocr(img_list, doc_layout_result)
        doc_layout_result = table_processor.recognize_tables(img_list, doc_layout_result)

        basename = save_file(args.output, single_pdf, doc_layout_result)
        logger.debug(f'Save file: {basename}.json')

        if args.vis:
            logger.info("Visualization mode enabled")
            get_visualize(img_list, doc_layout_result, args.render, args.output, basename)
        else:
            logger.info("Visualization mode disabled")

    logger.info(f'Finished! time cost: {int(time.time() - start)} s')
    logger.info('----------------------------------------')
