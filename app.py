# refactoring pdf_extract.py
import time

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

    # Params
    pdf_path: str = '1706.03762.pdf'
    output_dir: str = 'output'
    batch_size: int = 128
    vis: bool = False
    render: bool = False

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
    all_pdfs = pdf_processor.check_pdf(pdf_path)

    for idx, single_pdf, img_list in pdf_processor.process_all_pdfs(all_pdfs):

        doc_layout_result = analyzer.detect_layout(img_list)
        doc_layout_result = formulas.detect_recognize_formulas(img_list, doc_layout_result)
        doc_layout_result = ocr_processor.recognize_ocr(img_list, doc_layout_result)
        doc_layout_result = table_processor.recognize_tables(img_list, doc_layout_result)

        basename = save_file(output_dir, single_pdf, doc_layout_result)
        logger.debug(f'Save file: {basename}.json')

        if vis:
            get_visualize(img_list, doc_layout_result, render, output_dir, basename)

    logger.info(f'Finished! time cost: {int(time.time() - start)} s')
    logger.info('----------------------------------------')