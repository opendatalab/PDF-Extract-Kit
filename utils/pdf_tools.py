import os

from modules.extract_pdf import load_pdf_fitz

from .logging_config import setup_logging

# Apply the logging configuration
logger = setup_logging('pdf_tools')


def check_pdf(pdf_path: str):
    """
    Checks if the given path is a directory or a single PDF file.
    If it is a directory, it retrieves all the PDF files within the directory.
    Otherwise, it treats the path as a single PDF file.

    :param pdf_path: The path to the directory or PDF file.
    :type pdf_path: str
    :returns: A list of PDF file paths.
    :rtype: list[str]
    """
    if os.path.isdir(pdf_path):
        all_pdfs = [os.path.join(pdf_path, name) for name in os.listdir(pdf_path)]
    else:
        all_pdfs = [pdf_path]
    logger.info(f"Total files: {len(all_pdfs)}")
    return all_pdfs


def get_images(single_pdf: str, dpi: int = 200) -> list | None:
    """
    This function retrieves a list of images from a given PDF file.
    It uses the `load_pdf_fitz()` function to load the PDF and convert its contents into images.

    Parameters:
    - `single_pdf` (str): The path to the PDF file.
    - `dpi` (int): The resolution at which the PDF should be converted to images. Default is 200.

    Returns:
    - list or None: A list of images if the conversion was successful, otherwise None.

    Raises:
    - Any exceptions raised by the `load_pdf_fitz()` function are caught and logged, and the function returns None.
    """
    try:
        img_list = load_pdf_fitz(single_pdf, dpi=dpi)
    except Exception as e:
        logger.error(f"Unexpected error with PDF file '{single_pdf}': {e}")
        return None
    return img_list


def process_all_pdfs(all_pdfs: list, dpi: int = 200):
    """
    Processes a list of PDF files and yields information about each PDF file.

    Args:
        all_pdfs (list): A list of paths to PDF files.
        dpi (int, optional): DPI (dots per inch) value for converting PDF to images. Default is 200.

    Yields:
        Tuple[int, str, List]: A tuple containing the following information:
            - PDF index (int): Index of the PDF file in the list.
            - PDF path (str): Path to the PDF file.
            - PDF images (List): A list of images extracted from the PDF file.

    Notes:
        - If an error occurs while processing a PDF file, it will be skipped and the next PDF file will be processed.
        - The logger will output information about the PDF index and the number of pages in each PDF file.

    Example:
        >>> pdfs = [
        ...     'path/to/file1.pdf',
        ...     'path/to/file2.pdf',
        ... ]
        >>> for idx, pdf, images in process_all_pdfs(pdfs):
        ...     print(f"PDF index: {idx}, PDF path: {pdf}, Number of images: {len(images)}")
    """
    for idx, single_pdf in enumerate(all_pdfs):
        img_list = get_images(single_pdf, dpi)

        if img_list is None:
            continue

        logger.info(f"PDF index: {idx}, pages: {len(img_list)}")
        yield idx, single_pdf, img_list


if __name__ == '__main__':
    pdf_dir = "assets/examples/example.pdf"
    check_pdf(pdf_dir)