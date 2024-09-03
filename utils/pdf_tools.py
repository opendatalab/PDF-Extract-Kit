import os
from typing import List, Optional, Generator

from modules.extract_pdf import load_pdf_fitz
from utils.config import load_config, setup_logging


class PDFProcessor:
    """
    Class PDFTools

    This class provides a set of tools for working with PDF files.

    Methods:
    - __init__(config_path: Optional[str] = None): Initializes the PDFTools object.
    - load_config(config_path: Optional[str] = None) -> dict: Loads the configuration from a JSON file.
    - setup_logging(name: str) -> Logger: Sets up the logging configuration.

    Attributes:
    - config: A dictionary containing the configuration settings.
    - dpi: The DPI (dots per inch) for the PDF files.
    - logger: The logger object for logging messages.


    __init__(config_path: Optional[str] = None)
        Initializes the PDFTools object.

        Parameters:
            config_path (Optional[str]): Path to the configuration file. If None, the default configuration file will be loaded.

    load_config(config_path: Optional[str] = None) -> dict
        Loads the configuration from a JSON file.

        Parameters:
            config_path (Optional[str]): Path to the configuration file. If None, the default configuration file will be loaded.

        Returns:
            dict: A dictionary containing the configuration settings.

    setup_logging(name: str) -> Logger
        Sets up the logging configuration.

        Parameters:
            name (str): The name of the logger.

        Returns:
            Logger: The logger object for logging messages.

    Attributes:
    - config (dict): A dictionary containing the configuration settings.
    - dpi (int): The DPI (dots per inch) for the PDF files.
    - logger (Logger): The logger object for logging messages.
    """

    def __init__(self, config_path: Optional[str] = None):

        self.config = load_config(config_path) if config_path else load_config()
        self.dpi = self.config['model_args']['pdf_dpi']
        self.logger = setup_logging('pdf_tools')

    def check_pdf(self, pdf_path: str) -> List[str]:
        """
        This method is used to check if a given file path is a directory or a single PDF file.

        Parameters:
        - pdf_path (str): The file path to check. It can be either a directory or a single PDF file path.

        Returns:
        - List[str]: A list of PDF file paths.

        Example Usage:
        ```
        pdf_checker = PDFChecker()
        result = pdf_checker.check_pdf('/path/to/pdfs')
        print(result)
        ```

        Note: This method will return an empty list if no PDF files are found in the given directory or if the given file path is not a PDF file.
        """
        if os.path.isdir(pdf_path):
            all_pdfs = [os.path.join(pdf_path, name) for name in os.listdir(pdf_path) if name.endswith('.pdf')]
        else:
            all_pdfs = [pdf_path]
        self.logger.info(f"Total files: {len(all_pdfs)}")
        return all_pdfs

    def get_images(self, single_pdf: str) -> Optional[List[str]]:
        """
        This method retrieves a list of images from a single PDF file.

        Parameters:
        - single_pdf: A string representing the path to the PDF file.

        Returns:
        - Optional[List[str]]: A list of strings representing the images extracted from the PDF file. Returns None if there was an error during the extraction process.

        Raises:
        - None

        Example:
            obj = MyClass()
            images = obj.get_images('example.pdf')
        """
        try:
            img_list = load_pdf_fitz(single_pdf, self.dpi)
        except Exception as e:
            self.logger.error(f"Unexpected error with PDF file '{single_pdf}': {e}")
            return None
        return img_list

    def process_all_pdfs(self, all_pdfs: List[str]) -> Generator[tuple[int, str, List[str]], None, None]:
        """
        This method `process_all_pdfs` processes a list of PDF files and returns a generator that yields a tuple for each PDF file. The tuple contains the index of the PDF file in the list, the path of the PDF file, and a list of image paths extracted from the PDF file.

        Parameters:
        - `self`: The current instance of the class.
        - `all_pdfs`: A List of strings representing the paths of the PDF files to be processed.

        Returns:
        - `Generator[tuple[int, str, List[str]], None, None]`: A generator that yields a tuple for each PDF file. The tuple contains the index of the PDF file, the path of the PDF file, and a list of image paths extracted from the PDF file.

        Example usage:
        ```python
        pdf_processor = PDFProcessor()
        pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]
        for index, path, images in pdf_processor.process_all_pdfs(pdf_files):
            print(f"Processing PDF index: {index}")
            print(f"PDF path: {path}")
            print(f"Images: {images}")
        ```
        """
        for idx, single_pdf in enumerate(all_pdfs):
            img_list = self.get_images(single_pdf)

            if img_list is None:
                continue

            self.logger.info(f"PDF index: {idx}, pages: {len(img_list)}")
            yield idx, single_pdf, img_list