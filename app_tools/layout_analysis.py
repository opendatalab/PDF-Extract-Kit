import time
import gc
import torch
from typing import Optional

from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from app_tools.config import setup_logging, load_config


class LayoutAnalyzer:
    """
        class LayoutAnalyzer:
        This class analyzes the layout of documents by detecting the layout of each page in a document image.

        Attributes:
            logger: The logger object for logging debug, info, and error messages.
            config: The configuration settings for the layout analysis.
            model: The layout detection model.

        Methods:
            __init__(self, config_path: Optional[str] = None)
                Constructs a LayoutAnalyzer object.

            _init_model(self) -> Layoutlmv3_Predictor
                Initializes the layout detection model.

            detect_layout(self, img_list: list) -> list
                Detects the layout of multiple images.

            clear_model(self)
                Clears the layout detection model from memory.
    """
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes an instance and init model.

        Args:
            config_path (Optional[str]): The path to the configuration file. Defaults to None.
        """
        self.logger = setup_logging('layout_analysis')
        self.config = load_config(config_path) if config_path else load_config()
        self.model = self._init_model()

    def _init_model(self) -> Layoutlmv3_Predictor:
        """
        Initializes and returns an instance of the `Layoutlmv3_Predictor` class.

        Parameters:
            - self: The current object.

        Returns:
            A `Layoutlmv3_Predictor` object initialized with the specified `weight` value from the configuration.

        """
        weight = self.config['model_args']['layout_weight']
        model = Layoutlmv3_Predictor(weight)
        return model

    def detect_layout(self, img_list: list) -> list:
        """
        This method `detect_layout` is used to detect the layout of a list of images.

        Parameters:
        - `img_list`: A list of images to detect the layout from.

        Returns:
        - A list of layout results.

        Raises:
        - `ValueError`: If the model is not initialized. Please call `init_model` before `detect_layout`.

        The method performs the following steps:
        1. It checks if the model is initialized. If not, it raises a `ValueError`.
        2. It initializes an empty list `doc_layout_result` to store the layout results.
        3. It logs a debug message indicating the start of layout detection.
        4. It starts a timer to measure the time taken for layout detection.
        5. It iterates over each image in the `img_list`.
           a. It gets the height and width of the image.
           b. It passes the image to the model for layout detection.
           c. It adds additional information to the layout result, such as page number, height, and width.
           d. It appends the layout result to the `doc_layout_result` list.
           e. It deletes the layout result and clears the GPU memory.
        6. It logs a debug message indicating the completion of layout detection and the time taken.

        Example usage:
        ```
        layout_detector = LayoutDetector()
        layout_detector.init_model()
        results = layout_detector.detect_layout([image1, image2, image3])
        ```
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Please call `init_model` before `detect_layout`.")

        doc_layout_result = []

        self.logger.debug('Layout detection - init')
        start = time.time()

        for idx, image in enumerate(img_list):
            img_h, img_w = image.shape[0], image.shape[1]

            layout_res = self.model(image, ignore_catids=[])

            layout_res['page_info'] = {
                'page_no': idx,
                'height': img_h,
                'width': img_w
            }
            doc_layout_result.append(layout_res)

            del layout_res
            torch.cuda.empty_cache()
            gc.collect()

        self.logger.debug(f'Layout detection done in {round(time.time() - start, 2)}s!')

        return doc_layout_result

    def clear_model(self):
        """
        This method clears the model from memory by deleting the model object and freeing up GPU memory using torch.cuda.empty_cache().
        It also collects garbage to release any unreferenced memory.

        Example usage:
            obj.clear_model()
        """
        self.logger.info('Clearing the model from memory.')

        if self.model is not None:
            del self.model
            self.model = None

            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info('Model successfully cleared from memory.')
