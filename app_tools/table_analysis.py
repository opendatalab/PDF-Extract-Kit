import time
import torch
import gc
from PIL import Image
from struct_eqtable import build_model

from app_tools.config import load_config, setup_logging


class TableProcessor:
    """
    This class represents a Table Processor that is used for table recognition in documents. The `TableProcessor` class has the following methods:

    - `__init__(self, config_path: str = None)`: Initializes a Table Processor object. It takes an optional `config_path` parameter which specifies the path to a configuration file. If no `config_path` is provided, the default configuration will be used. This method also initializes a logger and loads the configuration. It calls the `_init_tr_model` method to initialize the table recognition model.

    - `_init_tr_model(self)`: Initializes the table recognition model. It retrieves the model weight, maximum time, and device from the configuration. It then builds the model using the specified weight and maximum time. If the device is set to 'cuda', the model is moved to the GPU. The initialized table recognition model is returned.

    - `recognize_tables(self, img_list: list, doc_layout_result: list) -> list`: Performs table recognition on a list of images. It takes `img_list` as input, which is a list of images to process. It also takes `doc_layout_result`, which is a list containing layout results for each image. This method iterates over each image and its corresponding layout results. If a layout result has a 'category_id' of 5, indicating it is a table, the image is cropped and passed to the table recognition model. The output of the model is stored in the layout result as 'latex'. If the table recognition operation takes longer than the maximum time specified in the configuration, the layout result will have a 'timeout' flag set to True. The updated `doc_layout_result` is returned.

    - `clear_memory(self)`: Clears the table recognition model from memory. This method deletes the table recognition model, clears the GPU cache, and performs garbage collection to free up memory.

    Note: The code does not include the implementation of functions like `setup_logging`, `load_config`, `build_model`, and the import statements for the necessary libraries.
    """
    def __init__(self, config_path: str = None):
        """
        This class initializes an instance of the software with the provided configuration path.

        Attributes:
        - logger: The logger instance for logging debug and error messages.
        - config: The configuration object that stores the loaded configuration from the provided path.
        - tr_model: The initialized text recognition model.

        Methods:
        - __init__(self, config_path: str = None): Initializes an instance of the software with the provided configuration path.
        - _init_tr_model(self): Initializes the text recognition model.

        Note: This class requires a logging setup function called 'setup_logging' to be defined and a config loading function called 'load_config' to be defined.
        """
        self.logger = setup_logging('table_analysis')
        self.config = load_config(config_path) if config_path else load_config()
        self.tr_model = self._init_tr_model()

    def _init_tr_model(self):
        """
        Initializes the translation model.

        This method initializes the translation model by setting the weight, maximum time, and device attributes based on the provided configuration. It also builds the model using the `build_model` function.

        Parameters:
            - self : object
                The instance of the class that this method is called upon.

        Returns:
            - tr_model : object
                The initialized translation model.
        """
        weight = self.config['model_args']['tr_weight']
        max_time = self.config['model_args']['table_max_time']
        device = self.config['model_args']['device']

        tr_model = build_model(weight, max_new_tokens=4096, max_time=max_time)
        if device == 'cuda':
            tr_model = tr_model.cuda()
        return tr_model

    def recognize_tables(self, img_list: list, doc_layout_result: list) -> list:
        """
        This method recognizes tables in a list of images based on the document layout results.

        Parameters:
        - img_list: a list of images to perform table recognition on
        - doc_layout_result: a list containing layout details of the document

        Returns:
        - A modified version of doc_layout_result with table recognition results added

        The method initializes the table recognition process and sets the maximum time for the recognition. It then iterates through each image in img_list and retrieves the layout details for that image from doc_layout_result.

        For each layout detail, if the category_id is 5 (indicating that it is a table), the method crops the image based on the polygon coordinates of the layout detail and performs table recognition on the cropped image.

        The table recognition operation might take significant time, so a timeout check is performed to determine if the recognition process exceeds the maximum time. If it does, the timeout flag is set to True in the layout detail.

        The recognized LaTeX output is assigned to the "latex" property of the layout detail.

        Finally, the method logs the completion of the table recognition process and returns the modified doc_layout_result.

        Note: The method uses torch and PIL libraries for image processing and table recognition.
        """
        max_time = self.config['model_args']['table_max_time']

        self.logger.debug('Table recognition - init')
        start_0 = time.time()

        for idx, image in enumerate(img_list):
            pil_img = Image.fromarray(image)
            single_page_res = doc_layout_result[idx]['layout_dets']

            for jdx, res in enumerate(single_page_res):
                if int(res['category_id']) == 5:  # Perform table recognition
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    crop_box = [xmin, ymin, xmax, ymax]
                    cropped_img = pil_img.crop(crop_box)

                    start = time.time()
                    with torch.no_grad():
                        start_1 = time.time()
                        output = self.tr_model(cropped_img)  # This operation might take significant time
                        self.logger.debug(f'{idx} - {jdx} tr_model generate in: {round(time.time() - start_1, 2)}s')

                    if (time.time() - start) > max_time:
                        res["timeout"] = True
                    res["latex"] = output[0]

        self.logger.info(f'Table recognition done in: {round(time.time() - start_0, 2)}s')

        return doc_layout_result

    def clear_memory(self):
        """
        Clears the table recognition model from memory.

        This method clears the table recognition model from memory by deleting the model object and releasing the memory occupied by the model. It also clears the CUDA cache and performs garbage collection.

        Parameters:
            None

        Returns:
            None

        Example:
            clear_memory()

        """
        self.logger.info('Clearing the table recognition model from memory.')

        if self.tr_model is not None:
            del self.tr_model
            self.tr_model = None

            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info('Table recognition model successfully cleared from memory.')

