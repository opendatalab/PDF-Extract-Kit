import time
import cv2
import numpy as np
from PIL import Image

from modules.self_modify import ModifiedPaddleOCR

from app_tools.config import setup_logging


class OCRProcessor:
    """
    This class represents an OCR Processor.
    It is responsible for performing OCR recognition on a list of images based on certain conditions defined in the code.

        Attributes:
            logger: Logger object for logging OCR analysis.
            ocr_model: Instance of the ModifiedPaddleOCR class used for OCR recognition.

        Methods:
            __init__(self, show_log: bool = True)
                Initializes the OCRProcessor object with a logger and an instance of the ModifiedPaddleOCR class.

            recognize_ocr(self, img_list: list, doc_layout_result: list) -> list:
                Performs OCR recognition on a list of images based on the given document layout results.
                Returns a modified document layout result list with any newly recognized text appended to it.
    """
    def __init__(self, show_log: bool = True):
        """
        This class is responsible for initializing the OCR Analysis object.

        Attributes:
            show_log (bool): A boolean value indicating whether to display log messages. Default is True.
        """
        self.logger = setup_logging('ocr_analysis')
        self.ocr_model = ModifiedPaddleOCR(show_log=show_log)

    def recognize_ocr(self, img_list: list, doc_layout_result: list) -> list:
        """
        This method `recognize_ocr` performs Optical Character Recognition (OCR) on a list of images and appends the recognized text to the document layout result.

        Parameters:
        - `img_list` (list): A list of images in numpy array format.
        - `doc_layout_result` (list): A list containing the document layout results, each result representing a page in the document.

        Returns:
        - `doc_layout_result` (list): The updated document layout result list with recognized text appended.

        1. Converts each input image from RGB color space to BGR color space using OpenCV's `cv2.cvtColor` method.
        2. Iterates over each image and its corresponding layout details in the document layout output.
        3. For each layout detail, checks whether the category ID is 13 or 14, which correspond to formula categories.
            If found, the bounding box coordinates of the layout detail are extracted and added to the `single_page_mfdetrec_res` list.
        4. Checks whether the category ID is one of [0, 1, 2, 4, 6, 7], which represent categories that require OCR.
            If found, the bounding box coordinates are extracted, and a region of interest (ROI) is cropped from the image using the `pil_img.crop` method. This ROI image is converted back to BGR color space.
        5. The `self.ocr_model.ocr` method is called, passing the cropped image along with the `single_page_mfdetrec_res` list, to perform the OCR.
            The OCR result is returned as a list of bounding boxes and their corresponding recognized text.
        6. If the OCR result is not empty, the method iterates over each bounding box and text pair in the result.
            The four corner points of the bounding box are extracted, along with the confidence score and the recognized text.
            A new layout detail with a category ID of 15 (corresponding to the recognized text) is created, and this detail is added to the `doc_layout_result`.
        7. The time taken for the OCR recognition is recorded and the updated `doc_layout_result` list is returned.
        """
        self.logger.debug('OCR recognition - init')
        start = time.time()

        for idx, image in enumerate(img_list):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            single_page_res = doc_layout_result[idx]['layout_dets']
            single_page_mfdetrec_res = []

            for res in single_page_res:
                if int(res['category_id']) in [13, 14]:  # Categories for formula
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    single_page_mfdetrec_res.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                    })

            for res in single_page_res:
                if int(res['category_id']) in [0, 1, 2, 4, 6, 7]:  # Categories that need OCR
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    crop_box = (xmin, ymin, xmax, ymax)
                    cropped_img = Image.new('RGB', pil_img.size, 'white')
                    cropped_img.paste(pil_img.crop(crop_box), crop_box)
                    cropped_img = cv2.cvtColor(np.asarray(cropped_img), cv2.COLOR_RGB2BGR)
                    ocr_res = self.ocr_model.ocr(cropped_img, mfd_res=single_page_mfdetrec_res)[0]
                    if ocr_res:
                        for box_ocr_res in ocr_res:
                            p1, p2, p3, p4 = box_ocr_res[0]
                            text, score = box_ocr_res[1]
                            doc_layout_result[idx]['layout_dets'].append({
                                'category_id': 15,
                                'poly': p1 + p2 + p3 + p4,
                                'score': round(score, 2),
                                'text': text,
                            })

        self.logger.info(f'OCR recognition done in: {round(time.time() - start, 2)}s')

        return doc_layout_result
