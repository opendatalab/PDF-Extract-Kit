from .batch_text_detector import BatchTextDetector
class ModifiedPaddleOCR:
    def __init__(self, **kwargs):
        self.batch_det_model = BatchTextDetector()
    