from pdf_extract_kit.registry.registry import TASK_REGISTRY


@TASK_REGISTRY.register("table_parsing")
class TableParsingTask:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        # Perform layout detection on input_data
        return self.model.predict(input_data)