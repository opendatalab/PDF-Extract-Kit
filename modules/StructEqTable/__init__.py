from .model import StructTable


def build_model(model_ckpt, **kwargs):
    
    model = StructTable(model_ckpt, **kwargs)
    return model