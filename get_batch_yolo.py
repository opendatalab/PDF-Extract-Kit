from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils import ARGV
from ultralytics.data.augment import LetterBox
from ultralytics.utils.checks import check_imgsz
def build_mfd_predictor(
        self,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ):
        

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
class mfd_process:
    def __init__(self, imgsz, stride, pt):
        self.imgsz = imgsz
        
        self.stride = stride
        self.pt = pt
    def __call__(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        imgsz = check_imgsz(self.imgsz, stride=self.stride, min_dim=2)
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(imgsz, auto=same_shapes and self.pt, stride=self.stride)
        return [letterbox(image=x) for x in im]
  
def fastpostprocess(self, preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""
    
    preds = ops.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det,
        classes=self.args.classes,
    )
    # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
    #     orig_imgs = ops.convert_torch2numpy_batch(orig_imgs) ## <-- this step only convert the channel order back and to cpu and to uni8, no need this 
    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        #pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape) # <-- lets do it outside since now we will feed normlized batch
        img_path = self.batch[0][i]
        results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    return results 
# import ultralytics
# ultralytics.models.yolo.detect.DetectionPredictor.postprocess = fastpostprocess
from ultralytics import YOLO
import os
def get_batch_YOLO_model(model_configs, batch_size)->YOLO:
    weight_path = model_configs['model_args']['mfd_weight']
    engine_weight= model_configs['model_args']['mfd_weight'][:-3]+f'.b{batch_size}.engine'
    if os.path.exists(engine_weight):
        mfd_model = YOLO(engine_weight,task='detect')
    else:
        mfd_model =  YOLO(weight_path)
    #mfd_model = YOLO(engine_weight,task='detect')
    #mfd_model =  YOLO(weight_path)
    img_size  = model_configs['model_args']['img_size']
    img_size  = (1888,1472) # <---- please fix use this, in normal YOLO assign it is automatively correct, but when using .engine file, it is not correct
    
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    build_mfd_predictor(mfd_model ,  imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)
    return mfd_model

  
