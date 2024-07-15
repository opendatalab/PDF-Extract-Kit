import os
import json
import argparse
from natsort import natsorted
from ultralytics.models.yolo.detect import MFDValidator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='yolov8l.yaml', help='model.yaml path')
    parser.add_argument('--imsize', type=int, default=1280, help='image sizes')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--cfg1', type=str, default='', help='Yaml file of validation')
    parser.add_argument('--cfg2', type=str, default='', help='Yaml file of validation')
    args = parser.parse_args()
    if args.weight.endswith("/"):
        args.weight = args.weight[0:-1]
    print(args)
    
    if args.weight.endswith(".pt"): ## 评测单个模型，可视化box
        model_name = os.path.basename(os.path.dirname(os.path.dirname(args.weight)))
        
        input_args1 = dict(
            model=args.weight, 
            data=args.cfg1,
            imgsz=args.imsize,
            conf=0.25,
            iou=0.45)
        eval_name1 = args.cfg1.split('.')[0]
        vis_dir = f"/mnt/hwfile/opendatalab/ouyanglinke/PDF_Formula/vis_v8/{eval_name1}--{model_name}"
        validator1 = MFDValidator(args=input_args1, save_dir=vis_dir)
        res1 = validator1()
        
        if args.cfg2:
            input_args2 = dict(
                model=args.weight, 
                data=args.cfg2,
                imgsz=args.imsize,
                conf=0.25,
                iou=0.45)
            eval_name2 = args.cfg2.split('.')[0]
            vis_dir = f"/mnt/hwfile/opendatalab/ouyanglinke/PDF_Formula/vis_v8/{eval_name2}--{model_name}"
            validator2 = MFDValidator(args=input_args2, save_dir=vis_dir)
            res2 = validator2()
        else:
            res2 = False
        if res1 and res2:
            print("metrics:", [res1['AP50'], res1['AR50'], res2['AP50'], res2['AR50']])
        elif res1:
            print("metrics:", [res1['AP50'], res1['AR50']])
        else:
            print("metrics:", [0, 0, 0, 0])
        
    else:   ## 评测多个模型，不可视化，且找出best.pt
        best_score = -1
        best_metrics = None
        best_model = None
        epoch_eval_results = {"epoch_res":{}}
        for model_name in natsorted(os.listdir(args.weight)):
            model_path = os.path.join(args.weight, model_name)
            if not "epoch" in model_name:
                continue
            epoch = int(model_name[5:-3])
            print("==> eval at epoch", epoch)
            
            input_args1 = dict(
                model=model_path, 
                data=args.cfg1,
                imgsz=args.imsize,
                conf=0.25,
                iou=0.45)
            validator1 = MFDValidator(args=input_args1, save_dir="runs/vis")
            res1 = validator1(vis_box=False)
            
            if args.cfg2:
                input_args2 = dict(
                    model=model_path, 
                    data=args.cfg2,
                    imgsz=args.imsize,
                    conf=0.25,
                    iou=0.45)
                validator2 = MFDValidator(args=input_args2, save_dir="runs/vis")
                res2 = validator2(vis_box=False)
            else:
                res2 = False
            
            if res1 and res2:
                model_score = 0.2*res1['AP50'] + 0.3*res1['AR50'] + 0.2*res2['AP50'] + 0.3*res2['AR50']
                epoch_eval_results["epoch_res"][epoch] = {
                    "score": model_score, 
                    "metrics": [res1['AP50'], res1['AR50'], res2['AP50'], res2['AR50']]
                }
            elif res1:
                model_score = 0.4*res1['AP50'] + 0.6*res1['AR50']
                epoch_eval_results["epoch_res"][epoch] = {
                    "score": model_score, 
                    "metrics": [res1['AP50'], res1['AR50']]
                }
            else:
                model_score = 0
                epoch_eval_results["epoch_res"][epoch] = {
                    "score": model_score, 
                    "metrics": [0, 0, 0, 0]
                }
            
            if model_score > best_score:
                best_score = model_score
                best_model = model_name
                if res1 and res2:
                    best_metrics = [res1['AP50'], res1['AR50'], res2['AP50'], res2['AR50']]
                elif res1:
                    best_metrics = [res1['AP50'], res1['AR50']]

        print("best epoch:", best_model, "metrics:", best_metrics)
        
        epoch_eval_results["best_score"] = best_score
        epoch_eval_results["best_epoch"] = int(best_model[5:-3])
        epoch_eval_results["best_metrics"] = best_metrics
        with open(os.path.join(os.path.dirname(args.weight), "epoch_eval_results.json"), "w") as f:
            f.write(json.dumps(epoch_eval_results, indent=2))