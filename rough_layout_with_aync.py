from rough_layout import *
import asyncio
async def deal_with_one_dataset(pdf_path, result_path, layout_model, mfd_model, 
                          ocrmodel=None, inner_batch_size=4, 
                          batch_size=32,num_workers=8,
                          timer=Timers(False)):
    do_text_det = True
    do_text_rec = False

    assert do_text_det
    dataset    = PDFImageDataset(pdf_path,layout_model.predictor.aug,layout_model.predictor.input_format,
                                 
                                 mfd_pre_transform=mfd_process(mfd_model.predictor.args.imgsz,mfd_model.predictor.model.stride,mfd_model.predictor.model.pt),
                                 det_pre_transform=ocrmodel.batch_det_model.prepare_image,
                                 return_original_image=do_text_rec
                                 )
   
    
    collate_fn = custom_collate_fn if do_text_rec else None
    dataloader = DataLoader(dataset, batch_size=batch_size,collate_fn=collate_fn, 
                            num_workers=num_workers,pin_memory=True, pin_memory_device='cuda',
                            prefetch_factor=3)        
    queue = asyncio.Queue()
    postprocess_task = asyncio.create_task(cpu_postprocess(queue, ocrmodel))

    featcher   = DataPrefetcher(dataloader,device='cuda')
    data_to_save = {}
    inner_batch_size = inner_batch_size
    pbar  = tqdm(total=len(dataset.metadata),position=2,desc="PDF Pages",leave=True)
    pdf_passed = set()
    batch = featcher.next()
    data_loading = []
    model_train  = []
    last_record_time = time.time()
    while batch is not None:

        data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
        pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        pdf_index_batch, page_ids_batch = batch["pdf_index"], batch["page_index"]
        mfd_layout_images_batch, layout_images_batch, det_layout_images_batch = batch["mfd_image"], batch["layout_image"], batch["det_images"]
        heights_batch, widths_batch = batch["height"], batch["width"]
        oimage_list = batch.get('oimage',None)
        pdf_index = set([t.item() for t in pdf_index_batch])
        new_pdf_processed = pdf_index - pdf_passed
        pdf_passed        = pdf_passed|pdf_index
        
        for j in tqdm(range(0, len(mfd_layout_images_batch), inner_batch_size),position=3,leave=False,desc="mini-Batch"):
            pdf_index  = pdf_index_batch[j:j+inner_batch_size]
            page_ids   = page_ids_batch[j:j+inner_batch_size]
            mfd_images = mfd_layout_images_batch[j:j+inner_batch_size]
            images     = layout_images_batch[j:j+inner_batch_size]
            heights    = heights_batch[j:j+inner_batch_size]
            widths     = widths_batch[j:j+inner_batch_size]
            oimages    = oimage_list[j:j+inner_batch_size] if oimage_list is not None else None
            detimages  = det_layout_images_batch[j:j+inner_batch_size]
            
            layout_pair   = (images, layout_model)
            mdf_pair      = (mfd_images, mfd_model)
            det_pair      = (detimages, ocrmodel.batch_det_model.net)
            size_pair     = (heights, widths)
            pdf_paths     = [dataset.metadata[pdf_index]['path'] for pdf_index in pdf_index]
            location_pair = (pdf_paths, page_ids)
            await process_image(queue, layout_pair, mdf_pair, det_pair, size_pair, location_pair, timer)
            # dt_boxaes_batch = gpu_inference(layout_pair, mdf_pair, det_pair, size_pair, location_pair, timer)
            # dt_boxes_list   = cpu_postprocess(dt_boxaes_batch,ocrmodel,timer)

        pbar.update(len(new_pdf_processed))

        timer.log()
        model_train.append(time.time() - last_record_time);last_record_time =time.time()
        pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        batch = featcher.next()
    await queue.put(None)  # Signal the consumer to exit
    await postprocess_task  # Wait for the consumer to finish
async def gpu_inference(layout_pair,
              mdf_pair,
              det_pair,
              size_pair,
              location_pair,
              timer):
    layout_images, layout_model = layout_pair
    mfd_images, mfd_model = mdf_pair
    detimages, det_model = det_pair
    heights, widths = size_pair
    pdf_paths, page_ids= location_pair
    with timer('get_layout'):
        layout_res = layout_model((layout_images,heights, widths), ignore_catids=[],dtype=torch.float16)
    with timer('get_mfd'):
        mfd_res    = mfd_model.predict(mfd_images, imgsz=(1888,1472), conf=0.3, iou=0.5, verbose=False)
    
    with timer('combine_layout_mfd_result'):
        rough_layout_this_batch =[]
        ori_shape_list = []
        pdf_and_page_id_this_batch = []
        for pdf_path, page_id, layout_det, mfd_det, real_input_height, real_input_width in zip(pdf_paths, page_ids, layout_res, mfd_res, heights, widths):
            mfd_height,mfd_width = mfd_det.orig_shape
            page_id= int(page_id)
            real_input_height = int(real_input_height)
            real_input_width  = int(real_input_width)
        
            layout_dets = clean_layout_dets(layout_det['layout_dets'])
            for xyxy, conf, cla in zip(mfd_det.boxes.xyxy.cpu(), 
                                    mfd_det.boxes.conf.cpu(), 
                                    mfd_det.boxes.cls.cpu()):
                xyxy =  ops.scale_boxes(mfd_images.shape[2:], xyxy, (real_input_height, real_input_width))
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    'category_id': 13 + int(cla.item()),
                    'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    'score': round(float(conf.item()), 2),
                    'latex': '',
                }
                layout_dets.append(new_item)
            ori_shape_list.append((real_input_height, real_input_width))
            pdf_and_page_id_this_batch.append((pdf_path, page_id))
            rough_layout_this_batch.append(layout_dets)
            assert real_input_height == 1920
            assert real_input_width  == 1472
        

    with timer('text_detection/collect_for_line_detect'):
        canvas_tensor_this_batch, partition_per_batch,canvas_idxes_this_batch,single_page_mfdetrec_res_this_batch = collect_paragraph_image_and_its_coordinate(detimages, rough_layout_this_batch,2)
    with timer('text_detection/stack'):
        canvas_tensor_this_batch = torch.stack(canvas_tensor_this_batch)
    with timer('text_detection/det_net'):
        with torch.no_grad():
            dt_boxaes_batch = det_model(canvas_tensor_this_batch)
            #dt_boxaes_batch = ocrmodel.batch_det_model.net(canvas_tensor_this_batch)
    torch.cuda.synchronize()
    dt_boxaes_batch = dt_boxaes_batch['maps'][:,0].cpu()
    return dt_boxaes_batch


async def cpu_postprocess(queue, ocrmodel):
    while True:
        dt_boxaes_batch = await queue.get()
        if dt_boxaes_batch is None:
            break
        dt_boxaes_batch=dt_boxaes_batch.numpy()
        post_result = ocrmodel.batch_det_model.fast_postprocess(dt_boxaes_batch, np.array([(1920,1472, 0.5, 0.5)]*len(dt_boxaes_batch)) )
        dt_boxes_list = [ocrmodel.batch_det_model.filter_tag_det_res(dt_boxes['points'][0], (1920,1472)) for dt_boxes in post_result]
        queue.task_done()

async def process_image(queue, layout_pair, mdf_pair, det_pair, size_pair, location_pair, timer):
    dt_boxaes_batch = await gpu_inference(layout_pair, mdf_pair, det_pair, size_pair, location_pair, timer)
    await queue.put(dt_boxaes_batch)

if __name__ == "__main__":

    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)

    img_size  = model_configs['model_args']['img_size']
    conf_thres= model_configs['model_args']['conf_thres']
    iou_thres = model_configs['model_args']['iou_thres']
    device    = model_configs['model_args']['device']
    dpi       = model_configs['model_args']['pdf_dpi']

    layout_model = get_layout_model(model_configs,accelerated=False)
    #layout_model.compile()
    inner_batch_size = 16 if get_gpu_memory() > 60 else 2
    mfd_model    = get_batch_YOLO_model(model_configs, inner_batch_size) 
    ocrmodel = None
    ocrmodel = ocr_model = ModifiedPaddleOCR(show_log=True)
    timer = Timers(False,warmup=5)
    
    asyncio.run(deal_with_one_dataset("debug.jsonl", 
                          "debug.stage_1.jsonl", 
                          layout_model, mfd_model, ocrmodel=ocrmodel, 
                          inner_batch_size=2, batch_size=4,num_workers=4,
                          timer=timer))
    
    
   