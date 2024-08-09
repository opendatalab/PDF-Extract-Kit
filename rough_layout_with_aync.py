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
    data_to_save = {}
    postprocess_task = asyncio.create_task(cpu_postprocess(queue, ocrmodel,data_to_save))

    featcher   = DataPrefetcher(dataloader,device='cuda')
    
    inner_batch_size = inner_batch_size
    pbar  = None#tqdm(total=len(dataset.metadata),position=2,desc="PDF Pages",leave=True)
    pdf_passed = set()
    batch = featcher.next()
    data_loading = []
    model_train  = []
    last_record_time = time.time()
    while batch is not None:

        data_loading.append(time.time() - last_record_time);last_record_time =time.time() 
        if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
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
            size_pair     = (heights, widths,inner_batch_size)
            pdf_paths     = [dataset.metadata[pdf_index]['path'] for pdf_index in pdf_index]
            location_pair = (pdf_paths, page_ids)
            await gpu_inference(queue, layout_pair, mdf_pair, det_pair, size_pair, location_pair, data_to_save, timer)

        update_seq = len(new_pdf_processed)
        if pbar:pbar.update(update_seq)

        timer.log()
        model_train.append(time.time() - last_record_time);last_record_time =time.time()
        if pbar:pbar.set_description(f"[Data][{np.mean(data_loading[-10:]):.2f}] [Model][{np.mean(model_train[-10:]):.2f}]")
        batch = featcher.next()
        if pbar is None:
            pbar = tqdm(total=len(dataset.metadata)-update_seq,position=2,desc="PDF Pages",leave=True)
    
    await queue.join()
    await queue.put(None)  # Signal the consumer to exit
    await postprocess_task  # Wait for the consumer to finish

    tqdm.write("we finish generate data, lets collect and save it")
    ### next, we construct each result for each pdf in pdf wise and remove the page_id by the list position 
    pdf_to_metadata = {t['path']:t for t in dataset.metadata}
    new_data_to_save = []
    for pdf_path, layout_dets_per_page in data_to_save.items():
        new_pdf_dict = copy.deepcopy(pdf_to_metadata[pdf_path])
        new_pdf_dict['height'] = layout_dets_per_page.pop('height')
        new_pdf_dict['width'] = layout_dets_per_page.pop('width')
        pages = [t for t in layout_dets_per_page.keys()]
        pages.sort()
        new_pdf_dict["doc_layout_result"]=[]
        for page_id in range(max(pages)):
            if page_id not in layout_dets_per_page:
                print(f"WARNING: page {page_id} of PDF: {pdf_path} fail to parser!!! ")
                now_row = {"page_id": page_id, "status": "fail", "layout_dets":[]}
            else:
                now_row = {"page_id": page_id, "layout_dets":layout_dets_per_page[page_id]}
            new_pdf_dict["doc_layout_result"].append(now_row)
        new_data_to_save.append(new_pdf_dict)
    if "s3:" in new_data_to_save and dataset.client is None:dataset.client=build_client()
    write_jsonl_to_path(new_data_to_save, result_path, dataset.client)

async def gpu_inference(queue,
              layout_pair,
              mdf_pair,
              det_pair,
              size_pair,
              location_pair,
              data_to_save,
              timer):
    layout_images, layout_model = layout_pair
    mfd_images, mfd_model = mdf_pair
    detimages, det_model = det_pair
    heights, widths,inner_batch_size = size_pair
    pdf_paths, page_ids= location_pair
    with timer('get_layout'):
        layout_res = inference_layout((layout_images,heights, widths),layout_model,inner_batch_size)
    with timer('get_mfd'):
        mfd_res    = inference_mfd(mfd_images,mfd_model,inner_batch_size)
    with timer('combine_layout_mfd_result'):
        rough_layout_this_batch, ori_shape_list = combine_layout_mfd_result(layout_res, mfd_res, heights, widths)

    pdf_and_page_id_this_batch=[]
    for pdf_path, page_id, layout_dets,ori_shape in zip(pdf_paths, page_ids, rough_layout_this_batch,ori_shape_list):
        page_id = int(page_id)
        if pdf_path not in data_to_save:
            data_to_save[pdf_path] = {'height':ori_shape[0], 'width':ori_shape[1]}
        data_to_save[pdf_path][page_id] = layout_dets
        pdf_and_page_id_this_batch.append((pdf_path, page_id))

    
    with timer('text_detection/collect_for_line_detect'):
        det_height, det_width = detimages.shape[2:]
        scale_height = int(heights[0])/int(det_height)
        scale_width  = int(widths[0])/int(det_width)
        assert scale_height == scale_width
        assert scale_height == 2
        canvas_tensor_this_batch, partition_per_batch,_,_ = collect_paragraph_image_and_its_coordinate(detimages, rough_layout_this_batch,scale_height) # 2 is the scale between detiamge and box_images
    with timer('text_detection/stack'):
        canvas_tensor_this_batch = torch.stack(canvas_tensor_this_batch)
    with timer('text_detection/det_net'):
        dt_boxaes_batch = inference_det(canvas_tensor_this_batch,det_model,128)
    
    
    torch.cuda.synchronize()
    result = (dt_boxaes_batch,partition_per_batch,pdf_and_page_id_this_batch)
    await queue.put(result)

async def cpu_postprocess(queue, ocrmodel,data_to_save):
    while True:
        state = await queue.get()
        if state is None:
            queue.task_done()
            break
        dt_boxaes_batch,partition_per_batch,pdf_and_page_id_this_batch = state
        dt_boxes_list = det_postprocess(dt_boxaes_batch,ocrmodel)
        for partition_id in range(len(partition_per_batch)-1):
            pdf_path, page_id = pdf_and_page_id_this_batch[partition_id]
            partition_start = partition_per_batch[partition_id]
            partition_end   = partition_per_batch[partition_id+1]
            dt_boxes_this_partition = dt_boxes_list[partition_start:partition_end]
            for dt_boxes in dt_boxes_this_partition: #(1, 4, 2)
                for line_box in dt_boxes:
                    p1, p2, p3, p4 = line_box.tolist()
                    data_to_save[pdf_path][page_id].append(
                        {
                            'category_id': 15,
                            'poly': p1 + p2 + p3 + p4,
                        }
                    )
        queue.task_done()
        

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
    total_memory = get_gpu_memory()
    inner_batch_size = 16 if total_memory > 60 else 2
    print(f"totally gpu memory is {total_memory} we use inner batch size {inner_batch_size}")
    mfd_model    = get_batch_YOLO_model(model_configs, inner_batch_size) 
    ocrmodel = None
    ocrmodel = ocr_model = ModifiedPaddleOCR(show_log=True)
    timer = Timers(False,warmup=5)
    
    asyncio.run(deal_with_one_dataset("debug.jsonl", 
                          "debug.stage_1.jsonl", 
                          layout_model, mfd_model, ocrmodel=ocrmodel, 
                          inner_batch_size=inner_batch_size, batch_size=16,num_workers=4,
                          timer=timer))
    
    
   