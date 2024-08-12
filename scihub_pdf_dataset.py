
from get_data_utils import *
from torch.utils.data import IterableDataset,get_worker_info,DataLoader
from utils import Timers
import torch

class PDFImageDataset(IterableDataset,DatasetUtils):
    #client = build_client()
    def __init__(self, metadata_filepath, aug, input_format, 
                 mfd_pre_transform, det_pre_transform=None,
                 return_original_image=False,timer=Timers(False),
                 partion_num = 1,
                 partion_idx = 0):
        super().__init__()
        self.metadata= self.smart_read_json(metadata_filepath)
        self.metadata= np.array_split(self.metadata, partion_num)[partion_idx]
        self.dpi = 200
        self.aug = aug
        self.input_format = input_format
        self.mfd_pre_transform = mfd_pre_transform
        self.return_original_image = return_original_image
        self.det_pre_transform = det_pre_transform
        self.timer = timer
    
    def get_pdf_by_index(self,index):
        pdf_path  = self.metadata[index]['path']
        return self.get_pdf_buffer(pdf_path)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.current_pdf_index = 0
            self.current_page_index = 0
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.current_pdf_index = worker_id
            self.current_page_index = 0

        self.pdf = self.get_pdf_by_index(self.current_pdf_index)
        return self
    
    @property
    def current_doc(self):
        if len(self.last_read_pdf_buffer)==0:return None
        return list(self.last_read_pdf_buffer.values())[0]

    def prepare_for_text_det(self,image):
        return self.text_det_transform(image)[0]

    def read_data_based_on_current_state(self):
        #print(f"read image from current_page_index={self.current_page_index} ")

        with self.timer("load_page"):
            page  = self.get_pdf_by_index(self.current_pdf_index).load_page(self.current_page_index)
        with self.timer("from_page_to_pimage"):
            oimage = process_pdf_page_to_image(page, self.dpi)
        original_image = oimage[:, :, ::-1] if self.input_format == "RGB" else oimage
        height, width = original_image.shape[:2]
        with self.timer("get_layout_image"):
            layout_image = self.aug.get_transform(original_image).apply_image(original_image)
            layout_image = torch.as_tensor(layout_image.astype("float32").transpose(2, 0, 1))[:,:1042,:800] ## it will be 1043x800 --> 1042:800
        ## lets make sure the image has correct size 
        # if layout_image.size(1) < 1042:
        #     layout_image = torch.nn.functional.pad(layout_image, (0, 0, 0, 1042-layout_image.size(1)))
        with self.timer("get_mfd_image"):
            mfd_image=self.prepare_for_mfd_model(oimage)
        with self.timer("get_det_image"):
            det_images = torch.from_numpy(self.det_pre_transform(original_image)[0])
        
        output= {"pdf_index":self.current_pdf_index, "page_index":self.current_page_index, "mfd_image":mfd_image, "layout_image":layout_image, "det_images":det_images, "height":height, "width":width}
        if self.return_original_image:
            output['oimage'] = original_image
        return output
    
    def check_should_skip(self):
        if self.current_pdf_index >= len(self.metadata):
            raise StopIteration
        current_pdf_page_num = self.get_pdf_by_index(self.current_pdf_index).page_count
        #print(f"current_page_index={self.current_page_index} current_pdf_page_num={current_pdf_page_num}|{len(self.get_pdf_by_index(self.current_pdf_index))}")

        if self.current_page_index >= current_pdf_page_num:
            worker_info = get_worker_info()
            step_for_pdf= 1 if worker_info is None else worker_info.num_workers
            self.current_pdf_index += step_for_pdf
            self.current_page_index = 0
            if self.current_pdf_index >= len(self.metadata):
                raise StopIteration



    def __next__(self):
        
        fail_times = 0
        output = None
        while output is None and fail_times<=10:
            self.check_should_skip()
            try:
                output = self.read_data_based_on_current_state()
                self.current_page_index += 1
                fail_times = 0
            except StopIteration:
                raise StopIteration
            except:
                fail_times +=1
        if fail_times>10 or output is None:
            raise StopIteration
        return output
        
    

       

    def prepare_for_mfd_model(self, im:np.ndarray):
        if self.mfd_pre_transform is None :return im
        assert im.ndim==3
        im = [im]
        im = np.stack(self.mfd_pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype('float')/255
        im = torch.from_numpy(im)
        return im[0]


def custom_collate_fn(batches):
    
    return_batch = {}
    for batch in batches:
        for key,val in batch.items():
            if key not in return_batch:
                return_batch[key] = []
            return_batch[key].append(val)
    
    keys = list(return_batch.keys())
    for key in keys:
        if key in ["pdf_index", "page_index","height", "width"]:
            return_batch[key] = torch.tensor(return_batch[key])
        elif key in ["mfd_image", "layout_image", "det_images"]:
            return_batch[key] = torch.stack(return_batch[key])
        elif key in ['oimage']:
            return_batch[key] = return_batch[key]
    return return_batch
    