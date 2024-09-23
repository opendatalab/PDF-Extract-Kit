
import json
import requests
import io
import os
import fitz 
fitz.TOOLS.mupdf_display_errors(on=False)


def clean_pdf_path(pdf_path):
    return pdf_path[len("opendata:"):] if pdf_path.startswith("opendata:") else pdf_path

def read_json_from_path(path, client):
    if "s3:" in path:
        buffer = client.get(path).replace(b'\x00', b'\n')
        if path.endswith('.json'):
            return json.loads(buffer)
        elif path.endswith('.jsonl'):
            whole_data = []
            for t in io.BytesIO(buffer).readlines():
                try:
                    data = json.loads(t)
                except:
                    print(t)
                    raise
                whole_data.append(data)
            return whole_data
        else:
            return {'content':str(buffer)}
    elif path.startswith('http'):
        response = requests.get(path)
        if response.status_code == 200:
            content = response.json()["content"]
            if path.endswith('.json'):
                content = json.loads(content)
            elif path.endswith('.md'):
                content = {'content':content}
            return content
        else:
            return None
    else:
        if path.endswith('.json'):
            with open(path,'r') as f:
                data = json.load(f)
                return data
        elif path.endswith('.jsonl'):
            with open(path,'r') as f:
                whole_data = []
                for t in f.readlines():
                    data = json.loads(t.strip())
                    whole_data.append(data)
                return whole_data
        else:
            raise NotImplementedError("please use json or jsonl file")



def write_jsonj_to_path(data, path, client):
    if "s3:" in path:
        byte_object = json.dumps(data).encode('utf-8')
        with io.BytesIO(byte_object) as f:
            client.put(path, f)
    else:
        assert not path.startswith('http'), "why you want to save the file to a online path?"
        thedir = os.path.dirname(path)
        os.makedirs(thedir, exist_ok=True)
        with open(path,'w') as f:
            json.dump(data, f)

def write_json_to_path(data, path, client):
    if "s3:" in path:
        byte_object = json.dumps(data).encode('utf-8')
        with io.BytesIO(byte_object) as f:
            client.put(path, f)
    else:
        assert not path.startswith('http'), "why you want to save the file to a online path?"
        thedir = os.path.dirname(path)
        os.makedirs(thedir, exist_ok=True)
        with open(path,'w') as f:
            json.dump(data, f)
def write_jsonl_to_path(data, path, client):
    byte_object = "\n".join([json.dumps(d) for d in data])
    if "s3:" in path:
        with io.BytesIO(byte_object.encode('utf-8')) as f:
            client.put(path, f)
    else:
        assert not path.startswith('http'), "why you want to save the file to a online path?"
        thedir = os.path.dirname(path)
        if thedir:
            os.makedirs(thedir, exist_ok=True)
        with open(path,'w') as f:
            for d in data:
                try:
                    byte_object = json.dumps(d)
                except:

                    raise NotImplementedError(f"fail to dump {d}")
                f.write(byte_object+'\n')


import boto3
from botocore.client import Config
class MyFastS2client:
    def __init__(self, ACCESS_KEY, SECRET_KEY,ENDPOINT):

        session = boto3.Session(
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=''
        )

        # Create an S3 client
        self.s3 = session.client(
            's3',
            endpoint_url=ENDPOINT,
            config=Config(signature_version='s3v4')  # Ensure compatibility
        )

    def get(self, path):
        # path is like opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub/part-66210c190659-000026.jsonl
        # obtain bucket_name and object_key
        bucket_name = path.split("//")[1].split("/")[0]
        object_key  = "/".join(path.split("//")[1].split("/")[1:])
        response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        return response['Body'].read()

    def put(self, path,data):
        # path is like opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub/part-66210c190659-000026.jsonl
        # obtain bucket_name and object_key
        bucket_name = path.split("//")[1].split("/")[0]
        object_key  = "/".join(path.split("//")[1].split("/")[1:])
    
        self.s3.put_object(Bucket=bucket_name, Key=object_key, Body=data)

    def contains(self, path):
        # path is like opendata:s3://llm-process-pperf/ebook_index_v4/scihub/v001/scihub/part-66210c190659-000026.jsonl
        # obtain bucket_name and object_key
        bucket_name = path.split("//")[1].split("/")[0]
        object_key  = "/".join(path.split("//")[1].split("/")[1:])
        try:
            self.s3.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except:
            return False
def build_client():
    #print(f"we will building ceph client...................")
    
    try:
        from petrel_client.client import Client  # 安装完成后才可导入
        client = Client(conf_path="~/petreloss.conf") # 实例化Petrel Client，然后就可以调用下面的APIs   
    except:
        
        ### get key and endpoint from local .client.conf
        with open(".client.conf",'r') as f:
            lines = f.readlines()
            for line in lines:
                if "key" in line:
                    ACCESS_KEY = line.split("=")[1].strip()
                if "secret" in line:
                    SECRET_KEY = line.split("=")[1].strip()
                if "endpoint" in line:
                    ENDPOINT = line.split("=")[1].strip()
        client = MyFastS2client(ACCESS_KEY, SECRET_KEY, ENDPOINT) # 实例化Petrel Client，然后就可以调用下面的APIs   
    

    #print(f"done..................")
    return client

def check_path_exists(path,client):
    print(path)
    if "s3:" in path:
        return client.contains(path)
    elif path.startswith('http'):
        assert 'get_data' in path, "please use get_data flag for data path"
        response = requests.get(path.replace('get_data','checkfile'))
        if response.status_code == 200:
            status = response.json()["status"]
            return status
        else:
            return False
    else:
        return os.path.exists(path)

def check_lock_exists(path, client):
    if "s3:" in path:
        raise NotImplementedError("s3 lock not implemented")
    elif path.startswith('http'):
        assert 'get_data' in path, "please use get_data flag for data path"
        response = requests.get(path.replace('get_data','checklock'))
        if response.status_code == 200:
            status = response.json()["status"]
            return status
        else:
            return False
    else:
        raise NotImplementedError("please donot use lock lock")
        return os.path.exists(path)

def check_lock_and_last_start_time(path, client):
    if "s3:" in path:
        raise NotImplementedError(f"s3 lock not implemented. Now path {path}")
    elif path.startswith('http'):
        assert 'checklocktime' in path, "please use `checklocktime` flag for data path"
        response = requests.get(path)
        if response.status_code == 200:
            content = response.json()
            if not content["status"]:return False
            return content['start_time']
        else:
            return False
    else:
        raise NotImplementedError("s3 lock not implemented")

def create_last_start_time_lock(path, client):
    if "s3:" in path:
        raise NotImplementedError("s3 lock not implemented")
    elif path.startswith('http'):
        assert 'createlocktime' in path, "please use `createlocktime` flag for data path"
        response = requests.get(path)
    else:
        raise NotImplementedError("s3 lock not implemented")

from PIL import Image
import numpy as np
UNIFIED_WIDTH  = 1472  # lets always make the oimage in such size
UNIFIED_HEIGHT = 1920  # lets always make the oimage in such size
def pad_image_to_ratio(image, output_width = UNIFIED_WIDTH,output_height=UNIFIED_HEIGHT, ):
    """
    Pads the given PIL.Image object to fit the specified width-height ratio
    by adding padding only to the bottom and right sides.

    :param image: PIL.Image object
    :param target_ratio: Desired width/height ratio (e.g., 16/9)
    :return: New PIL.Image object with the padding applied
    """
    # Original dimensions
    input_width, input_height = image.size
    height = min(input_height, output_height)
    width  = min(input_width,   output_width)

    if output_height == input_height and output_width == input_width:
        return image

    if input_height / output_height > input_width / output_width:
        # Resize to match height, width will be smaller than output_width
        height = output_height
        width = int(input_width * output_height / input_height)
    else:
        # Resize to match width, height will be smaller than output_height
        width = output_width
        height = int(input_height * output_width / input_width)
    image= image.resize((width, height), resample=3)
    # Create new image with target dimensions and a white background
    new_image = Image.new("RGB", (output_width, output_height), (255, 255, 255))
    new_image.paste(image, (0, 0))

    return new_image

def process_pdf_page_to_image(page, dpi,output_width=UNIFIED_WIDTH,output_height=UNIFIED_HEIGHT):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    if pix.width > 3000 or pix.height > 3000:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    else:
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

    image = pad_image_to_ratio(image, output_width = output_width,output_height=output_height)
    
    image = np.array(image)[:,:,::-1]
    return image.copy()

def read_pdf_from_path(path, client):
    if "s3:" in path:
        buffer = client.get(path)
        return fitz.open(stream = buffer, filetype="pdf")
    else:
        return fitz.open(path)

import pymupdf
class DatasetUtils:
    client = None
    last_read_pdf_buffer={}
    def smart_read_json(self, json_path):
        if "s3:" in json_path and self.client is None: self.client = build_client()
        if json_path.startswith("s3:"): json_path = "opendata:"+ json_path
        return read_json_from_path(json_path, self.client)
    
    def smart_write_json(self, data, targetpath):
        if "s3:" in targetpath and self.client is None: self.client = build_client()
        if json_path.startswith("s3:"): json_path = "opendata:"+ json_path
        write_json_to_path(data, targetpath, self.client)
    
    def check_path_exists(self, path):
        if "s3:" in path and self.client is None: self.client = build_client()
        if path.startswith("s3:"): path = "opendata:"+ path
        return check_path_exists(path, self.client)

    def smart_load_pdf(self, pdf_path):
        if "s3:" in pdf_path and self.client is None: self.client = build_client()
        if pdf_path.startswith("s3:"): pdf_path = "opendata:"+ pdf_path
        with self.timer("smart_load_pdf"):
            try:
                pdf_buffer = read_pdf_from_path(pdf_path, self.client)
            except pymupdf.mupdf.FzErrorFormat:
                print(f"""
                      ========================================
                      error in loading pdf {pdf_path}, we pass
                      ========================================
                      """)
                pdf_buffer = None
            except Exception as e:
                print(f"error in loading pdf {pdf_path}")
                print(e)
                raise
        return pdf_buffer
    
    def clean_pdf_buffer(self):
        return 
        keys = list(self.last_read_pdf_buffer.keys())
        for key in keys:
            if self.last_read_pdf_buffer[key] is not None:
                self.last_read_pdf_buffer[key].close()
            del self.last_read_pdf_buffer[key]


    def get_pdf_buffer(self,path, buffer_num=1):
        if "s3:" in path and self.client is None: self.client = build_client()
        if path.startswith("s3:"): path = "opendata:"+ path
        if path not in self.last_read_pdf_buffer:
            if buffer_num is not None and len(self.last_read_pdf_buffer) >= buffer_num:
                self.clean_pdf_buffer()
            self.last_read_pdf_buffer[path] = self.smart_load_pdf(path)
            
        pdf_buffer = self.last_read_pdf_buffer[path]
        return pdf_buffer
    

from tqdm.auto import tqdm
import json,os
from multiprocessing import Pool
FILEROOT = "page_num_map"
def process_file(filename):
    with open(os.path.join(FILEROOT, filename)) as f:
        data = json.load(f)
    return data
def get_page_num_map_whole():

    page_num_map_whole = {}
    files = os.listdir(FILEROOT)
    num_thread=4
    print("to get page num map whole")
    with Pool(num_thread) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))

    for result in results:
        page_num_map_whole.update(result)
    return page_num_map_whole

output_width =1472 #pdf_metadata['width']#1472
output_height=1920 #pdf_metadata['height']#1920
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from batch_running_task.utils import convert_boxes
def build_dict(pdf_metadata_list, track_id_key = "track_id"):
    pdf_metadata_dict = {}
    for pdf_metadata in pdf_metadata_list:
        track_id = pdf_metadata[track_id_key]
        height   = pdf_metadata.get('height', 1920)
        width    = pdf_metadata.get('width',1472)
        if height == output_height and width == output_width:
            pass
        else:
            ### lets do the bbox convertion
            doc_layout_result=pdf_metadata['doc_layout_result']
            for pdf_page_metadata in doc_layout_result:
                page_id = pdf_page_metadata['page_id']
                layout_dets = []
                for res in pdf_page_metadata["layout_dets"]:
                    new_res = res.copy()
                    xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                    xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                    bbox= [xmin, ymin, xmax, ymax]
                    bbox= convert_boxes([bbox], pdf_metadata['width'], pdf_metadata['height'], output_width, output_height)[0]
                    poly= [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]
                    res['poly'] = poly
        page_id_to_metadata = {pdf_page_metadata['page_id']: pdf_page_metadata for pdf_page_metadata in pdf_metadata['doc_layout_result']}
        pdf_metadata_dict[track_id] = page_id_to_metadata
            
    return pdf_metadata_dict

def read_data_with_patch(result_path, client):
    if result_path.startswith("s3:"):
        result_path = "opendata:"+result_path
    pdf_path_map_to_page_num = []
    #assert "layoutV" in result_path
    filename   = os.path.basename(result_path)
    patch_path = os.path.join(os.path.dirname(os.path.dirname(result_path)),"det_patch_good",filename)
    missingpath= os.path.join(os.path.dirname(os.path.dirname(result_path)),"fix_missing_page_version2",filename)
    # mfr_patchpath  = os.path.join(os.path.dirname(os.path.dirname(result_path)),"mfr_patch",filename)
    # mfr_patch_bf16path = os.path.join(os.path.dirname(os.path.dirname(result_path)),"mfr_patch_bf16",filename)
    # rec_patchpath  = os.path.join(os.path.dirname(os.path.dirname(result_path)),"rec_patch",filename)

    assert check_path_exists(result_path,client)
    #tqdm.write("reading result")
    result  = read_json_from_path(result_path,client)
    result_dict      = build_dict(result)
    
    patch_add_dict   = build_dict(read_json_from_path(patch_path,client)) if check_path_exists(patch_path,client) else {}
    
    missing_dict     = build_dict(read_json_from_path(missingpath,client)) if check_path_exists(missingpath,client) else {}
    # mfr_patch_dict     = build_dict(read_json_from_path(mfr_patchpath,client)) if check_path_exists(mfr_patchpath,client) else {}
    # mfr_patch_bf16_dict     = build_dict(read_json_from_path(mfr_patch_bf16path,client)) if check_path_exists(mfr_patch_bf16path,client) else {}
    # rec_patch_dict     = build_dict(read_json_from_path(rec_patchpath,client)) if check_path_exists(rec_patchpath,client) else {}
    
    #tqdm.write("reading done")
    if len(patch_add_dict) == 0 and len(missing_dict) == 0:
        #tqdm.write(f"no patch and missing for {result_path}")
        pass
    else:

        for track_id, pdf_metadata in result_dict.items():
            for patch_dict in [patch_add_dict, missing_dict]:
                if track_id in patch_dict:
                    patch_pdf_metadata = patch_dict[track_id]
                    for page_id, pdf_page_metadata in patch_pdf_metadata.items():
                        if page_id in pdf_metadata:
                            ## then merge page result
                            pdf_metadata[page_id]["layout_dets"].extend(pdf_page_metadata["layout_dets"])
                        else:
                            pdf_metadata[page_id] = pdf_page_metadata  
        for pdf_metadata in result:
            track_id = pdf_metadata['track_id']
            pdf_metadata['height'] = output_height
            pdf_metadata['width'] = output_width
            doc_layout_result = []
            for page_id, pdf_page_metadata in result_dict[track_id].items():
                doc_layout_result.append(pdf_page_metadata)
            pdf_metadata['doc_layout_result'] = doc_layout_result       
    return result

def merge_rec_result(pdf_metadata, rec_patch_dict, track_id_key = "path"):
    track_id = pdf_metadata[track_id_key]
    if track_id in rec_patch_dict:
        current_rec_patch = rec_patch_dict[track_id]
    else:
        return 
    for pdf_page_metadata in pdf_metadata['doc_layout_result']:
        page_id = pdf_page_metadata['page_id']
        bbox_count = 0
        for bbox_metadata in pdf_page_metadata['layout_dets']:
            if bbox_metadata['category_id'] != 15:continue
            bbox_count+=1
        if bbox_count == 0: continue
        patch_rec_list = current_rec_patch[page_id]["layout_dets"]
        assert len(patch_rec_list) == bbox_count, f"pdf={track_id} page={page_id} => bbox count {bbox_count} not equal to patch count {len(patch_rec_list)}"
        bbox_id = 0
        for bbox_metadata in pdf_page_metadata['layout_dets']:
            if bbox_metadata['category_id'] != 15:continue
            bbox_metadata.update(patch_rec_list[bbox_id])
            bbox_id += 1

def merge_mfr_result(pdf_metadata, mfr_patch_dict, track_id_key = "path"):
    track_id = pdf_metadata[track_id_key]
    if track_id in mfr_patch_dict:
        current_mfr_patch = mfr_patch_dict[track_id]
    else:
        return 
    for pdf_page_metadata in pdf_metadata['doc_layout_result']:
        page_id = pdf_page_metadata['page_id']
        bbox_count = 0
        for bbox_metadata in pdf_page_metadata['layout_dets']:
            if bbox_metadata['category_id'] not in [13, 14]:continue
            bbox_count+=1
        if bbox_count == 0: continue
        patch_mfr_list = current_mfr_patch[page_id]["layout_dets"]
        assert len(patch_mfr_list) == bbox_count, f"pdf={track_id} page={page_id} => bbox count {bbox_count} not equal to patch count {len(patch_mfr_list)}"
        bbox_id = 0
        for bbox_metadata in pdf_page_metadata['layout_dets']:
            if bbox_metadata['category_id'] not in [13, 14]:continue
            bbox_metadata.update(patch_mfr_list[bbox_id])
            bbox_id += 1
        
def read_data_with_mfr(result_path, client):
    if result_path.startswith("s3:"):
        result_path = "opendata:"+result_path

    filename   = os.path.basename(result_path)

    mfr_patchpath  = os.path.join(os.path.dirname(os.path.dirname(result_path)),"mfr_patch",filename)
    mfr_patch_bf16path = os.path.join(os.path.dirname(os.path.dirname(result_path)),"mfr_patch_bf16",filename)
    rec_patchpath  = os.path.join(os.path.dirname(os.path.dirname(result_path)),"rec_patch",filename)

    assert check_path_exists(result_path,client)
    #tqdm.write("reading result")
    result      = read_json_from_path(result_path,client)
    
    track_id_key    = 'path'
    mfr_patch_dict     = build_dict(read_json_from_path(mfr_patchpath,client),track_id_key = track_id_key)      if check_path_exists(mfr_patchpath,client) else {}
    mfr_patch_bf16_dict= build_dict(read_json_from_path(mfr_patch_bf16path,client),track_id_key = track_id_key) if check_path_exists(mfr_patch_bf16path,client) else {}
    mfr_patch_dict.update(mfr_patch_bf16_dict)
    if len(mfr_patch_dict)>0:
        for pdf_metadata in tqdm(result, desc="adding patch and missing", leave=False, position=3):
            merge_mfr_result(pdf_metadata, mfr_patch_dict)

    track_id_key    = 'path'
    rec_patch_dict  = build_dict(read_json_from_path(rec_patchpath,client),track_id_key = track_id_key) if check_path_exists(rec_patchpath,client) else {}
    if len(rec_patch_dict)>0:
        for pdf_metadata in tqdm(result, desc="[REC] adding patch and missing", leave=False, position=3):
            merge_rec_result(pdf_metadata, rec_patch_dict, track_id_key=track_id_key)
    
    return result


def read_data_with_version(result_path, client):
    if result_path.startswith("s3:"):
        result_path = "opendata:"+result_path
    #assert "layoutV" in result_path
    filename = os.path.basename(result_path)
    rootpath = os.path.dirname(os.path.dirname(result_path))
    version1 = os.path.join(rootpath,"add_mfr",filename)
    version2 = os.path.join(rootpath,"rec_fixed_final",filename)
    version3 = os.path.join(rootpath,"fix_missing_page_version2",filename)

    assert check_path_exists(result_path,client)
    #tqdm.write("reading result")
    result  = read_json_from_path(result_path,client)
    result_dict      = build_dict(result)
    patch_version1_dict   = build_dict(read_json_from_path(version1,client)) if check_path_exists(version1,client) else {}
    patch_version2_dict   = build_dict(read_json_from_path(version2,client)) if check_path_exists(version2,client) else {}
    patch_version3_dict   = build_dict(read_json_from_path(version3,client)) if check_path_exists(version3,client) else {}
    
    
    #tqdm.write("reading done")
    for track_id, pdf_metadata in result_dict.items():
        for patch_dict in [patch_version1_dict, patch_version2_dict, patch_version3_dict]:
            if track_id in patch_dict:
                patch_pdf_metadata = patch_dict[track_id]
                for page_id, pdf_page_metadata in patch_pdf_metadata.items():
                    if page_id in pdf_metadata:
                        assert len(pdf_page_metadata["layout_dets"]) == len(pdf_metadata[page_id]["layout_dets"]), f"pdf={track_id} page={page_id} => bbox count {len(pdf_metadata[page_id]['layout_dets'])} not equal to patch count {len(pdf_page_metadata['layout_dets'])}"
                        for box1_dict, box2_dict in zip(pdf_metadata[page_id]["layout_dets"], pdf_page_metadata["layout_dets"]):
                            assert box1_dict['category_id'] == box2_dict['category_id'], f"pdf={track_id} page={page_id} => category_id {box1_dict['category_id']} not equal to patch category_id {box2_dict['category_id']}"
                            assert box1_dict['poly'] == box2_dict['poly'], f"pdf={track_id} page={page_id} => poly {box1_dict['poly']} not equal to patch poly {box2_dict['poly']}"
                            if box1_dict['category_id'] == 15:
                                if box2_dict.get('text',"") == "":continue
                                if box1_dict.get('text',"") == "":
                                    box1_dict['text'] = box2_dict.get('text',"")
                                
                                else:
                                    assert box1_dict['text'] == box2_dict['text'], f"pdf={track_id} page={page_id} => text {box1_dict['text']} not equal to patch text {box2_dict['text']}"
                            
                            if box1_dict['category_id'] in {13, 14}:
                                if box2_dict.get('latex',"") == "":continue
                                if box1_dict.get('latex',"") == "":
                                    box1_dict['latex'] = box2_dict['latex']
                                else:
                                    assert box1_dict['latex'] == box2_dict['latex'], f"pdf={track_id} page={page_id} => latex {box1_dict['latex']} not equal to patch latex {box2_dict['latex']}" 
                            box1_dict.update(box2_dict)
                    else:
                        pdf_metadata[page_id] = pdf_page_metadata  
    
    for pdf_metadata in result:
        track_id = pdf_metadata['track_id']
        pdf_metadata['height'] = output_height
        pdf_metadata['width'] = output_width
        doc_layout_result = []
        for page_id, pdf_page_metadata in result_dict[track_id].items():
            doc_layout_result.append(pdf_page_metadata)
        pdf_metadata['doc_layout_result'] = doc_layout_result    

    print(len(result))
    # mfr_patch_dict     = build_dict(read_json_from_path(mfr_patchpath,client)) if check_path_exists(mfr_patchpath,client) else {}
    # mfr_patch_bf16_dict     = build_dict(read_json_from_path(mfr_patch_bf16path,client)) if check_path_exists(mfr_patch_bf16path,client) else {}
    # rec_patch_dict     = build_dict(read_json_from_path(rec_patchpath,client)) if check_path_exists(rec_patchpath,client) else {}
       
    return result
