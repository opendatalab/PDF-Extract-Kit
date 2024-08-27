
import json
import requests
import io
import os
import fitz 
fitz.TOOLS.mupdf_display_errors(on=False)
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
                    data = json.loads(t)
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
            f.write(byte_object)


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

def process_pdf_page_to_image(page, dpi):
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    if pix.width > 3000 or pix.height > 3000:
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    else:
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

    image = pad_image_to_ratio(image, output_width = UNIFIED_WIDTH,output_height=UNIFIED_HEIGHT)
    
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