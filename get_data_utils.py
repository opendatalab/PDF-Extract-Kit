
import json
import requests
import io
import os
import fitz 
def read_json_from_path(path, client):
    if "s3" in path:
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

def write_json_to_path(data, path, client):
    if "s3" in path:
        byte_object = json.dumps(data).encode('utf-8')
        with io.BytesIO(byte_object) as f:
            client.put(path, f)
    else:
        assert not path.startswith('http'), "why you want to save the file to a online path?"
        thedir = os.path.dirname(path)
        os.makedirs(thedir, exist_ok=True)
        with open(path,'w') as f:
            json.dump(data, f)

def build_client():
    print(f"we will building ceph client...................")
    from petrel_client.client import Client  # 安装完成后才可导入
    client = Client(conf_path="~/petreloss.conf") # 实例化Petrel Client，然后就可以调用下面的APIs   
    print(f"done..................")
    return client

def check_path_exists(path,client):
    if "s3" in path:
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
    if "s3" in path:
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

def read_pdf_from_path(path, client):
    if "s3" in path:
        buffer = client.get(path)
        return fitz.open(stream = buffer, filetype="pdf")
    else:
        return fitz.open(path)