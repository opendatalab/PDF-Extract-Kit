import re
import boto3
import os
import json
import uuid
import random

from typing import Iterator, Union, Dict, List, Tuple

from botocore.client import Config
from botocore.response import StreamingBody
from boto3.s3.transfer import TransferConfig

from tenacity import retry, wait_random, stop_after_attempt, wait_fixed

###############################
#### Global Configurations ####
###############################

__re_s3_path = re.compile("^s3a?://([^/]+)(?:/(.*))?$")
__re_bytes = re.compile("^([0-9]+)([,-])([0-9]+)$")

__spark_configs = {
    "spark.hadoop.fs.s3a.connection.maximum": "50",  # may no enough sometime.
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.list.version": "1",
    "spark.hadoop.fs.s3a.paging.maximum": "1000",
    "spark.hadoop.fs.s3a.committer.name": "directory",
}


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("s3a://")


def ensure_s3a_path(path: str) -> str:
    if not path.startswith("s3://"):
        return path
    return "s3a://" + path[len("s3://") :]


def ensure_s3_path(path: str) -> str:
    if not path.startswith("s3a://"):
        return path
    return "s3://" + path[len("s3a://") :]


def split_s3_path(path: str):
    "split bucket and key from path"
    m = __re_s3_path.match(path)
    if m is None:
        return "", ""
    return m.group(1), (m.group(2) or "")


def __get_s3_bucket_config(path: str):
    bucket = split_s3_path(path)[0] if path else ""
    bucket_config = s3_buckets.get(bucket)
    if not bucket_config:
        bucket_config = s3_buckets.get("[default]")
        assert bucket_config is not None
    return bucket_config


def __get_s3_config(bucket_config: tuple, outside: bool, prefer_ip=False):
    cluster, user = bucket_config
    cluster_config = s3_clusters[cluster]

    if not outside and cluster_config.get("cluster") == get_cluster_name():
        endpoint_key = "inside"
    else:
        endpoint_key = "outside"

    if prefer_ip and f"{endpoint_key}_ips" in cluster_config:
        endpoint_key = f"{endpoint_key}_ips"

    endpoints = cluster_config[endpoint_key]
    endpoint = random.choice(endpoints)
    return {"endpoint": endpoint, **s3_users[user]}


def get_s3_config(path: Union[str, List[str]], outside=False):
    paths = [path] if type(path) == str else path
    bucket_config = None
    for p in paths:
        bc = __get_s3_bucket_config(p)
        if bucket_config in [bc, None]:
            bucket_config = bc
            continue
        raise Exception(f"{paths} have different s3 config, cannot read together.")
    if not bucket_config:
        raise Exception("path is empty.")
    return __get_s3_config(bucket_config, outside, prefer_ip=True)


def get_s3_spark_configs(outside=False) -> List[Tuple[str, str]]:
    ret = [(k, v) for k, v in __spark_configs.items()]
    sc_prefix = "spark.hadoop.fs.s3a"
    sc_items = {
        "ak": "access.key",
        "sk": "secret.key",
        "endpoint": "endpoint",
        "path.style.access": "path.style.access",
    }

    default_config = __get_s3_config(s3_buckets["[default]"], outside)
    for key, sc_item in sc_items.items():
        if key not in default_config:
            continue
        ret.append((f"{sc_prefix}.{sc_item}", default_config[key]))

    for bucket, bucket_config in s3_buckets.items():
        if bucket == "[default]":
            continue
        s3_config = __get_s3_config(bucket_config, outside)
        for key, sc_item in sc_items.items():
            if key not in s3_config:
                continue
            if s3_config[key] == default_config[key]:
                continue
            ret.append((f"{sc_prefix}.bucket.{bucket}.{sc_item}", s3_config[key]))

    return ret


def get_s3_client(path: Union[str, List[str]], config=None, outside=False):
    if config:
        s3_config = config
    else:
        s3_config = get_s3_config(path, outside)
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8, "mode": "standard"}),
        )
    except:
        # older boto3 do not support retries.mode param.
        return boto3.client(
            "s3",
            aws_access_key_id=s3_config["ak"],
            aws_secret_access_key=s3_config["sk"],
            endpoint_url=s3_config["endpoint"],
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}),
        )


def get_s3_object(client, path: str, **kwargs) -> dict:
    bucket, key = split_s3_path(path)
    return client.get_object(Bucket=bucket, Key=key, **kwargs)


def read_s3_object(client, path: str, bytes: Union[str, None] = None) -> StreamingBody:
    """
    ### Usage
    ```
    obj = read_object("s3://bkt/path/to/file.txt")
    for line in obj.iter_lines():
      handle(line)
    ```
    """
    kwargs = {}
    if bytes:
        m = __re_bytes.match(bytes)
        if m is not None:
            frm = int(m.group(1))
            to = int(m.group(3))
            sep = m.group(2)
            if sep == ",":
                to = frm + to - 1
            kwargs["Range"] = f"bytes={frm}-{to}"

    obj = get_s3_object(client, path, **kwargs)
    return obj["Body"]


def put_s3_object(client, path: str, body: bytes):
    bucket, key = split_s3_path(path)
    return client.put_object(Bucket=bucket, Key=key, Body=body)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def read_s3_object_content(s3_client, s3_key) -> bytes:
    with read_s3_object(s3_client, s3_key) as stream:
        buf:bytes = stream.read()
        return buf
    
@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def write_s3_object_content(s3_client, s3_key: str, content: bytes):
    put_s3_object(s3_client, s3_key, content)


def list_s3_objects(client, path: str, recursive=False, is_prefix=False, limit=0):
    for content in list_s3_objects_detailed(client, path, recursive, is_prefix, limit):
        yield content[0]


def list_s3_objects_detailed(client, path: str, recursive=False, is_prefix=False, limit=0):
    if not path.endswith("/") and not is_prefix:
        path += "/"
    bucket, prefix = split_s3_path(path)
    marker = None
    while True:
        list_kwargs = dict(MaxKeys=1000, Bucket=bucket, Prefix=prefix)
        if limit:
            list_kwargs["MaxKeys"] = limit
        if not recursive:
            list_kwargs["Delimiter"] = "/"
        if marker:
            list_kwargs["Marker"] = marker
        response = client.list_objects(**list_kwargs)
        # TODO: not sure if cp in next pages is empty or not.
        if not recursive:
            for cp in response.get("CommonPrefixes", []):
                yield (f"s3://{bucket}/{cp['Prefix']}", cp)
        contents = response.get("Contents", [])
        for content in contents:
            if not content["Key"].endswith("/"):
                yield (f"s3://{bucket}/{content['Key']}", content)
        if limit or not response.get("IsTruncated") or len(contents) == 0:
            break
        marker = contents[-1]["Key"]


class S3UploadAcc:
    def __init__(self, sc):
        from app.common.spark_ext import DictAccumulatorParam

        self.acc = sc.accumulator({}, DictAccumulatorParam())

    def incr(self, field: str, sub_path: str, value: Union[int, list]):
        self.acc.add({f"_:{field}": value})
        if sub_path:
            self.acc.add({f"{sub_path}:{field}": value})

    def to_dict(self) -> dict:
        acc_value: Dict[str, int] = self.acc.value  # type: ignore
        sub_paths = {}

        for key, value in acc_value.items():
            sub_path, field = key.split(":")
            sub_path_dict = sub_paths.get(sub_path, {})
            sub_path_dict[field] = value
            sub_paths[sub_path] = sub_path_dict

        d = {**sub_paths.get("_", {})}
        d["sub_paths"] = dict(sorted(filter(lambda i: i[0] != "_", sub_paths.items())))
        return d


@retry(wait=wait_random(min=300, max=600), stop=stop_after_attempt(3))
def s3_upload_with_retry(client, tmp_filename, bucket, key, config):
    client.upload_file(tmp_filename, bucket, key, Config=config)


def upload_to_s3(output_path: str, ext: str, acc: S3UploadAcc, track_ratio: float, skip_loc=False):
    from pyspark import TaskContext
    from pyspark.sql import Row

    uid = str(uuid.uuid4())[:8]
    prefix = output_path.rstrip("/")

    def mix(d: dict, row: Row, changed=False) -> dict:
        ret = {**d}
        for key, val in row.asDict().items():
            if key != "value" and key not in d:
                ret[key] = val
        if changed:
            ret["changed"] = True
        return ret

    def handle(iter: Iterator[Row]):
        ctx = TaskContext.get()

        if ctx is None:
            raise Exception("cannot get task context.")

        # avoid importing pymongo if track_ratio is le zero.
        doc_tracker = None
        if track_ratio is None or track_ratio > 0.0:
            if "/debug/" in output_path:
                from app.common.cassandra import CassandraDocTracker as DocTracker
            else:
                from app.common.track import DocTracker

            doc_tracker = DocTracker(output_path, track_ratio)

        def get_output_file(sub_path):
            output_name = f"part-{str(ctx.partitionId()).zfill(6)}-{uid}.{ext}"
            if sub_path:
                output_file = f"{prefix}/{sub_path}/{output_name}"
            else:
                output_file = f"{prefix}/{output_name}"
            return output_file

        # sub_path -> (tmp_fh, tmp_filename, offset, output_file)
        tmp_files = {}
        try:
            for row in iter:
                d = json_loads(str(row.value))
                sub_path = (mix(d, row).get(FIELD_SUB_PATH) or "").strip("/")

                # handle flags and metrics
                for key, val in d.items():
                    if is_flag_field(key) and val == True:
                        acc.incr(key, sub_path, 1)
                    elif is_acc_field(key) and type(val) == int and val >= 0:
                        acc.incr(key, sub_path, [val])

                if d.get("dropped"):
                    if doc_tracker:
                        doc_tracker.track(mix(d, row), sub_path)
                    acc.incr("dropped", sub_path, 1)
                    continue

                changed = False
                if "changed" in d:
                    changed = bool(d["changed"])
                    del d["changed"]

                if changed:
                    acc.incr("changed", sub_path, 1)

                tmp_file = tmp_files.get(sub_path)
                if not tmp_file:
                    tmp_filename = os.path.join(TMP_DIR, f"{str(uuid.uuid4())}.{ext}")
                    tmp_fh = open(tmp_filename, "a", encoding="utf-8")
                    tmp_file = (tmp_fh, tmp_filename, [0], get_output_file(sub_path))
                    tmp_files[sub_path] = tmp_file

                tmp_fh, _, offset, output_file = tmp_file

                # delete deprecated fields.
                if "doc_offset" in d:
                    del d["doc_offset"]
                if "doc_length" in d:
                    del d["doc_length"]

                if not skip_loc and "doc_loc" in d:
                    track_loc = d.get("track_loc") or []
                    track_loc.append(d["doc_loc"])
                    d["track_loc"] = track_loc

                doc = json_dumps(d)
                doc_length = len(doc.encode("utf-8"))

                # add doc_loc if doc has id
                if not skip_loc and FIELD_ID in d:
                    last_doc_length = 0
                    while last_doc_length != doc_length:
                        last_doc_length = doc_length
                        d["doc_loc"] = f"{output_file}?bytes={offset[0]},{doc_length}"
                        doc = json_dumps(d)
                        doc_length = len(doc.encode("utf-8"))

                tmp_fh.write(doc)
                tmp_fh.write("\n")
                offset[0] += doc_length + 1
                acc.incr("rows", sub_path, 1)
                acc.incr("bytes", sub_path, [doc_length + 1])

                if "content" in d and type(d["content"]) is str:
                    acc.incr("cbytes", sub_path, [len(d["content"].encode("utf-8"))])

                if doc_tracker:
                    doc_tracker.track(mix(d, row, changed), sub_path)

            if doc_tracker:
                doc_tracker.flush()

            client = get_s3_client(output_path)
            for sub_path in tmp_files.keys():
                tmp_fh, tmp_filename, _, output_file = tmp_files[sub_path]
                tmp_fh.close()

                # upload
                MB = 1024**2
                config = TransferConfig(
                    multipart_threshold=128 * MB,
                    multipart_chunksize=16 * MB,
                )
                bucket, key = split_s3_path(output_file)
                s3_upload_with_retry(client, tmp_filename, bucket, key, config)
                acc.incr("files", sub_path, 1)

        finally:
            for sub_path in tmp_files.keys():
                tmp_filename = tmp_files[sub_path][1]
                os.remove(tmp_filename)

    return handle


def __write_mark_in_path(mark: str, path: str, body_text: str = ""):
    prefix = ensure_s3_path(path)
    mark_file = f"{prefix.rstrip('/')}/{mark}"
    client = get_s3_client(mark_file)
    body = body_text.encode("utf-8")
    put_s3_object(client, mark_file, body)


def mark_failure_in_s3(output_path: str, task_info: dict):
    body_text = json_dumps(task_info)
    __write_mark_in_path(FAILURE_MARK_FILE, output_path, body_text)


def mark_success_in_s3(output_path: str, task_info: dict):
    body_text = json_dumps(task_info)
    __write_mark_in_path(SUCCESS_MARK_FILE, output_path, body_text)


def mark_reserve_in_s3(output_path: str):
    __write_mark_in_path(RESERVE_MARK_FILE, output_path)


def mark_deleted_in_s3(output_path: str):
    __write_mark_in_path(DELETED_MARK_FILE, output_path)


def mark_summary_in_s3(output_path: str, summary_info: dict):
    summary_pairs = []
    for k, v in summary_info.items():
        if k == "sub_paths" or is_flag_field(k) or is_acc_field(k):
            continue
        if type(v) == dict:
            v = v["sum"]
        summary_pairs.append(f"_{k}_{v}")

    mark_file = f'{SUMMARY_MARK_FILE}{"".join(summary_pairs)}'
    body_text = json_dumps(summary_info)
    __write_mark_in_path(mark_file, output_path, body_text)


def is_s3_empty_path(output_path: str) -> bool:
    check_path = output_path.rstrip("/") + "/"
    client = get_s3_client(check_path)
    contents = list_s3_objects(client, check_path, recursive=True, limit=10)
    for c in contents:
        if c.endswith(RESERVE_MARK_FILE):
            continue
        return False
    return True


def is_s3_success_path(input_path: str) -> bool:
    client = get_s3_client(input_path)

    def is_success_path(path: str) -> bool:
        check_path = path.rstrip("/") + "/"
        for mark in [SUCCESS_MARK_FILE, SUCCESS_MARK_FILE2]:
            if head_s3_object(client, check_path + mark):
                return True
        return False

    if is_success_path(input_path):
        return True

    sub_dirs = list(list_s3_objects(client, input_path))

    # fmt: off
    return len(sub_dirs) > 0 and \
        all([dir.endswith("/") for dir in sub_dirs]) and \
        all([is_success_path(dir) for dir in sub_dirs])
    # fmt: on


def is_s3_path_exists(path: str) -> bool:
    check_path = path.rstrip("/") + "/"
    client = get_s3_client(check_path)
    contents = list_s3_objects(client, check_path, limit=10)
    for c in contents:
        return True
    return False


def detect_s3_multi_layer_path(input_path: str) -> str:
    if "*" in input_path:
        return ensure_s3a_path(input_path)

    check_path = ensure_s3_path(input_path.rstrip("/") + "/")
    client = get_s3_client(check_path)
    contents = list_s3_objects(client, check_path, recursive=True, limit=10)

    relative_path = None
    for c in contents:
        sub_path = c[len(check_path) :]
        if sub_path.startswith("_") or sub_path.startswith("."):
            continue
        relative_path = sub_path
        break

    if not relative_path:
        raise Exception(f"cannot find file in input path [{input_path}].")

    num_relative_parts = len(relative_path.split("/"))
    if num_relative_parts > 1:
        check_path += "*/" * (num_relative_parts - 1)

    return ensure_s3a_path(check_path)


def head_s3_object(client, path: str) -> Union[Dict, None]:
    bucket, key = split_s3_path(path)
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        return resp
    except Exception:
        return None



s3_assets_path = "s3://llm-process-snew/_assets"
lock_timeout_seconds = 300
TMP_DIR = os.environ.get("TMP_DIR") or "/tmp"


def touch(path):
    with open(path, "a"):
        os.utime(path, None)


def try_remove(path):
    try:
        os.remove(path)
    except Exception:
        pass


def download_s3_asset(asset_name: str, s3_cfg) -> str:
    """
    load asset and return local path.
    raise Exception if asset absent.

    example asset_name: fasttext_model/model.bin
    example assets path on s3: s3://llm-process-snew/_assets/fasttext_model/model.bin
    """

    asset_name = asset_name.strip("/")

    if is_s3_path(asset_name):
        s3_path = asset_name
        bucket, key = split_s3_path(s3_path)
        slug_name = f"{bucket}/{key}".replace("/", "__")
    else:
        s3_path = f"{s3_assets_path}/{asset_name}"
        slug_name = asset_name.replace("/", "__")
    if len(slug_name) > 50:
        slug_name = slug_name[-50:]
    local_path = os.path.join(TMP_DIR, f"asset__{slug_name}")
    lock_path = f"{local_path}.lock"
    client = get_s3_client(s3_path, s3_cfg)

    def wait_for_lock():
        while os.path.exists(lock_path):
            try:
                lock_ctime = os.path.getctime(lock_path)
                lock_elapsed = time.time() - lock_ctime
                if lock_elapsed > lock_timeout_seconds:
                    print(f"remove lock: {lock_path}")
                    try_remove(lock_path)
                    break
            except Exception:
                pass
            print(f"wait for lock: {lock_path}")
            time.sleep(5)

    def download_file():
        print(f"downloading {s3_path} ...")
        try:
            bucket, key = split_s3_path(s3_path)
            client.download_file(bucket, key, local_path)
        except Exception as e:
            try_remove(local_path)
            raise e
        finally:
            try_remove(lock_path)

    while True:
        wait_for_lock()

        if os.path.exists(local_path):
            print("local asset found.")
            local_asset_size = os.path.getsize(local_path)
            s3_head = head_s3_object(client, s3_path)
            s3_asset_size = s3_head["ContentLength"] if s3_head else -1
            if local_asset_size == s3_asset_size:
                print("local asset size OK.")
                return local_path

        if os.path.exists(lock_path):
            continue

        touch(lock_path)
        download_file()
        return local_path


def get_s3_cfg_by_bucket(s3_path):
    ### 根据bucket名称判断使用哪个配置 ###
    with open(os.path.join(os.path.expanduser("~"), ".aws/s3_bucket_cfg.json"),'r') as f:    
        s3_bucket_cfg = json.load(f)

    bucket_name = re.findall("/(.*)/", s3_path)[0].lstrip("/").split("/")[0]
    if bucket_name in s3_bucket_cfg:
        return s3_bucket_cfg[bucket_name]
    else:
        print("! Unkown bucket name: ", bucket_name, "try default cfg(langchao_xuchao)")
        return s3_bucket_cfg['llm-pdf-text']