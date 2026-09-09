# 调用MMPretrain v3 text Model脚本
# NOTE：模型输出embedding, 
#          纯文本的text_embedding
# NOTE：embedding均为128维，norm经过了归一化
# NOTE：如发现代码有bug，请联系@zhouxinzhe
import json
import bytedlogger
#import tensorflow as tf
import numpy as np
from laplace import Laplace
import bytedabase
import os
import concurrent.futures
import byted_tensorproto as tb
from typing import IO, Any, List
from contextlib import contextmanager
import subprocess
import cityhash
HADOOP_BIN = 'HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs'
def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    is_hdfs = hdfs_path.startswith('hdfs')
    if is_hdfs:
        return hdfs_open(hdfs_path, mode)
    else:
        return open(hdfs_path, mode)
@contextmanager  # type: ignore
def hdfs_open(hdfs_path: str, mode: str = "r") -> IO[Any]:
    """ 
        打开一个 hdfs 文件, 用 contextmanager.
        Args:
            hfdfs_path (str): hdfs文件路径
            mode (str): 打开模式，支持 ["r", "w", "wa"]
    """
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.wait()
        return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))
def tb_decode(x):
    if isinstance(x, str):
        return tb.decode(x.encode()).tolist()
    return tb.decode(x).tolist()
# class AbaseHandler():
#     def __init__(self, psm='',
#                  table_name='',
#                  source=''):
#         self.abase_client = bytedabase.Client(psm=psm, table=table_name)
#         self.source = source
#     def hash_slot(self, value, slot, version=1):
#         if version == 1:
#             HASH_KEY_BITS = 54
#         else:
#             HASH_KEY_BITS = 48
#         return ((cityhash.CityHash64(str(value)) & ((1 << HASH_KEY_BITS) - 1)) | slot << HASH_KEY_BITS)
#     def convert2key(self, item_id, version=1):
#         fid = self.hash_slot(item_id, slot=2, version=version)
#         key = "{},{}".format(self.source, fid)
#         return key
#     def get(self, item_id, version=1):
#         key = self.convert2key(item_id, version=version)
#         return self.abase_client.get(key)
#     def exists(self, item_ids, version=1):
#         keys = [self.convert2key(item_id, version=version) for item_id in item_ids]
#         return self.abase_client.exists(*keys)
#     #def set(self, item_id, value, ex=7776000, write_bytes=True, version=1):  # 默认过期时间为90天
#     #def set(self, item_id, value, ex=31536000, write_bytes=True, version=1):  # 默认过期时间为365天
#     def set(self, item_id, value, ex=7776000, write_bytes=True, version=1):  # 默认过期时间为90天
#         key = self.convert2key(item_id, version=version)
#         if write_bytes:
#             if isinstance(value, list):
#                 value = tf.make_tensor_proto(value).SerializeToString()
#         else:
#             if isinstance(value, list):
#                 value = json.dumps(value)
#         retry = 2  # 最多尝试2次
#         flag = False
#         while retry > 0:
#             try:
#                 flag = self.abase_client.set(key, value, ex=ex)
#                 if flag:
#                     break
#             except Exception as e:
#                 if retry == 1:
#                     #bytedlogger.logging.error('[DEBUG_LOG][ITEM_FEATURE][AbaseHandler] set abase exc={}, item_id={}, source={}'.format(e, item_id, self.source))
#                     pass
#             retry -= 1
#         return flag
# #bytedlogger.logging.info(f"abase info: {abases}")
class MMPretrainModel():
    def __init__(self, tce_dc='maliva', retry_time=1):
        self.model_name = 'tt_mmpretrain_v3_text'
        self.psm = 'tiktok.data.mmpretrain_embedding_v3_text'
        self.tce_dc = tce_dc
        self.laplace = Laplace('{}?cluster=default&idc={}'.format(self.psm, self.tce_dc), timeout=2)
        self.retry_time = retry_time
    def process_input(self, inputs):                                                          
        data = {}
        data['title'] = [inputs['text'].encode()]
        data['stickers'] = [b'']
        data['challenge'] = [b'']
        #print(data)
        return data
    def parse_rsp(self, inputs, rsp,abases):
        object_id = int(inputs['item_id'])
        text_embedding_byte = rsp.output_bytes_lists['text_embedding'][0]
        # debug
        #mm_embedding = list(tb_decode(mm_embedding_byte))
        #visual_embedding = list(tb_decode(visual_embedding_byte))
        text_embedding = list(tb_decode(text_embedding_byte))
        #print(f"mm_emb:\n{mm_embedding}")
        #print(f"visual_emb:\n{visual_embedding}")
        #print(f"text_emb:\n{text_embedding}")
        return text_embedding
        """
        code = 0
        for region in abases.keys():
            abase = abases[region]
            version = abase["version"]
            #f3 = abase["ocr"].set(object_id, text_embedding_byte, version=version)
            f3 = abase["asr"].set(object_id, text_embedding_byte, version=version)
            code = (code<<1) + int(f3)
        return code
        """
    def inference(self,inputs):
        data = {}
        try:
            data = self.process_input(inputs)
        except Exception as e:
            #bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} process_input error: {}".format(self.model_name, e))
            return -1
        # model inference
        for _retry_i in range(self.retry_time):
            try:
                # befor rsp
                is_rsp = 0
                rsp = self.laplace.matx_inference(self.model_name, data)
                # parse rsp
                is_rsp = 1
                code = self.parse_rsp(inputs, rsp,abases=None)
                return code
            except Exception as e:
                #bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} , item_id: {}, retry: {}, inference error: {}".format(self.model_name, inputs['item_id'], _retry_i, e))
                if _retry_i == self.retry_time - 1:
                    return None
        return None

model = MMPretrainModel(tce_dc="maliva")
res = model.inference({"item_id":123,"text":"Hello world"})
print(len(res))