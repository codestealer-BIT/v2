import argparse



import numpy as np
import torch
import tqdm
import os
import json
import base64

import sys
sys.path.append('/opt/tiger/MMPretrain')

from config import cfg

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from video_clip import VideoMusicTextCLIP
from dataset import VideoRetrievalDataset

import os
import math
import time
#from tiktok_cu_review_callback.item_features_handler import ItemInfo
from bytedance import metrics
import bytedlogger
from laplace import Laplace
import byted_tensorproto as tb
bytedlogger.config_default()
def tb_decode(x):
    if isinstance(x, str):
        return tb.decode(x.encode()).tolist()
    return tb.decode(x).tolist()
def get_main_idc():
    try:
        f = os.popen("""sd report|grep "Data center"|awk '{print $3}'""")
        idc = f.readlines()[0].strip()
    except Exception as e:
        idc = 'maliva'
    if idc in ['sg1', 'my', 'my2', 'alisg']:
        idc = 'sg1'
    if idc in ['useast5', 'useast8']:
        idc = 'useast5'
    if idc in ['useast2a', 'ie', 'no1a']:
        idc = 'useast2a'
    return idc


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



import os
import math
import time
from bytedance import metrics
import bytedlogger
from laplace import Laplace
import byted_tensorproto as tb
bytedlogger.config_default()
def tb_decode(x):
    if isinstance(x, str):
        return tb.decode(x.encode()).tolist()
    return tb.decode(x).tolist()
def get_main_idc():
    try:
        f = os.popen("""sd report|grep "Data center"|awk '{print $3}'""")
        idc = f.readlines()[0].strip()
    except Exception as e:
        idc = 'maliva'
    if idc in ['sg1', 'my', 'my2', 'alisg']:
        idc = 'sg1'
    if idc in ['useast5', 'useast8']:
        idc = 'useast5'
    if idc in ['useast2a', 'ie', 'no1a']:
        idc = 'useast2a'
    return idc

class MMPretrainV3():
    def __init__(self, debug=False):
        self.debug = debug
        self.metric_client = metrics.Client(prefix="tiktok.data.vine")
        self.tce_dc = 'maliva'
        self.model_name = 'tt_mmpretrain_v3'
        self.psm = 'tiktok.data.mmpretrain_embedding_v3'
        self.laplace = Laplace('{}?cluster=default&idc={}'.format(self.psm, self.tce_dc), timeout=2)
        self.image_num = 8
        self.retry_time = 2
    def process_input(self, frames):
        data = {}
        data['title'] = ["text".encode()]
        data['stickers'] = [b'']
        data['challenge'] = [b'']
        frames = [base64.b64decode(im) for im in frames]
        for i in range(self.image_num):
            data['frame%s'%i] = [frames[i]]
        return data
    def parse_rsp(self, rsp):
        mm_embedding_byte = rsp.output_bytes_lists['mm_embedding'][0]
        visual_embedding_byte = rsp.output_bytes_lists['visual_embedding'][0]
        text_embedding_byte = rsp.output_bytes_lists['text_embedding'][0]
        mm_embedding = list(tb_decode(mm_embedding_byte))
        visual_embedding = list(tb_decode(visual_embedding_byte))
        text_embedding = list(tb_decode(text_embedding_byte))
        # check nan
        mm_nan = math.isnan(mm_embedding[0])
        visual_nan = math.isnan(visual_embedding[0])
        text_nan = math.isnan(text_embedding[0])
        if self.debug:
            print("------MMPretrain V3 Model------------")
            print("pretrain mm_embedding", mm_embedding)
            print("pretrain visual_embedding", visual_embedding)
            print("pretrain text_embedding", text_embedding)
        ret = {}
        if not mm_nan:
            ret.update({"mm_embedding": mm_embedding})
        if not visual_nan:
            ret.update({"visual_embedding": visual_embedding})
        if not text_nan:
            ret.update({"text_embedding": text_embedding})
        return ret
    def inference(self, frames):
        
        data = {}
        try:
            data = self.process_input(frames)
        except Exception as e:
            bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} process_input error: {}".format(self.model_name, e))
        # model inference
        for _retry_i in range(self.retry_time):
            try:
                # befor rsp
                is_rsp = 0
                rsp = self.laplace.matx_inference(self.model_name, data)
                # parse rsp
                is_rsp = 1
                raw_feature = self.parse_rsp(rsp)
                self.metric_client.emit_counter('model.inference.success', 1, tags={"model_name": self.model_name, 'gid_dc': '123'})
                return raw_feature
            except Exception as e:
                bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} , gid: {}, retry: {}, inference error: {}".format(self.model_name, '123', _retry_i, e))
                if _retry_i == self.retry_time - 1:
                    self.metric_client.emit_counter('model.inference.failed', 1, tags={"model_name": self.model_name, 'gid_dc': '123', 'is_rsp': str(is_rsp)})
        return {}


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def validate_msrvtt(vision_model, text_model, dataloader,
                    recall_k_list=[1, 5, 10],
                    eval_batch_size=32, output_dir=None):
    device = "cuda"

    video_features = []
    text_features = []

    # compute text and vision features
    print('Computing text and vision features', flush=True)
    for idx,data in tqdm.tqdm(enumerate(dataloader)):
        captions = data['captions']
        
        
        text_feat = []
        vis_feat = []

        
        for i in range(len(captions)):
            try:
                text_emb = text_model.inference({"item_id":123,"text":captions[i]})
                frames = data['video_b64'][i].split(',')
                vis_emb = vision_model.inference(frames)['visual_embedding']
                vis_feat.append(vis_emb)
                text_feat.append(text_emb)
            except Exception:
                print('encounter broken file: %s' % (captions[i]))

        video_features.append(torch.tensor(vis_feat))
        text_features.append(torch.tensor(text_feat))
    
    text_features = torch.cat(text_features)
    video_features = torch.cat(video_features)

    print('Computing metrics', flush=True)

    texts_emb = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    images_emb = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

    # get the score for each text and image pair
    scores = torch.matmul(texts_emb, images_emb.t().to(texts_emb.device))
    # logit_scale, logit_bias = model.logit_scale.to(texts_emb.device), model.logit_bias.to(texts_emb.device)
    # scores = scores * logit_scale.exp() + logit_bias
    probs = torch.sigmoid(scores)

    for i in range(20):
        print(f"{probs[i][i]:.4%} that image 0 is target")

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True

    scores_T = scores.T
    positive_pairs_T = positive_pairs.T

    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f't2v_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores, positive_pairs, eval_batch_size, scores.device,
                             k=recall_k) > 0).float().mean().item()
        metrics[f'v2t_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores_T, positive_pairs_T, eval_batch_size, scores.device,
                             k=recall_k) > 0).float().mean().item()

    print(metrics)

    if output_dir is not None:
        with open(output_dir + "/val_log.txt", "a") as f:
            f.write(json.dumps(metrics) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate MSR-VTT', add_help=False)
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--video_root',default="/mnt/bn/yexiaoyu-test/data/lavis/msr-vtt/MSRVTT/videos/all", type=str, help='msr vtt video paths')
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--anno_file_path', default='/mnt/bn/yexiaoyu-test/data/lavis/msr-vtt/msrvtt_test.jsonl', type=str, help="The annotation path")
    #parser.add_argument('--pretrained_path',default='/mnt/bn/yexiaoyu-test/models/model_state_epoch_170000.th', type=str, help='')
    parser.add_argument('--output_path',default=None, type=str, help='validation results export path')
    parser.add_argument('--batch_size',default=8, type=int, help='evaluation batch size')
    args = parser.parse_args()

    

    test_dataset = VideoRetrievalDataset(video_root=args.video_root, 
                 ann_file=args.anno_file_path, 
                 max_text_len = 64,
                 max_frame_len=args.max_frames,
                 tokenizer_dir=args.model_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_collate,
    ) 
    vision_model = MMPretrainV3()
    text_model = MMPretrainModel(tce_dc="maliva")
    metrics = validate_msrvtt(vision_model=vision_model, text_model= text_model, dataloader = test_loader, eval_batch_size=args.batch_size, output_dir = args.output_path)





# class MMPretrainV3():
#     def __init__(self, debug=False):
#         self.debug = debug
#         #self.item_info = ItemInfo(frame_num=16, enable_ocr=True, enable_asr=True)
#         self.metric_client = metrics.Client(prefix="tiktok.data.vine")
#         self.tce_dc = 'maliva'
#         self.model_name = 'tt_mmpretrain_v3'
#         self.psm = 'tiktok.data.mmpretrain_embedding_v3'
#         self.laplace = Laplace('{}?cluster=default&idc={}'.format(self.psm, self.tce_dc), timeout=2)
#         self.image_num = 8
#         self.retry_time = 1
#     def process_input(self, frames):
#         data = {'item_id': '123'}
#         data['title'] = [b'']
#         data['stickers'] = [b'']
#         data['challenge'] = [b'']
#         frames = [base64.b64decode(im.encode('utf-8')) for im in frames]
#         #print("type(frames[0]):", type(frames[0]))
#         # data["frames"] = frames
#         for i in range(self.image_num):
#             print('frame%s'%i)
#             data['frame%s'%i] = [frames[i]]
#         return data
#     def parse_rsp(self, rsp):
#         #object_id = inputs['gid']
#         mm_embedding_byte = rsp.output_bytes_lists['mm_embedding'][0]
#         visual_embedding_byte = rsp.output_bytes_lists['visual_embedding'][0]
#         text_embedding_byte = rsp.output_bytes_lists['text_embedding'][0]
#         mm_embedding = list(tb_decode(mm_embedding_byte))
#         visual_embedding = list(tb_decode(visual_embedding_byte))
#         text_embedding = list(tb_decode(text_embedding_byte))
#         # check nan
#         mm_nan = math.isnan(mm_embedding[0])
#         visual_nan = math.isnan(visual_embedding[0])
#         text_nan = math.isnan(text_embedding[0])
#         if self.debug:
#             print("------MMPretrain V3 Model------------")
#             print("pretrain mm_embedding", mm_embedding)
#             print("pretrain visual_embedding", visual_embedding)
#             print("pretrain text_embedding", text_embedding)
#         ret = {}
#         if not mm_nan:
#             ret.update({"mm_embedding": mm_embedding})
#         if not visual_nan:
#             ret.update({"visual_embedding": visual_embedding})
#         if not text_nan:
#             ret.update({"text_embedding": text_embedding})
#         return ret
#     def inference(self, frames):
#         data = {}
#         try:
#             data = self.process_input(frames)
#         except Exception as e:
#             bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} process_input error: {}".format(self.model_name, e))
#         # model inference
#         for _retry_i in range(self.retry_time):
#             try:
#                 # befor rsp
#                 is_rsp = 0
#                 rsp = self.laplace.matx_inference(self.model_name, data)
#                 # parse rsp
#                 is_rsp = 1
#                 raw_feature = self.parse_rsp(rsp)
#                 self.metric_client.emit_counter('model.inference.success', 1, tags={"model_name": self.model_name, 'gid_dc': 0})
#                 return raw_feature
#             except Exception as e:
#                 bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} , gid: {}, retry: {}, inference error: {}".format(self.model_name, 123, _retry_i, e))
#                 if _retry_i == self.retry_time - 1:
#                     self.metric_client.emit_counter('model.inference.failed', 1, tags={"model_name": self.model_name, 'gid_dc': '0', 'is_rsp': str(is_rsp)})
#         return {}