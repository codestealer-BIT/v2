import argparse



import numpy as np
import torch
import tqdm
import os
import json

import sys
sys.path.append('/opt/tiger/MMPretrain')

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from dataset import TikTokRetrievalDataset


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
import thriftpy2
import euler
import bytedlogger
import numpy as np
from euler import base_compat_middleware
bytedlogger.config_default()
dirpath = os.path.dirname(os.path.realpath(__file__))
class UeEmbedding(object):
    def __init__(self, ue_config='fc_tiktok_ps_multi_modal_rec'):
        self.ue_config = ue_config
        self.base_info = thriftpy2.load(os.path.join("/opt/tiger/MMPretrain", "mmembed/idls/base.thrift"))
        self.emb_info = thriftpy2.load(os.path.join("/opt/tiger/MMPretrain", "mmembed/idls/embedding_flow.thrift"))
        self.ue_info = thriftpy2.load(os.path.join("/opt/tiger/MMPretrain", 'mmembed/idls/universal_embeddings.thrift'))
        self.ue_client = euler.Client(self.ue_info.UniversalEmbeddingService,
                                      "sd://data.embedding.feed?cluster=default&idc=maliva",
                                      timeout=1) #idc=maliva&
        self.source = self.ue_info.EmbeddingSource(embedding_name=ue_config)
        self.embedding_info = self.ue_info.EmbeddingInfo(source=self.source)
        self.base = self.base_info.Base(Caller="data.ue_service.dorado.debug")
    def get(self, key, extract_inside_ue=True):
        #print(key)
        fid = self.ue_info.FIDInfo(fids=[key], extract_inside_ue=extract_inside_ue)
        #print(fid)
        req = self.ue_info.Request(fid_infos=[fid], embedding_info=self.embedding_info, Base=self.base)
        #print(req)
        resp = self.ue_client.get_embedding(req)
        #print(resp)
        return np.array(resp.fid_rsps[0].vals)
    def batch_get(self, keys):
        fids = []
        for key in keys:
            fids.append(self.ue_info.FIDInfo(fids=[key], extract_inside_ue=True))
        #print(fid)
        req = self.ue_info.Request(fid_infos=fids, embedding_info=self.embedding_info, Base=self.base)
        #print(req)
        resp = self.ue_client.get_embedding(req)
        vals = []
        for rsp in resp.fid_rsps:
            vals.append(rsp.vals)
        return vals


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

def clean_lists(A, B):
    """
    Remove elements from A that are all 0.0, and their corresponding elements from B.
    """
    new_A, new_B = [], []
    removed_ids = []
    for i, (a, b) in enumerate(zip(A, B)):
        if all(x == 0.0 for x in a):  # check if all elements in sublist are 0.0
            removed_ids.append(i)
            #print(f"Warning: element {i} in mm embedding is all zeros, removing it and its pair in captions")
            continue
        new_A.append(a)
        new_B.append(b)
    return new_A, new_B, removed_ids

def merge_lists(A, B):
    """
    Merge two lists of lists element-wise:
    - If A[i] is all zeros and B[i] is not, take B[i].
    - Otherwise keep A[i].
    """
    merged = []
    for i, (a, b) in enumerate(zip(A, B)):
        is_a_zero = all(x == 0.0 for x in a)
        is_b_zero = all(x == 0.0 for x in b)
        
        if is_a_zero and not is_b_zero:
            merged.append(b)
        else:
            merged.append(a)
    return merged




def validate_msrvtt(model, ue, ue1, dataloader,
                    recall_k_list=[1, 5, 10],
                    eval_batch_size=32, output_dir=None):

    video_features = []
    text_features = []

    # compute text and vision features
    print('Computing text and vision features', flush=True)
    for idx,data in tqdm.tqdm(enumerate(dataloader)):
        
        captions = data['fusion_caption']
        item_ids = [int(ids) for ids in data['item_ids']]
        caption_pooled = []

        fused_0 = ue.batch_get(item_ids)
        fused_pooled = fused_0
        fused_1 = ue1.batch_get(item_ids)

        fused_pooled = merge_lists(fused_0,fused_1)

        for cap in captions:
            res = model.inference({"item_id":123,"text":cap})
            caption_pooled.append(res)
        
        fused_pooled,caption_pooled,removed_ids = clean_lists(fused_pooled,caption_pooled)
        for id in removed_ids:
            print("got zero embedding in: ", item_ids[id])

        video_features.append(torch.Tensor(fused_pooled))
        text_features.append(torch.Tensor(caption_pooled))
    
    text_features = torch.cat(text_features)
    video_features = torch.cat(video_features)

    print('Computing metrics', flush=True)

    texts_emb = text_features# / text_features.norm(p=2, dim=-1, keepdim=True)
    images_emb = video_features# / video_features.norm(p=2, dim=-1, keepdim=True)

    # get the score for each text and image pair
    scores = torch.matmul(texts_emb, images_emb.t().to(texts_emb.device))
    #logit_scale, logit_bias = model.logit_scale.to(texts_emb.device), model.logit_bias.to(texts_emb.device)
    #scores = scores * logit_scale.exp() + logit_bias
    probs = torch.sigmoid(scores)

    for i in range(20):
        print(f"{probs[i][i]:.4%} that image 0 is target")
    #print(prob[:8][:8])

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
    parser.add_argument('--bert_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased", type=str, help='the config path of the bert tokenizer to be used')
    parser.add_argument('--video_root',default="/mnt/bn/jiny-ttls-i18n-fr1q/tiktok_30k_recaption_long_fusion.jsonl", type=str, help='video paths')
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--pretrained_path',default=None, type=str, help='the pretrained siglip2 models to be used')
    parser.add_argument('--output_path',default=None, type=str, help='validation results export path')
    parser.add_argument('--batch_size',default=8, type=int, help='evaluation batch size')
    args = parser.parse_args()

    

    test_dataset = TikTokRetrievalDataset(video_root=args.video_root, 
                max_text_len=256,
                max_frame_len=args.max_frames,
                tokenizer_dir=args.model_path,
                bert_tokenizer_dir=args.bert_path,
                )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_collate,
    ) 
    model = MMPretrainModel(tce_dc="maliva")
    ue = UeEmbedding('fc_tiktok_mmpretrainv3_pskv')
    ue1 = UeEmbedding('fc_tiktok_mmpretrain_v3_mm_emb1')
    # ue = UeEmbedding('fc_tiktok_mmpretrain_v3_visual_emb1')
    metrics = validate_msrvtt(model = model, ue = ue,  ue1 = ue1, dataloader = test_loader, eval_batch_size=args.batch_size, output_dir = args.output_path)