from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import math
import time
#from tiktok_cu_review_callback.item_features_handler import ItemInfo
from bytedance import metrics
import bytedlogger
from laplace import Laplace
import byted_tensorproto as tb
bytedlogger.config_default()

import torch
import numpy as np
import random
import os
import sys
sys.path.append('/opt/tiger/MMPretrain')
import base64

from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse

from util import parallel_apply, get_logger
from eval_dataloaders import DATALOADER_DICT

global logger

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

def get_args(description='Siglip2 on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--dict_path', type=str, default='data/videos_feature.pickle', help='gpt caption path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=64, help='')
    parser.add_argument('--max_frames', type=int, default=8, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_config", default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help="pretrained_config")
    parser.add_argument("--pretrained_model", default="/mnt/bn/yexiaoyu-test/checkpoints/debug_dataset_fix", type=str, help="pretrained_model")
    parser.add_argument("--blip_path", default="/mnt/bn/yexiaoyu-test/models/huggingface/blip2", type=str, help="pretrained_model")

    args = parser.parse_args()

    return args

def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_model():

    vision_model = MMPretrainV3()
    text_model = MMPretrainModel(tce_dc="maliva")

    return vision_model, text_model




def _run_on_single_gpu(batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_sequence_output_list):
        each_row = []
        for idx2, b2 in enumerate(batch_visual_output_list):
            
            texts_emb = b1 / b1.norm(p=2, dim=-1, keepdim=True)
            images_emb = b2 / b2.norm(p=2, dim=-1, keepdim=True)

            # get the score for each text and image pair
            scores = torch.matmul(texts_emb, images_emb.t().to(texts_emb.device))
            
            b1b2_logits = scores.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, vision_model, text_model, test_dataloader):

    

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    assert multi_sentence_ == False, "multi_sentence not implemented"

    with torch.no_grad():
        
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t for t in batch)
            _,_,_,_,captions,videos = batch
            videos = [vid.split(',') for vid in videos]

            text_feat = []
            vis_feat = []

            
            for i in range(len(captions)):
                try:
                    text_emb = text_model.inference({"item_id":123,"text":captions[i]})
                    frames = videos[i]
                    vis_emb = vision_model.inference(frames)['visual_embedding']
                    vis_feat.append(vis_emb)
                    text_feat.append(text_emb)
                except Exception:
                    print('encounter broken file: %s' % (captions[i]))


            batch_sequence_output_list.append(torch.tensor(text_feat))
            batch_visual_output_list.append(torch.tensor(vis_feat))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        
        sim_matrix = _run_on_single_gpu(batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)


    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def main():
    global logger

    args = get_args()

    args = set_seed_logger(args)
    

    assert  args.task_type == "retrieval"
    vision_model, text_model = init_model()

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, args.pretrained_config)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, args.pretrained_config, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", test_length)
    logger.info("  Batch size = %d", args.batch_size_val)
    logger.info("  Num steps = %d", len(test_dataloader))
    logger.info("***** Running val *****")
    logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    
    eval_epoch(args, vision_model,text_model, test_dataloader)

if __name__ == "__main__":
    main()
