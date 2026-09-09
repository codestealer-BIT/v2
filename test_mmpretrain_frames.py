import os
import math
import time
from .item_features_handler import ItemInfo
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
        self.item_info = ItemInfo(frame_num=16, enable_ocr=True, enable_asr=True)
        self.metric_client = metrics.Client(prefix="tiktok.data.vine")
        self.tce_dc = get_main_idc()
        self.model_name = 'tt_mmpretrain_v3'
        self.psm = 'tiktok.data.mmpretrain_embedding_v3'
        self.laplace = Laplace('{}?cluster=default&idc={}'.format(self.psm, self.tce_dc), timeout=2)
        self.image_num = 8
        self.retry_time = 1
    def process_input(self, inputs):
        data = {}
        data['title'] = [inputs['title']]
        data['stickers'] = [inputs['stickers']]
        data['challenge'] = [inputs['hash_tags']]
        frames = inputs['frames_uniform']
        for i in range(self.image_num):
            data['frame%s'%i] = [frames[i]]
        return data
    def parse_rsp(self, inputs, rsp):
        object_id = inputs['gid']
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
    def inference(self, object_id):
        aweme_type, title, stickers, nickname, music, frames, ocr, asr, bio_description, frames_uniform, followers_count, create_idc, hash_tags, a_duration, create_time, region, text_language, type_label, poi_name, clear_music_name = \
            self.item_info.request(int(object_id), frame_limit=8, log_id='', metric_client=self.metric_client, need_pm=True)
        self.gid_dc = create_idc
        inputs = {
            'title':title, 'stickers':stickers, 'nickname':nickname, 'music':music, 'asr':asr, 'ocr':ocr,
            'author_description':bio_description, 'aweme_type':aweme_type, 'gid':object_id, 'create_idc':create_idc,
            'followers_count':followers_count, 'hash_tags': hash_tags, 'a_duration': a_duration, 'create_time':create_time,
            'region':region, 'text_language':text_language, 'type_label':type_label, 'poi_name':poi_name, 'clear_music_name':clear_music_name
        }
        if self.debug:
            print(inputs)
            print(f"frames: {len(frames)}")
            print(f"frames_uniform: {len(frames_uniform)}")
        inputs['frames'] = frames
        inputs['frames_uniform'] = frames_uniform
        data = {}
        try:
            data = self.process_input(inputs)
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
                raw_feature = self.parse_rsp(inputs, rsp)
                self.metric_client.emit_counter('model.inference.success', 1, tags={"model_name": self.model_name, 'gid_dc': self.gid_dc})
                return raw_feature
            except Exception as e:
                bytedlogger.logging.error("[DEBUG_LOG][MODEL_INFERENCE]{} , gid: {}, retry: {}, inference error: {}".format(self.model_name, inputs['gid'], _retry_i, e))
                if _retry_i == self.retry_time - 1:
                    self.metric_client.emit_counter('model.inference.failed', 1, tags={"model_name": self.model_name, 'gid_dc': self.gid_dc, 'is_rsp': str(is_rsp)})
        return {}