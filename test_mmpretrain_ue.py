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
        self.base_info = thriftpy2.load(os.path.join(dirpath, "mmembed/idls/base.thrift"))
        self.emb_info = thriftpy2.load(os.path.join(dirpath, "mmembed/idls/embedding_flow.thrift"))
        self.ue_info = thriftpy2.load(os.path.join(dirpath, 'mmembed/idls/universal_embeddings.thrift'))
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
if __name__ == '__main__':
    ue = UeEmbedding('fc_tiktok_mmpretrainv3_pskv')
    res = ue.batch_get([7513955820778949895,7502306653103049992,7502316822214085893,7502310931590941957])
    print(f'size {len(res)}')
    print(res)