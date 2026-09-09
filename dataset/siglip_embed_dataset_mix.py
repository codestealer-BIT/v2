      
import torch
from torch.utils.data import IterableDataset
import random

class VideoDatasetMixture(IterableDataset):
    """
    将多个IterableDataset按指定概率组合成一个新的IterableDataset
    
    Args:
        datasets: 一个包含多个IterableDataset的列表
        weights: 每个数据集对应的采样概率权重列表
    """
    def __init__(self, internvid_dataset, hdfs_dataset_list, hdfs_weight):
        
        self.internvid_dataset = internvid_dataset
        self.hdfs_dataset = hdfs_dataset_list
        self.hdfs_weight = hdfs_weight
        assert len(hdfs_dataset_list) == len(hdfs_weight), "数据集数量和权重数量必须相等"
        # 确保权重总和为1（或接近1，考虑浮点数误差）
        assert abs(sum(hdfs_weight) - 1.0) < 1e-6, "权重总和必须为1.0"
        
        self.internvid_iter = iter(self.internvid_dataset)
        self.hdfs_iter = [iter(ds) for ds in hdfs_dataset_list]
        
    def __iter__(self):
        while True:
            data = {}
            try:
                data.update(next(self.internvid_iter))
            except StopIteration:
                self.internvid_iter = iter(self.internvid_dataset)
                data.update(next(self.internvid_iter))
            
            try:
                # 根据权重随机选择一个数据集
                idx = random.choices(range(len(self.hdfs_dataset)), weights=self.hdfs_weight)[0]
                # 从选中的数据集获取下一个样本
                data.update(next(self.hdfs_iter[idx]))
            except StopIteration:
                # 当某个数据集耗尽时，重新创建其迭代器
                self.hdfs_iter[idx] = iter(self.hdfs_dataset[idx])
                # 继续从该数据集获取样本
                data.update(next(self.hdfs_iter[idx]))
            
            yield data

# 使用示例
if __name__ == "__main__":
    from siglip_embed_dataset_internvid import VideoDatasetInternVid, create_mm_embed_dataloader
    from siglip_embed_dataset_short_long import VideoHDFSDatasetShortLong
    from siglip_embed_dataset_large import VideoHDFSDatasetLarge

    data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/vd-foundation___InternVid-10M-FLT/intervid_anno/InternVid-10M-FLT-INFO_with_path_long_caption.jsonl'
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    blip_tokenizer_dir = "/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
    
    internvid_dataset = VideoDatasetInternVid(stage=2,data_path=data_path, shuffle=True, repeat=True, image_query_num = 10, buffer_size=256, max_frame_len=5, max_short_len = 10, max_text_len = 64,tokenizer_dir=tokenizer_dir,blip_tokenizer_dir = blip_tokenizer_dir)

    data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model/finetune_v1_data_plus_files_filtered.json'
    dict_path = '/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model'
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    blip_tokenizer_dir = "/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
    
    hdfs_dataset = VideoHDFSDatasetShortLong(stage=2,data_path=data_path, dict_path = dict_path, shuffle=True, repeat=True, image_query_num = 10, buffer_size=256, max_frame_len=5, max_short_len = 20, max_text_len = 30,tokenizer_dir=tokenizer_dir,blip_tokenizer_dir = blip_tokenizer_dir)

    data_path = '/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model/pretrain_data_0911_100m_data_files_filtered.json'
    dict_path = '/mnt/bn/jiny-ttls-i18n-fr1q/final_tiktok_data_for_small_model'
    tokenizer_dir = '/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base'
    blip_tokenizer_dir = "/mnt/bn/yexiaoyu-test/models/huggingface/blip2"
    
    large_dataset = VideoHDFSDatasetLarge(stage=2,data_path=data_path, dict_path = dict_path, shuffle=True, repeat=True, image_query_num = 10, buffer_size=256, max_frame_len=5, max_short_len = 30, max_text_len = 30,tokenizer_dir=tokenizer_dir,blip_tokenizer_dir = blip_tokenizer_dir)

    # make mixture
    mix_dataset = VideoDatasetMixture(internvid_dataset=internvid_dataset, hdfs_dataset_list=[large_dataset, hdfs_dataset], hdfs_weight=[0.75, 0.25])

    loader = create_mm_embed_dataloader(
        mix_dataset,
        batch_size=1,
        num_workers=16,
        cuda_prefetch=False,
    )
    count = 0
    for data in loader:
        print("data['internvid_short_input_ids'].shape: ", data['internvid_short_input_ids'].shape)
        print("data['title_input_ids'].shape: ", data['title_input_ids'].shape)
        break




