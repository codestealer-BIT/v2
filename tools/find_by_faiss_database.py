import argparse
import torch
import faiss
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

import sys
sys.path.append('/opt/tiger/MMPretrain')

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from video_clip import VideoSigLIP2ShortLongBLIP
from dataset import TikTokRetrievalDataset

def load_index_and_map(faiss_path = None):
    index_path= faiss_path + "/vectors.index"
    cpu_index = faiss.read_index(index_path)
    return cpu_index

# -----------------------------
# Step 6: Batched search
# -----------------------------
def batched_search(index, queries, batch_size=1024, k=5):
    results_ids = []
    results_scores = []
    num_queries = queries.shape[0]
    for i in range(0, num_queries, batch_size):
        batch_queries = queries[i:i+batch_size]
        faiss.normalize_L2(batch_queries)
        scores, retrieved_ids = index.search(batch_queries, k)
        results_ids.append(retrieved_ids)
        results_scores.append(scores)
    return np.vstack(results_ids), np.vstack(results_scores)

# -----------------------------
# Step 7: Example Usage
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='find embeddings by faiss database', add_help=False)
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--bert_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased", type=str, help='the config path of the bert tokenizer to be used')
    parser.add_argument('--blip_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/blip2", type=str, help='the config path of the blip tokenizer to be used')
    parser.add_argument('--video_root',default="/mnt/bn/jiny-ttls-i18n-fr1q/tiktok_30k_recaption_long_fusion_dedup.jsonl", type=str, help='video paths')
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--faiss_path',default=None, type=str, help='where to store the faiss vectors and id maps')
    parser.add_argument('--pretrained_path',default=None, type=str, help='the pretrained siglip2 models to be used')
    parser.add_argument('--output_path',default=None, type=str, help='validation results export path')
    parser.add_argument('--batch_size',default=1, type=int, help='evaluation batch size')
    args = parser.parse_args()

    model = VideoSigLIP2ShortLongBLIP(model_path=args.model_path, blip_path = args.blip_path,interpolate=6)
    
    model.load_state_dict(torch.load(args.pretrained_path + '/pytorch_model.bin',map_location='cpu'))
    
    # Load index + ID map later
    cpu_index = load_index_and_map(args.faiss_path)
    
    # # Move index to GPU(s) if available
    # gpu_index = move_index_to_gpu(cpu_index)
    
    # 30k queries
    
    test_dataset = TikTokRetrievalDataset(video_root=args.video_root, 
                max_short_len=64,
                max_text_len=284,
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

    count = 0
    model.eval()
    device = "cuda"
    model.to(device)
    for(idx, data) in enumerate(test_loader):
        count += 1

        item_ids = data['item_ids']

        pixel_values = data['pixel_values'].to(device)
        pixel_attention_mask = data['pixel_attention_mask'].to(device)
        spatial_shapes = data['spatial_shapes'].to(device)
        
        caption_input_ids = data['fusion_input_ids'].to(device)#caption的segmentid是zero可以直接传 

        title_input_ids = data['title_input_ids'].to(device)
        
        batch_size, concat, seq_len = title_input_ids.shape
        segment_ids = torch.zeros_like(title_input_ids)# 0: title
        segment_ids = segment_ids.view(batch_size, concat* seq_len).contiguous()
        segment_ids[:, 64:128] = 1    # 1: sticker
        segment_ids[:, 128:192] = 2   # 2: ocr
        segment_ids[:, 192:256] = 3   # 3: asr

        title_input_ids = title_input_ids.view(batch_size * concat, seq_len).contiguous()
        
        with torch.no_grad():
            # caption
            caption_embeds, caption_pooled = model.encode_text(input_ids=caption_input_ids)
            # title
            title_embeds, title_pooled = model.encode_text(input_ids = title_input_ids)

            title_embeds = title_embeds.view(batch_size, concat, seq_len, -1).contiguous()

            title_embeds = title_embeds.view(batch_size, concat * seq_len, -1).contiguous()
            #video
            video_embeds, video_pooled = model.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
            #fusion
            fused_embeds, fused_pooled = model.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                                vision_attn_mask=None, title_attn_mask = None, segment_ids = segment_ids)
    
        k = 5
        query_emb = fused_pooled.detach().cpu().numpy().astype("float32")
        retrieved_int_ids, scores = batched_search(cpu_index, query_emb, batch_size=1, k=k)

    
        print(f"Example retrieved IDs for {count}th query {data['item_ids'][0]}: ", retrieved_int_ids[0])
        print("Similarity scores:", scores[0])

        if(count >= 10):
            break
