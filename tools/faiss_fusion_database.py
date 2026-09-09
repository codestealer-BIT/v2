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
from video_clip import VideoSigLIP2TemporalFusion
from dataset import TikTokRetrievalDataset,VideoHDFSDatasetLarge,create_mm_embed_dataloader

# -----------------------------
# Step 1: Encode dataset -> embeddings
# -----------------------------
def build_embeddings(dataloader, model, index_size = 10000, batch_size=256, device="cuda"):
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    all_ids_int = []
    id_map = {}  # FAISS int ID -> string item_id
    
    index_count = 0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader)):
            index_count += batch_size
            
            item_ids = data['item_ids']

            pixel_values = data['pixel_values'].to(device)
            pixel_attention_mask = data['pixel_attention_mask'].to(device)
            spatial_shapes = data['spatial_shapes'].to(device)
            
            title_input_ids = data['title_input_ids'].to(device)
            title_segment_ids = data['title_segment_ids'].to(device)
            title_attention_mask = data['title_attention_mask'].to(device)#在text encoder的时候不要传

            # title
            title_embeds, title_pooled = model.encode_text(input_ids = title_input_ids, segment_ids = title_segment_ids)
            #video
            video_embeds, video_pooled = model.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
            #fusion
            fused_embeds, fused_pooled = model.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                                vision_attn_mask=None, title_attn_mask = title_attention_mask)


            emb = fused_pooled.detach().cpu().numpy().astype("float32")
            all_embeddings.append(emb)
            
            # Assign integer IDs for FAISS
            start_idx = len(all_ids_int)
            batch_ids_int = np.arange(start_idx, start_idx + batch_size, dtype=np.int64)
            all_ids_int.extend(batch_ids_int)
            
            # Map FAISS int IDs → original string IDs
            for int_id, str_id in zip(batch_ids_int, item_ids):
                id_map[int_id] = str_id
            
            if(index_count >= index_size):
                break
    
    embeddings = np.vstack(all_embeddings)
    all_ids_int = np.array(all_ids_int, dtype=np.int64)
    return embeddings, all_ids_int, id_map

# -----------------------------
# Step 2: Build FAISS index
# -----------------------------
def build_faiss_index(embeddings, ids_int):
    d = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    cpu_index = faiss.IndexIDMap(faiss.IndexFlatIP(d))
    cpu_index.add_with_ids(embeddings, ids_int)
    return cpu_index

# -----------------------------
# Step 3: Move index to GPU(s)
# -----------------------------
def move_index_to_gpu(cpu_index):
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    return gpu_index

# -----------------------------
# Step 4: Save index + ID map
# -----------------------------
def save_index_and_map(cpu_index, id_map, faiss_path = None):
    index_path= faiss_path + "/vectors.index"
    map_path= faiss_path + "/faiss_id_to_item_id.pkl"
    faiss.write_index(cpu_index, index_path)
    with open(map_path, "wb") as f:
        pickle.dump(id_map, f)
    print(f"Saved FAISS index to {index_path} and ID map to {map_path}")

# -----------------------------
# Step 5: Load index + ID map
# -----------------------------
def load_index_and_map(faiss_path = None):
    index_path= faiss_path + "/vectors.index"
    map_path= faiss_path + "/faiss_id_to_item_id.pkl"
    cpu_index = faiss.read_index(index_path)
    with open(map_path, "rb") as f:
        id_map = pickle.load(f)
    return cpu_index, id_map

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

    parser = argparse.ArgumentParser(description='build video-to-video retrieval by faiss', add_help=False)
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--bert_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased", type=str, help='the config path of the bert tokenizer to be used')
    parser.add_argument('--data_root',nargs='+', default=[], help="data path")
    parser.add_argument('--filelist_cache', default=None, type=str, help='cache annotation')
    parser.add_argument('--dict_path', default='/mnt/bn/yexiaoyu-test/data/tt_captions_for_music.pkl', type=str, help="The caption path")
    parser.add_argument('--data_base_size', default=1e6, type=int, help='number of vectors in vector database')
    parser.add_argument('--query_root',default="/mnt/bn/yexiaoyu-test/data/tiktok_data_fusion_30k", type=str, help='query paths')
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--pretrained_path',default=None, type=str, help='the pretrained siglip2 models to be used')
    parser.add_argument('--faiss_path',default=None, type=str, help='where to store the faiss vectors and id maps')
    parser.add_argument('--build_faiss',action='store_true', help='whether to build faiss database')
    parser.add_argument('--batch_size',default=32, type=int, help='evaluation batch size')
    args = parser.parse_args()

    model = VideoSigLIP2TemporalFusion(model_path=args.model_path, tokenizer_path=args.bert_path)
    
    model.load_state_dict(torch.load(args.pretrained_path + '/pytorch_model.bin',map_location='cpu'))

    #build faiss database
    if(args.build_faiss):
        hdfs_dataset = VideoHDFSDatasetLarge(data_path=args.data_root, 
                                             dict_path=args.dict_path, 
                                             shuffle=True, repeat=True, 
                                             buffer_size=256, max_frame_len=8, 
                                             max_text_len=256,
                                             tokenizer_dir=args.model_path, 
                                             bert_tokenizer_dir = args.bert_path,
                                             filelist_cache=args.filelist_cache)
        # torch.cuda.set_device(0)

        loader = create_mm_embed_dataloader(
            hdfs_dataset,
            batch_size=args.batch_size,
            num_workers=16,
            cuda_prefetch=False,
        )
    
        # Encode embeddings
        embeddings, ids_int, id_map = build_embeddings(loader, model, batch_size=args.batch_size,
                                                       index_size=args.data_base_size,
                                                        device="cuda" )
        
        # Build FAISS CPU index
        cpu_index = build_faiss_index(embeddings, ids_int)
        
        # Save CPU index + ID map for later
        save_index_and_map(cpu_index, id_map, faiss_path=args.faiss_path)
    
    # Load index + ID map later
    cpu_index, id_map = load_index_and_map(args.faiss_path)
    
    # # Move index to GPU(s) if available
    # gpu_index = move_index_to_gpu(cpu_index)
    
    # 30k queries
    
    test_dataset = TikTokRetrievalDataset(video_root=args.query_root, 
                max_text_len=256,
                max_frame_len=args.max_frames,
                tokenizer_dir=args.model_path,
                bert_tokenizer_dir=args.bert_path,
                )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        collate_fn=default_collate,
    ) 
    count = 0
    model.to("cuda")
    for(idx, data) in enumerate(test_loader):
        count += 1

        item_ids = data['item_ids']

        pixel_values = data['pixel_values'].to("cuda")
        pixel_attention_mask = data['pixel_attention_mask'].to("cuda")
        spatial_shapes = data['spatial_shapes'].to("cuda")

        title_input_ids = data['title_input_ids'].to("cuda")
        title_segment_ids = data['title_segment_ids'].to("cuda")
        title_attention_mask = data['title_attention_mask'].to("cuda")#在text encoder的时候不要传
        
        with torch.no_grad():
            # title
            title_embeds, title_pooled = model.encode_text(input_ids = title_input_ids, segment_ids = title_segment_ids)
            #video
            video_embeds, video_pooled = model.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
            #fusion
            fused_embeds, fused_pooled = model.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                                vision_attn_mask=None, title_attn_mask = title_attention_mask)
    
        k = 5
        query_emb = fused_pooled.detach().cpu().numpy().astype("float32")
        retrieved_int_ids, scores = batched_search(cpu_index, query_emb, batch_size=1, k=k)
        
        # Map back to string item_ids
        retrieved_str_ids = [[id_map[i] for i in row] for row in retrieved_int_ids]
    
        print(f"Example retrieved IDs for {count}th query {data['item_ids'][0]}: ", retrieved_str_ids[0])
        print("Similarity scores:", scores[0])

        if(count >= 10):
            break
