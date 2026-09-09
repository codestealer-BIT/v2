import os
import torch
import jsonlines
import pickle
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent import futures
from tqdm import tqdm
from IPython import embed
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Example: read data_path from command line")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory or file')
    parser.add_argument('--output_csv_path', type=str, help='Path to the output CSV file for checking cases result')
    args = parser.parse_args()
    return args


def load_music_vectors_from_file(file_path):
    if file_path.endswith('.pt'):
        music_tensor_list = torch.load(file_path)
        ids = [int(item['music_id']) for item in music_tensor_list]
        vecs = [item['music_ue_vector'] for item in music_tensor_list]
        ids = torch.LongTensor(ids)
        vecs = torch.stack(vecs, dim=0)
        return ids, vecs
    elif file_path.endswith('.csv'):
        ids = []
        vecs = []
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                clip_id_str, vec_str = parts[0].strip(), parts[1].strip()
                try:
                    cid = int(clip_id_str)
                except Exception:
                    continue
                if vec_str.startswith('[') and vec_str.endswith(']'):
                    vec_str = vec_str[1:-1]
                arr = np.fromstring(vec_str, sep=',', dtype=np.float32)
                if arr.size == 0:
                    arr = np.fromstring(vec_str, sep=' ', dtype=np.float32)
                if arr.size == 0:
                    continue
                ids.append(cid)
                vecs.append(torch.from_numpy(arr))
        if len(vecs) == 0:
            raise RuntimeError(f"No vectors parsed from {file_path}")
        ids = torch.LongTensor(ids)
        vecs = torch.stack(vecs, dim=0)
        return ids, vecs
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def calculate_similarity(all_video_embeds, music_tensor_file, task_id, topk=1000):
    gpu_id = task_id % torch.cuda.device_count()
    all_music_ids, all_music_ue_vector = load_music_vectors_from_file(music_tensor_file)
    all_music_ue_vector = all_music_ue_vector.to(all_video_embeds.dtype)
    # print(all_music_ids.shape, all_music_ue_vector.shape)
    
    all_video_embeds = all_video_embeds.to(f'cuda:{gpu_id}')
    all_music_ue_vector = all_music_ue_vector.to(f'cuda:{gpu_id}')

    all_video_embeds = F.normalize(all_video_embeds, dim=1)
    all_music_ue_vector = F.normalize(all_music_ue_vector, dim=1)

    sim = torch.mm(all_video_embeds, all_music_ue_vector.t())
    print("sim.shape",sim.shape)
    topk_sim, topk_idx = torch.topk(sim, topk, dim=1)
    topk_sim = topk_sim.cpu()   # [num_video, 1000]
    topk_idx = topk_idx.cpu()   # [num_video, 1000]
    del all_video_embeds
    del all_music_ue_vector

    topk_music_ids = []
    for index in range(len(topk_idx)):
        topk_music_ids.append(all_music_ids[topk_idx[index]])

    topk_music_ids = torch.stack(topk_music_ids, dim=0)
    return topk_sim, topk_music_ids

@torch.no_grad()
def find_topk(all_video_embeds, music_library_dir, topk_num = 50):
    """
    given video embeddings, find the topk music ids with highest similarity
    """
    
    print(f"The video embedding shape: {all_video_embeds.shape}")
        
    if not os.path.exists(music_library_dir) or not os.path.isdir(music_library_dir):
        raise ValueError(f"music_library_dir {music_library_dir} is not a directory or does not exist")

    music_vec_files = [os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.pt') or file.endswith('.csv') or file.startswith('part')]
    print(f"Loading music vectors from {music_vec_files}, there are {len(music_vec_files)} pt files")
    all_music_sim = []
    all_topk_music_ids = []
    task_queue = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        for index in range(len(music_vec_files)):
            task_queue.append(executor.submit(calculate_similarity, all_video_embeds, music_vec_files[index], index, topk_num))
        
        for future in tqdm(futures.as_completed(task_queue)):
            topk_sim, topk_music_ids = future.result()
            all_music_sim.append(topk_sim)
            all_topk_music_ids.append(topk_music_ids)

    all_music_sim = torch.cat(all_music_sim, dim=1)
    all_topk_music_ids = torch.cat(all_topk_music_ids, dim=1)

    print(all_music_sim.shape)
    

    final_topk_sim, final_topk_idx = torch.topk(all_music_sim, topk_num, dim=1)
    final_topk_music_ids = all_topk_music_ids[torch.arange(all_topk_music_ids.size(0)).unsqueeze(1), final_topk_idx]

    del all_music_sim
    del all_topk_music_ids

    return final_topk_sim, final_topk_music_ids

@torch.no_grad()
def retrieve_from_music_library_cases(music_library_dir, output_csv_path='LT/recall_results.csv', topk_num=50):
    # demo_image_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/MMpretrain'
    demo_image_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/lutong/'
    all_video_embeds = []

    image_path_list = [
        'demo_cases/test_flight.jpeg',
        'demo_cases/test_flight_3.png',
        'demo_cases/test_food_3.jpeg',
        'demo_cases/test_food_4.png',
        'demo_cases/cat2.jpg',
        'demo_cases/scene.jpg', 
        'demo_cases/makeup.png',
        'demo_cases/climb.png',
        'demo_cases/food3.png',
        'demo_cases/animal_4.jpg',
        'demo_cases/bay_scene.jpg',
        'demo_cases/children.png',
        'demo_cases/plane2.webp',
        'demo_cases/moon_tea.webp',
        'demo_cases/hotel.jpg',
        'demo_cases/wedding.jpeg',
        'demo_cases/kids.jpg',
        'demo_cases/my_cat.jpg',
        'demo_cases/argument.png',
        'demo_cases/kpop.jpg',
        'demo_cases/Thanksgiving.png',
        'demo_cases/sad.jpg',
        'demo_cases/Fish_and_chips.jpg',
        'demo_cases/car.jpeg',
        'demo_cases/celeberity.jpg',
        'demo_cases/workout_man.jpg',
        'demo_cases/zumba-dance.jpg',
        'demo_cases/diy_cup.png',
        'demo_cases/cars_vintage.jpg',
        'demo_cases/Xiaomi_SU7.jpg',
        'demo_cases/football.jpg',
        'demo_cases/selfie_mid_woman.png',
        'demo_cases/selfie_caucasion.png',
        'demo_cases/aunties_fashion.png',
        'demo_cases/outfit_mid_man.jpg',
        'demo_cases/tiananmen.jpg',
        'demo_cases/winter.jpg',
        'demo_cases/fishing.jpg',
        'demo_cases/sushi.jpg',
        'demo_cases/christmas.jpg',
        # rescaled
        'demo_cases/fishing_rescaled.jpeg',
        'demo_cases/winter_rescaled.jpeg',
        'demo_cases/cat2_rescaled.jpeg',
    ]
    # image_path_list = [
    #     'demo_cases/fishing_high.jpeg',
    #     'demo_cases/fishing_low.jpeg',
    #     'demo_cases/cute_high.jpeg',
    #     'demo_cases/cute_low.jpeg',
    #     'demo_cases/paint_high.jpeg',
    #     'demo_cases/paint_low.jpeg',
    # ]
    # image_path_list = [
    #     # "demo_cases/junqi_food.webp",
    #     "demo_cases/online1_text.webp",
    #     "demo_cases/online2_sandwich.webp"
    # ]

    all_video_embeds = []
    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]

        with open(os.path.join(demo_image_dir, f'demo_cases/{image_name}_vector.pkl'), 'rb') as fr:
            all_video_embeds.append(pickle.load(fr))  # [bs, 128]
    
    all_video_embeds = torch.cat(all_video_embeds, dim=0)
    final_topk_sim, final_topk_music_ids = find_topk(all_video_embeds, music_library_dir, topk_num=topk_num)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['image_name', 'recall_results', 'similarity_scores'])
        for idx, image_path in enumerate(image_path_list):
            image_name = os.path.basename(image_path).split('.')[0]
            final_topk_music_id = [str(x) for x in final_topk_music_ids[idx].tolist()]
            final_topk_sim_scores = [f'{x:.5f}' for x in final_topk_sim[idx].tolist()]
            writer.writerow([image_name, ' '.join(final_topk_music_id), ' '.join(final_topk_sim_scores)])
    print(f"Results saved to: {output_csv_path}")
    


@torch.no_grad()
def retrieve_from_music_library_eval_top20(video_emb_dir, output_jsonl_path, music_library_dir, total_rank = 16, topk_num=20):
    for rank_num in range(total_rank):
        eval_top20_each_rank(video_emb_dir, output_jsonl_path, rank_num, music_library_dir, topk_num=topk_num)


@torch.no_grad()
def eval_top20_each_rank(video_emb_dir, output_jsonl_path, rank_num, music_library_dir, topk_num=20):
    all_cache_files = []
    subdir = os.path.join(video_emb_dir, f"rank_{rank_num}")
    print(f"Loading cache files from {subdir}")
    cache_files = [os.path.join(subdir, file) for file in os.listdir(subdir)]
    all_cache_files.extend(cache_files)
    print(f"Total cache files found: {len(all_cache_files)} ")

    if len(all_cache_files) == 0:
        raise RuntimeError(f"No cache files found under {video_emb_dir}")

    all_video_embeds = []
    all_video_item_ids = []

    for cache_file in tqdm(all_cache_files):
        data = torch.load(cache_file, map_location='cpu')
        video_embed = data['video_embed']
        video_ids = data['video_ids']
        all_video_embeds.append(video_embed)
        all_video_item_ids.extend(video_ids)

    all_video_embeds = torch.cat(all_video_embeds, dim=0)

    if len(all_video_item_ids) != all_video_embeds.size(0):
        print(f"Mismatched lengths: {len(all_video_item_ids)} ids vs {all_video_embeds.size(0)} embeds")

    final_topk_sim, final_topk_music_ids = find_topk(all_video_embeds, music_library_dir, topk_num=topk_num)

    output_data = [
        {
            "item_id": all_video_item_ids[i],
            "topk_music_ids": final_topk_music_ids[i].tolist(),
        }
        for i in range(len(all_video_item_ids))
    ]

    output_part_name = os.path.basename(output_jsonl_path).split('.')[0] + f"_rank_{rank_num}.jsonl"
    output_jsonl_path = os.path.join(os.path.dirname(output_jsonl_path), output_part_name)
    with jsonlines.open(output_jsonl_path, 'w') as fw:
        fw.write_all(output_data)
    
    del all_video_embeds
    return output_jsonl_path

@torch.no_grad()
def retrieve_evalset_from_music_library(data_path):
    all_video_embeds = []
    all_video_item_ids = []

    with open(os.path.join(data_path, 'evalset_1000_image_embs.pkl'), 'rb') as fr:   
        evalset_item_id_to_emb = pickle.load(fr)
        for item_id, video_embed in evalset_item_id_to_emb.items():
            all_video_embeds.append(video_embed)
            all_video_item_ids.append(item_id)
  
    all_video_embeds = torch.cat(all_video_embeds, dim=0)
    
    print(f"The video embedding shape: {all_video_embeds.shape}")
    # music_library_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/music_library_video_siglip2_baseline/baseline_1025_ckpt_94000'

    # music_vec_files = [os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.pt') or file.startswith('part')]
    # music_vec_files = [os.path.join(data_path, 'music_ue_vector.pt')]
    music_vec_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.pt')]

    print(f"Loading music vectors from {music_vec_files}")
    all_music_ids_list = []
    all_music_vec_list = []
    for file in music_vec_files:
        ids, vecs = load_music_vectors_from_file(file)
        all_music_ids_list.append(ids)
        all_music_vec_list.append(vecs)
    all_music_ids = torch.cat(all_music_ids_list, dim=0)
    all_music_ue_vector = torch.cat(all_music_vec_list, dim=0).to(all_video_embeds.dtype)
    print(f"The music embedding shape: {all_music_ue_vector.shape}")
    print(f"Total music ids: {len(all_music_ids)}")

    all_video_embeds = F.normalize(all_video_embeds, dim=1)
    all_music_ue_vector = F.normalize(all_music_ue_vector, dim=1)
    topk = len(all_music_ids)
    sim = torch.mm(all_video_embeds, all_music_ue_vector.t())
    print("sim.shape",sim.shape)
    topk_sim, topk_idx = torch.topk(sim, topk, dim=1)
    topk_sim = topk_sim.cpu()  
    topk_idx = topk_idx.cpu()
    print("topk_sim.shape", topk_sim.shape)
    print("topk_idx.shape", topk_sim.shape)

    output_file = os.path.join(data_path, 'video_to_1w_music_similarity.csv')
    with open(output_file, "w") as fw:
        for i, item_id in enumerate(all_video_item_ids):
            music_indices = topk_idx[i].tolist()
            sim_scores = topk_sim[i].tolist()
            music_ids = [str(all_music_ids[idx].item()) for idx in music_indices]
            sim_scores_str = [f"{score:.6f}" for score in sim_scores]

            line = f"{item_id}," \
                   f"{' '.join(music_ids)}," \
                   f"{' '.join(sim_scores_str)}\n"
            fw.write(line)

    print(f"✅ Done! Results saved to {output_file}")

def load_video_embeds_from_npz(npz_path):
    data = np.load(npz_path)
    item_ids = data["item_ids"]
    embeds = data["embeddings"]
    if isinstance(item_ids, np.ndarray):
        item_ids = item_ids.tolist()
    return item_ids, torch.from_numpy(embeds)

@torch.no_grad()
def retrieve_from_video_embed_npz(video_embed_npz, music_library_dir, output_csv_path, topk_num=50):
    item_ids, all_video_embeds = load_video_embeds_from_npz(video_embed_npz)
    if not isinstance(all_video_embeds, torch.Tensor):
        all_video_embeds = torch.from_numpy(all_video_embeds)
    print(f"The video embedding shape: {all_video_embeds.shape}, topk_num: {topk_num}")
    final_topk_sim, final_topk_music_ids = find_topk(all_video_embeds, music_library_dir, topk_num=topk_num)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['item_id', 'topk_music_ids', 'similarity_scores'])
        for idx, item_id in enumerate(item_ids):
            final_topk_music_id = [str(x) for x in final_topk_music_ids[idx].tolist()]
            final_topk_sim_scores = [f'{x:.5f}' for x in final_topk_sim[idx].tolist()]
            writer.writerow([item_id, ' '.join(final_topk_music_id), ' '.join(final_topk_sim_scores)])
    print(f"Top {topk_num} results saved to: {output_csv_path}")


if __name__ == "__main__":
    args = parse_args()

    # # 1.top20 benchmark
    # video_emb_dir = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/evals_video_emb/bm_0.1_lowq_7_real_residual_add_user_info_100000"
    # output_jsonl_path = '/mnt/bn/jiny-ttls-i18n-fr1q/lutong/evals_top20/bm_0.1_lowq_7_real_residual_add_user_info_100000.jsonl'
    # music_library_dir = os.path.join(args.data_path, "650w_music")
    # retrieve_from_music_library_eval_top20(video_emb_dir, music_library_dir, output_jsonl_path)

    # 2.check case
    # music_library_dir = os.path.join(args.data_path, "650w_music")
    # music_library_dir = os.path.join(args.data_path, "8000w_deploy")
    # music_library_dir = os.path.join(args.data_path, "200w_deploy")
    # # music_library_dir = os.path.join(args.data_path, "135w_deploy")
    # retrieve_from_music_library_cases(music_library_dir, args.output_csv_path, topk_num=300)

    # 3.auc benchmark
    data_path = os.path.join(args.data_path, "evalset_1w_data/")
    retrieve_evalset_from_music_library(data_path)

    # # 4. find top music given video embs
    # video_embed_path =  os.path.join(args.data_path, "top5_itemid_to_embed.npz")
    # music_library_dir = os.path.join(args.data_path, "135w_deploy")
    # retrieve_from_video_embed_npz( video_embed_path, music_library_dir, args.output_csv_path, topk_num=300 )
