import os
import torch
import jsonlines
import pickle
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures
from tqdm import tqdm
from IPython import embed
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Example: read data_path from command line")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory or file')
    args = parser.parse_args()
    return args

def calculate_similarity(all_video_embeds, music_tensor_file, task_id, topk=1000, join_set=None):
    gpu_id = task_id % 2
    music_tensor_list = torch.load(music_tensor_file)
    all_music_ids = []
    all_music_ue_vector = []
    for item in music_tensor_list:
        if join_set is None or item['music_id'] in join_set:
            all_music_ids.append(item['music_id'])
            all_music_ue_vector.append(item['music_ue_vector'])

    all_music_ids = torch.LongTensor(all_music_ids)
    all_music_ue_vector = torch.stack(all_music_ue_vector).to(all_video_embeds.dtype)
    # print(all_music_ids.shape, all_music_ue_vector.shape)
    print("all_music_ue_vector.shape",all_music_ue_vector.shape)
    print("all_music_ids.shape",all_music_ue_vector.shape)

    # all_video_embeds = all_video_embeds.to(f'cuda:{gpu_id}')
    # all_music_ue_vector = all_music_ue_vector.to(f'cuda:{gpu_id}')

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
def retrieve_from_music_library(data_path):
    # demo_image_dir = '/mnt/bn/search-ad-creative-zhijie/wangxiuqi.0601/mm/MMpretrain'
    demo_image_dir = data_path
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
    ]

    for image_path in image_path_list:
        image_name = os.path.split(image_path)[-1].split('.')[0]

        with open(os.path.join(demo_image_dir, f'demo_cases/{image_name}_ko-KR_vector.pkl'), 'rb') as fr:
            all_video_embeds.append(pickle.load(fr))  # [bs, 128]
    
    all_video_embeds = torch.cat(all_video_embeds, dim=0)
    
    print(f"The video embedding shape: {all_video_embeds.shape}")
    
    topk_num = 50
    music_library_dir = data_path

    music_vec_files = ["/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_residual_add_user_info/checkpoint-60000/evalset_1w_data/music_ue_vector.pt"] 
    # music_vec_files = ["/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/checkpoints/video_siglip2_for_music_baseline_retrain_positive_filter_discriminator_0.01_lowq_8//checkpoint-20000/evalset_1w_data/music_ue_vector.pt"]
    # music_vec_files =[os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.pt') or file.startswith('part')]
    print(f"Loading music vectors from {music_vec_files}")
    all_music_sim = []
    all_topk_music_ids = []
    task_queue = []

    with ProcessPoolExecutor(max_workers=16) as executor:
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

    # print("recall score: ", final_topk_sim)

    for idx, image_path in enumerate(image_path_list):
        final_topk_music_id = final_topk_music_ids[idx].tolist()
        final_topk_music_id = [str(_) for _ in final_topk_music_id]
        print(f"{image_path}," + ' '.join(final_topk_music_id))

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
    music_library_dir = '/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/checkpoints/video_siglip2_discriminator_bm_0.1_lowq_7_residual_add_user_info/checkpoint-100000/evalset_whole_data'

    music_vec_files = [os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.pt') or file.startswith('part')]
    # music_vec_files = [os.path.join(data_path, 'music_ue_vector.pt')]
    print(f"Loading music vectors from {music_vec_files}")
    all_music_ids = []
    all_music_ue_vector = []
    for index in range(len(music_vec_files)):
        music_tensor_list = torch.load(music_vec_files[index])
        for item in music_tensor_list:
            # if evalset_clip_id_join_set is None or item['music_id'] in evalset_clip_id_join_set:
            all_music_ids.append(item['music_id'])
            all_music_ue_vector.append(item['music_ue_vector'])

    all_music_ids = torch.LongTensor(all_music_ids)
    all_music_ue_vector = torch.stack(all_music_ue_vector).to(all_video_embeds.dtype)
    print(f"The music embedding shape: {all_music_ue_vector.shape}")
    print(f"Total music ids: {len(all_music_ids)}")

    all_video_embeds = F.normalize(all_video_embeds, dim=1)
    all_music_ue_vector = F.normalize(all_music_ue_vector, dim=1)
    topk =  20 #len(all_music_ids)
    sim = torch.mm(all_video_embeds, all_music_ue_vector.t())
    print("sim.shape",sim.shape)
    topk_sim, topk_idx = torch.topk(sim, topk, dim=1)
    topk_sim = topk_sim.cpu()  
    topk_idx = topk_idx.cpu()
    print("topk_sim.shape", topk_sim.shape)
    print("topk_idx.shape", topk_sim.shape)

    output_file = os.path.join(data_path, 'video_to_600w_music_similarity.csv')
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
    # topk_music_ids = []
    # for index in range(len(topk_idx)):
    #     topk_music_ids.append(all_music_ids[topk_idx[index]])

    # topk_music_ids = torch.stack(topk_music_ids, dim=0)

    # for idx in range(len(all_video_item_ids)):
    #     final_topk_music_id = final_topk_music_ids[idx].tolist()
    #     final_topk_music_id = [str(_) for _ in final_topk_music_id]
    #     print(str(all_video_item_ids[idx]) + "," + ' '.join(final_topk_music_id[:20]))


        #     task_queue.append(executor.submit(calculate_similarity, all_video_embeds, music_vec_files[index], index, topk_num, join_set=evalset_clip_id_join_set))
        
        # for future in tqdm(futures.as_completed(task_queue)):
        #     topk_sim, topk_music_ids = future.result()
        #     all_music_sim.append(topk_sim)
        #     all_topk_music_ids.append(topk_music_ids)

    # all_music_sim = torch.cat(all_music_sim, dim=1)
    # all_topk_music_ids = torch.cat(all_topk_music_ids, dim=1)


    # final_topk_sim, final_topk_idx = torch.topk(all_music_sim, topk_num, dim=1)
    # final_topk_music_ids = all_topk_music_ids[torch.arange(all_topk_music_ids.size(0)).unsqueeze(1), final_topk_idx]


    # for idx in range(len(all_video_item_ids)):
    #     final_topk_music_id = final_topk_music_ids[idx].tolist()
    #     final_topk_music_id = [str(_) for _ in final_topk_music_id]
    #     print(str(all_video_item_ids[idx]) + "," + ' '.join(final_topk_music_id[:20]))


if __name__ == "__main__":
    args = parse_args()
    data_path = os.path.join(args.data_path, "evalset_1w_data/")
    retrieve_evalset_from_music_library(data_path)
    # retrieve_from_music_library(data_path)
