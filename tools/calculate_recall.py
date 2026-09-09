import os
import torch
import jsonlines
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures
from tqdm import tqdm
from IPython import embed


def load_one_file(filepath):
    data = torch.load(filepath, map_location='cpu')
    video_embed = data['video_embed']
    music_embed = data['music_embed']
    video_embed = F.normalize(video_embed, dim=1)
    music_embed = F.normalize(music_embed, dim=1)
    video_ids = data['video_ids']
    music_ids = data['music_ids']
    return video_embed, music_embed, video_ids, music_ids



def load_embeds_in_cpu(cache_dir):
    all_cache_files = []
    for rank in range(0, 32):
        subdir = os.path.join(cache_dir, f"rank_{rank}")
        if not os.path.exists(subdir):
            continue
        cache_files = [os.path.join(subdir, file) for file in os.listdir(subdir)]
        all_cache_files.extend(cache_files)
    
    all_video_embeds = []
    all_music_embeds = []
    all_video_item_ids = []
    all_music_item_ids = []
    task_queue = []

    with ProcessPoolExecutor(max_workers=128) as executor:
        for filepath in all_cache_files:
            task_queue.append(executor.submit(load_one_file, filepath))

        for future in tqdm(futures.as_completed(task_queue)):
            video_embed, music_embed, video_ids, music_ids = future.result()
            all_video_embeds.append(video_embed)
            all_music_embeds.append(music_embed)
            all_video_item_ids.extend(video_ids)
            all_music_item_ids.extend(music_ids)

    print(len(all_video_item_ids), len(set(all_video_item_ids)))
    print(len(all_music_item_ids), len(set(all_music_item_ids)))
    all_video_embeds = torch.cat(all_video_embeds, dim=0)
    all_music_embeds = torch.cat(all_music_embeds, dim=0)

    video_id_to_music_id = {all_video_item_ids[i]:all_music_item_ids[i] for i in range(len(all_video_item_ids))}

    return all_video_embeds, all_music_embeds, all_video_item_ids, video_id_to_music_id, all_music_item_ids


def deduplicate(all_music_embeds, all_music_item_ids):
    music_id_set = {}
    for i in range(len(all_music_item_ids)):
        music_id = all_music_item_ids[i]
        if music_id not in music_id_set:
            music_id_set[music_id] = [i]
        else:
            music_id_set[music_id].append(i)

    filtered_music_ids = []
    filtered_music_embeds = []
    for music_id in music_id_set:
        index = music_id_set[music_id][0]
        filtered_music_ids.append(music_id)
        filtered_music_embeds.append(all_music_embeds[index:index+1])

    filtered_music_embeds = torch.cat(filtered_music_embeds, dim=0)

    return filtered_music_embeds, filtered_music_ids


def check_top_recall_hit(topk_idx, music_item_ids, video_ids, video_id_to_music_id):
    recall_dict = {
        50 : 0,
        100 : 0,
        200 : 0,
        1000 : 0,
    }
    for topk in [50, 100, 200, 1000]:
        cur_topk_idx = topk_idx[:, :topk]
        for vidx in range(len(video_ids)):
            topk_music_ids =set([music_item_ids[idx] for idx in cur_topk_idx[vidx]])
            video_id = video_ids[vidx]
            if video_id_to_music_id[video_id] in topk_music_ids:
                recall_dict[topk] += 1

    return recall_dict


@torch.no_grad()
def retrieve_topk(cache_dir, video_embeds, all_video_item_ids, video_id_to_music_id, music_embeds, music_item_ids):
    video_embeds = video_embeds.cuda()
    music_embeds = music_embeds.cuda().to(torch.bfloat16)
    batch_size = 10240

    recall_dict = {
        50 : 0,
        100 : 0,
        200 : 0,
        1000 : 0,
    }
    sim_list = []

    for i in tqdm(range(0, len(video_embeds), batch_size)):
        video_embed = video_embeds[i:i+batch_size]
        batch_video_item_ids = all_video_item_ids[i:i+batch_size]
        sim = torch.mm(video_embed, music_embeds.t())
        topk_sim, topk_idx = torch.topk(sim, 1000, dim=1)
        sim_list.append(topk_idx.cpu())

    topk_idx = torch.cat(sim_list, dim=0)
    print("topk_idx: ", topk_idx.shape)
    top20_dict = {}
    for i in range(len(topk_idx)):
        top20_dict[all_video_item_ids[i]] = [music_item_ids[idx] for idx in topk_idx[i, :20].tolist()]

    torch.save(top20_dict, os.path.join(cache_dir, "top20_dict.pt"))

    task_queue = []
    with ProcessPoolExecutor(max_workers=128) as executor:
        for vidx in range(0, len(topk_idx), batch_size):
            task_queue.append(executor.submit(check_top_recall_hit, topk_idx[vidx:vidx+batch_size], music_item_ids, all_video_item_ids[vidx:vidx+batch_size], video_id_to_music_id))
    
        for future in tqdm(futures.as_completed(task_queue)):
            cur_recall_dict = future.result()
            for key in recall_dict:
                recall_dict[key] += cur_recall_dict[key]
    
    print(recall_dict)
    video_num = len(set(all_video_item_ids))
    for key in recall_dict:
        recall_dict[key] = recall_dict[key] / video_num
        print(f"Recall@{key}: {recall_dict[key]}")


def retrieve_video_topk(cache_dir, video_embeds, all_video_item_ids):
    video_embeds = video_embeds.cuda()
    batch_size = 10240

    sim_list = []

    for i in tqdm(range(0, len(video_embeds), batch_size)):
        video_embed = video_embeds[i:i+batch_size]
        batch_video_item_ids = all_video_item_ids[i:i+batch_size]
        sim = torch.mm(video_embed, video_embeds.t())
        topk_sim, topk_idx = torch.topk(sim, 100, dim=1)
        sim_list.append(topk_idx.cpu())

    topk_idx = torch.cat(sim_list, dim=0)
    print("topk_idx: ", topk_idx.shape)

    top100_dict = {}
    for i in range(len(topk_idx)):
        top100_dict[all_video_item_ids[i]] = [all_video_item_ids[idx] for idx in topk_idx[i, :100].tolist()]

    torch.save(top100_dict, os.path.join(cache_dir, "video_i2i_top100_dict.pt"))


def evaluate_music_recall():
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/non_ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/origin_model'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-170000/non_ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-170000/ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-200000/ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline_dedup/checkpoint-200000/non_ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline_dedup/checkpoint-200000/non_ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_dedup_v3_large_queue/checkpoint-200000/ema'
    cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_text_clip/checkpoint-200000/ema'
    all_video_embeds, all_music_embeds, all_video_item_ids, video_id_to_music_id, all_music_item_ids = load_embeds_in_cpu(cache_dir)
    all_music_embeds, all_music_item_ids = deduplicate(all_music_embeds, all_music_item_ids)
    print(all_music_embeds.shape, len(all_music_item_ids), len(set(all_music_item_ids)))
    retrieve_topk(cache_dir, all_video_embeds, all_video_item_ids, video_id_to_music_id, all_music_embeds, all_music_item_ids)


def evaluate_video_i2i():
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline/checkpoint-200000/ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_baseline_dedup/checkpoint-100000/ema'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_dedup_v3_large_queue/checkpoint-200000/ema'
    cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_text_clip/checkpoint-200000/ema'
    all_video_embeds, all_music_embeds, all_video_item_ids, video_id_to_music_id, all_music_item_ids = load_embeds_in_cpu(cache_dir)
    all_video_embeds, all_video_item_ids = deduplicate(all_video_embeds, all_video_item_ids)
    print(all_video_embeds.shape, len(all_video_item_ids), len(set(all_video_item_ids)))

    retrieve_video_topk(cache_dir, all_video_embeds, all_video_item_ids)


def calculate_similarity(all_video_embeds, music_tensor_file, task_id, topk=1000):
    gpu_id = task_id % 8
    music_tensor_list = torch.load(music_tensor_file)
    all_music_ids = []
    all_music_ue_vector = []
    for item in music_tensor_list:
        all_music_ids.append(item['music_id'])
        all_music_ue_vector.append(item['music_ue_vector'])

    all_music_ids = torch.LongTensor(all_music_ids)
    all_music_ue_vector = torch.stack(all_music_ue_vector).to(all_video_embeds.dtype)
    # print(all_music_ids.shape, all_music_ue_vector.shape)
    
    all_video_embeds = all_video_embeds.to(f'cuda:{gpu_id}')
    all_music_ue_vector = all_music_ue_vector.to(f'cuda:{gpu_id}')

    all_video_embeds = F.normalize(all_video_embeds, dim=1)
    all_music_ue_vector = F.normalize(all_music_ue_vector, dim=1)

    sim = torch.mm(all_video_embeds, all_music_ue_vector.t())
    topk_sim, topk_idx = torch.topk(sim, topk, dim=1)
    topk_sim = topk_sim.cpu()   # [num_video, 1000]
    topk_idx = topk_idx.cpu()   # [num_video, 1000]

    topk_music_ids = []
    for index in range(len(topk_idx)):
        topk_music_ids.append(all_music_ids[topk_idx[index]])

    topk_music_ids = torch.stack(topk_music_ids, dim=0)
    return topk_sim, topk_music_ids


@torch.no_grad()
def retrieve_from_music_library():
    cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_clip_dedup_v3_large_queue/checkpoint-200000/ema_custom'
    # cache_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/results/video_music_text_clip/checkpoint-200000/ema_custom'
    all_cache_files = []

    for rank in range(0, 32):
        subdir = os.path.join(cache_dir, f"rank_{rank}")
        if not os.path.exists(subdir):
            continue
        cache_files = [os.path.join(subdir, file) for file in os.listdir(subdir)]
        all_cache_files.extend(cache_files)

    all_video_embeds = []
    all_video_item_ids = []

    for cache_file in tqdm(all_cache_files):
        data = torch.load(cache_file, map_location='cpu')
        video_embed = data['video_embed']
        video_ids = data['video_ids']
        all_video_embeds.append(video_embed)
        all_video_item_ids.extend(video_ids)

    all_video_embeds = torch.cat(all_video_embeds, dim=0)
    
    topk_num = 1000
    music_library_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_library'
    music_vec_files = [os.path.join(music_library_dir, file) for file in os.listdir(music_library_dir) if file.endswith('.snappy')]
    all_music_sim = []
    all_topk_music_ids = []
    task_queue = []

    with ProcessPoolExecutor(max_workers=64) as executor:
        for index in range(len(music_vec_files)):
            task_queue.append(executor.submit(calculate_similarity, all_video_embeds, music_vec_files[index], index, topk_num))
        
        for future in tqdm(futures.as_completed(task_queue)):
            topk_sim, topk_music_ids = future.result()
            all_music_sim.append(topk_sim)
            all_topk_music_ids.append(topk_music_ids)

    all_music_sim = torch.cat(all_music_sim, dim=1)
    all_topk_music_ids = torch.cat(all_topk_music_ids, dim=1)

    final_topk_sim, final_topk_idx = torch.topk(all_music_sim, topk_num, dim=1)
    final_topk_music_ids = all_topk_music_ids[torch.arange(all_topk_music_ids.size(0)).unsqueeze(1), final_topk_idx]

    results_dict = {}
    for index in range(len(final_topk_music_ids)):
        results_dict[all_video_item_ids[index]] = final_topk_music_ids[index].tolist()

    torch.save(results_dict, os.path.join(cache_dir, "music_library_top1000_dict.pt"))


if __name__ == "__main__":
    # evaluate_music_recall()
    evaluate_video_i2i()
    
    # save_the_music_library()
    # retrieve_from_music_library()
    