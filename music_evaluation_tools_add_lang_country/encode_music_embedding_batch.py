from encode_music_embedding import build_model, to_tensor, process_text, str2bool
from trainer_misc import init_distributed_mode
import torch
import numpy as np
import random
import json
import os
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process script', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model_dtype', default='bf16', type=str, help="The Model Dtype: bf16 or df16")
    parser.add_argument('--model_path', default='', type=str, help='The pre-trained weight path')
    parser.add_argument('--output_dir', type=str, default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory or file')
    parser.add_argument('--add_lyrics_ue', default=False, type=str2bool)
    parser.add_argument('--add_title_ue', default=False, type=str2bool)    
    parser.add_argument('--add_title_text', default=False, type=str2bool)
    parser.add_argument('--add_user_lang_and_country_method', default="residual", type=str)
    parser.add_argument('--user_language_code', default=1, type=int, help="The user language code for checking cases, e.g. 1:en")
    parser.add_argument('--user_country_code', default=1, type=int, help="The user country code for checking cases, e.g. 1:US")
    parser.add_argument('--embed_lang_caption', default=False, type=str2bool)
    parser.add_argument('--output_pt', default=False, type=str2bool, help="whether to output the pt file, it is for internal case check")
    return parser.parse_args()

def extract_music_embeds_main_func(model_path, ckpt_path, output_dir, args):
    init_distributed_mode(args)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda')
    rank = args.rank
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model, processor, tokenizer = build_model(model_path, ckpt_path, args)
    model = model.eval()
    model = model.to(device)
    # input_path = "/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_song_candidates_200w" # change this 
    # input_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/with_ugc_1219_data_150w.jsonl"
    # input_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music_clip_ids_1009_enriched_650w_final_dedup.jsonl"
    # input_path = "/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_song_candidates_8000w"
    # input_path="/mnt/bn/jiny-ttls-i18n-fr1q/guyanhang/data/all_song_candidates_8000w_0509_new"
    input_path="/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/MGR/core_music_data/20260101_20260331_200w_with_new_joined_160w/"
    if os.path.isdir(input_path):
        music_files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if (
                f.startswith("part-")
                or f.endswith(".jsonl")
                or f.endswith(".json")
            )
        ])
    else:
        music_files = [input_path]

    print(f"input_path = {input_path}")
    print(f"Total input files: {len(music_files)}")
    print("First 10 input files:")
    for f in music_files[:10]:
        print(f)

    total_num = 0
    print(f"Processing rank {rank} of {args.world_size}")
    per_rank_output = os.path.join(output_dir, f"music_ue_vector_rank{rank}.jsonl")
    if os.path.exists(per_rank_output):
        os.remove(per_rank_output)
    
    if len(music_files) == 1:
        print("Single file, shard by line")
        # shard by line
        finished_num = calculate_deployset_music_vectors(music_files[0], model, tokenizer, output_dir, device, args, shard_lines=True)
        total_num += finished_num
    else:
        print("Multiple files, shard by file")
        # shard by file
        for file_idx, music_dict_file in enumerate(music_files):
            if (file_idx % args.world_size) != rank:
                continue
            print(f"Processing file {file_idx}: {music_dict_file}")
            finished_num = calculate_deployset_music_vectors(music_dict_file, model, tokenizer, output_dir, device, args, shard_lines=False)
            total_num += finished_num
    
    print(f"Deal with {total_num} music items finished, starting to merge results...")
    torch.distributed.barrier()
    if rank == 0:
        merged_file = os.path.join(output_dir, "music_ue_vector.jsonl")
        with open(merged_file, "w") as fw:
            for r in range(args.world_size):
                part_file = os.path.join(output_dir, f"music_ue_vector_rank{r}.jsonl")
                if os.path.exists(part_file):
                    with open(part_file, "r") as fr:
                        for line in fr:
                            fw.write(line)
        print(f"Merged JSONL written to: {merged_file}")


def calculate_deployset_music_vectors(music_dict_file, model, tokenizer, output_dir, device, args, shard_lines=True):
    all_music_ids = []
    all_music_content_vector, all_music_title_vector, all_music_cover_vector, all_music_lyrics_vector = [], [], [], []
    all_music_attribute_input_ids, all_music_caption_input_ids = [], []
    all_music_dist_codes = []
    with open(music_dict_file, 'r') as f:
        for line_idx, line in enumerate(f):
            if shard_lines and ((line_idx % args.world_size) != args.rank):
                continue
            music_caption_dict = json.loads(line.strip())
            music_id = music_caption_dict['clip_id']

            # lang_in_caption = any([lang in music_caption_dict.get('content', '') for lang in ["念白", "对话", "口白", "口播", "独白","诵读", "音频", "旁白", "广播"]])
            # if lang_in_caption:
            #     continue    

            if ('content_vector' not in music_caption_dict) or ('content' not in music_caption_dict):
                continue
            music_content_vector = to_tensor(music_caption_dict['content_vector'])
            music_title_vector = to_tensor(music_caption_dict['title_vector'])
            music_cover_vector = to_tensor(music_caption_dict['cover_vector'])
            if 'lyric_vector' in music_caption_dict:
                music_lyrics_vector = to_tensor(music_caption_dict['lyric_vector'])
            else:
                music_lyrics_vector = torch.zeros_like(music_title_vector)
            music_attribute_input_ids, music_caption_input_ids = process_text(tokenizer, music_caption_dict, args)
            all_music_ids.append(music_id)
            all_music_content_vector.append(music_content_vector)
            all_music_title_vector.append(music_title_vector)
            all_music_cover_vector.append(music_cover_vector)
            all_music_lyrics_vector.append(music_lyrics_vector)
            all_music_attribute_input_ids.append(music_attribute_input_ids)
            all_music_caption_input_ids.append(music_caption_input_ids)
            all_music_dist_codes.append(2 if int(music_caption_dict.get('is_pgc', 0)) == 1 else (1 if music_caption_dict.get('meta_song_id', None) is not None else 0))
    all_music_content_vector = torch.stack(all_music_content_vector, dim=0)
    all_music_title_vector = torch.stack(all_music_title_vector, dim=0)
    all_music_cover_vector = torch.stack(all_music_cover_vector, dim=0)
    all_music_lyrics_vector = torch.stack(all_music_lyrics_vector, dim=0)
    all_music_attribute_input_ids = torch.stack(all_music_attribute_input_ids, dim=0)
    all_music_caption_input_ids = torch.stack(all_music_caption_input_ids, dim=0)

    feature_cache = {
        'all_music_ids': all_music_ids,
        'all_music_content_vector': all_music_content_vector,
        'all_music_title_vector': all_music_title_vector,
        'all_music_cover_vector': all_music_cover_vector,
        'all_music_lyrics_vector': all_music_lyrics_vector,
        'all_music_attribute_input_ids': all_music_attribute_input_ids,
        'all_music_caption_input_ids': all_music_caption_input_ids,
        'all_music_dist_codes': all_music_dist_codes,
    }
    feature_cache_file = os.path.join(output_dir, f"feature_cache_{os.path.basename(music_dict_file)[0:10]}_rank{args.rank}") # change this name to be rank specific

    torch.save(feature_cache, feature_cache_file)

    batch_size = 2048
    all_music_ue_vector = []
    print(f"Starting to inference")
    for i in tqdm(range(0, len(all_music_content_vector), batch_size)):
        batch_music_content_vector = all_music_content_vector[i:i+batch_size].to(device)
        batch_music_cover_vector = all_music_cover_vector[i:i+batch_size].to(device)
        batch_music_title_vector = all_music_title_vector[i:i+batch_size].to(device)
        batch_music_lyrics_vector = all_music_lyrics_vector[i:i+batch_size].to(device)
        batch_music_attribute_input_ids = all_music_attribute_input_ids[i:i+batch_size].to(device)
        batch_music_caption_input_ids = all_music_caption_input_ids[i:i+batch_size].to(device)
        batch_music_dist_codes = all_music_dist_codes[i:i+batch_size]

        with torch.no_grad():
            batch_music_ue_vector = model.extract_music_embeds(batch_music_content_vector, batch_music_cover_vector, batch_music_attribute_input_ids, batch_music_caption_input_ids, batch_music_dist_codes)
            batch_music_ue_vector = batch_music_ue_vector.cpu()
        
        all_music_ue_vector.append(batch_music_ue_vector)

    all_music_ue_vector = torch.cat(all_music_ue_vector, dim=0)
    
    output_file = os.path.join(output_dir, f"music_ue_vector_rank{args.rank}.jsonl")
    with open(output_file, "a") as fw:
        for index in range(len(all_music_ids)):
            record = {
                'music_id': all_music_ids[index],
                'music_emb': all_music_ue_vector[index].tolist(),
                'create_time': 'v20251222_whole'  # change this 
            }
            fw.write(json.dumps(record) + "\n")
    
    if args.output_pt:
        pt_file = os.path.join(output_dir, f"music_ue_vector_rank{args.rank}_{os.path.basename(music_dict_file)[0:10]}.pt")
        output_data = []
        for index in range(len(all_music_ids)):
            output_data.append({
                'music_id': all_music_ids[index],
                'music_ue_vector': all_music_ue_vector[index],
            })
        torch.save(output_data, pt_file)
    return len(all_music_ids)


if __name__ == '__main__':
    args = get_args()
    print(f"args:{args}")
    model_path = "/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base"
    ckpt_path =   os.path.join(args.data_path, "pytorch_model.bin")
    print(f"ckpt_path={ckpt_path}")

    output_dir =  os.path.join(args.data_path, "160w_music_MGR")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    extract_music_embeds_main_func(model_path, ckpt_path, output_dir, args)