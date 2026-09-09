# The misc tools for the project
import os
import sys
import json
sys.path.append('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_content')
from data_pipeline_utils import TiktokDataFusedPipeline
from tt_server import ImageUriToURL
import jsonlines
import time
import torch
import pickle
from tqdm import tqdm
import pandas as pd
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataset.hdfs_io import hopen, hlist_files
from IPython import embed
import random

data_pipeline = TiktokDataFusedPipeline()


def read_jsonl_file(filepath):
    try:
        with jsonlines.open(filepath, 'r') as reader:
            data = list(reader)
        return data
    except Exception as e:
        print(f"Loading {filepath} ERROR: {e}")
        return []


def upload_file_to_hdfs(local_path, remote_path):
    try:
        os.system(f"hdfs dfs -put {local_path} {remote_path}")
    except Exception as e:
        print(f"Upload ERROR {e}")


def process_one_file(hdfs_file, output_dir):
    parsed_json_items = []
    local_file_path = os.path.join(output_dir, os.path.basename(hdfs_file))
    with hopen(hdfs_file, 'r') as fr:
        for line in fr:
            item_list = line.decode().strip().split('\t')
            item_dict = {
                'item_id' : int(item_list[0]),
                'music_id' : int(item_list[1]),
                'music_content_vector': item_list[2],
                'music_title_vector': item_list[3],
                'music_cover_vector': item_list[4],
            }
            parsed_json_items.append(item_dict)

    with jsonlines.open(local_file_path, 'w') as fw:
        fw.write_all(parsed_json_items)


def process_music_library_file(hdfs_file, output_dir):
    parsed_json_items = []
    music_id_set = set()
    local_file_path = os.path.join(output_dir, os.path.basename(hdfs_file))

    with hopen(hdfs_file, 'r') as fr:
        for line in fr:
            item_list = line.decode().strip().split('\t')
            music_id = int(item_list[0])
            if music_id in music_id_set:
                continue
            
            item_dict = {
                'music_id' : music_id,
                'music_content_vector': item_list[1],
                'music_title_vector': item_list[2],
                'music_cover_vector': item_list[3],
            }

            music_content_vector = torch.tensor([float(ele) for ele in item_dict['music_content_vector'].split(',')])
            music_title_vector = torch.tensor([float(ele) for ele in item_dict['music_title_vector'].split(',')])
            music_cover_vector = torch.tensor([float(ele) for ele in item_dict['music_cover_vector'].split(',')])
            music_ue_vector = (music_content_vector + music_title_vector + music_cover_vector) / 3

            parsed_json_items.append({
                'music_id': music_id,
                'music_ue_vector': music_ue_vector,
            })
            music_id_set.add(music_id)

    torch.save(parsed_json_items, local_file_path)
    return len(parsed_json_items)


def prepare_music_rec_data():
    """
        Download the training data from HDFS (unpack the snappy data format to json format)
    """
    # hdfs_remote = 'hdfs://harunava//home/byte_recommend_anote/data/relevance_model/fine_tune_online_merged/20250503_spark'
    # hdfs_remote = 'hdfs://harunava//home/byte_recommend_anote/data/relevance_model/fine_tune_online_merged/20250503_spark_us'
    # hdfs_remote = 'hdfs://harunava/home/byte_recommend_anote/data/relevance_model/fine_tune_online_merged/20250531_spark'
    # hdfs_remote = 'hdfs://harunava//home/byte_recommend_anote/data/relevance_model/fine_tune_eval/0611'
    hdfs_remote = 'hdfs://harunava/home/byte_recommend_anote/data/relevance_model/fine_tune_eval/music_candidates_0618'
    # output_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_rec_data_v1'
    output_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_library'
    hdfs_files = hlist_files([hdfs_remote])
        
    task_queue = []
    total_num = 0
    with ProcessPoolExecutor(max_workers=128) as executor:
        for hdfs_file in hdfs_files:
            # task_queue.append(executor.submit(process_one_file, hdfs_file, output_dir))
            task_queue.append(executor.submit(process_music_library_file, hdfs_file, output_dir))

        with tqdm(total=len(task_queue)) as pbar:
            for future in futures.as_completed(task_queue):
                number = future.result()
                total_num += number
                pbar.update(1)

    print(f"Totally {total_num} items to deal with")


def spilt_meta_file_to_hdfs():
    """
        split all the training data to several hdfs files
    """
    # local_save_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_rec_data_demo'
    # local_save_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_rec_data_v1'
    local_save_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_eval_data_v1'
    item_num_per_file = 30
    file_num_per_subdir = 10000
    item_num_each_meta_file = item_num_per_file * file_num_per_subdir
    max_buffer_size = item_num_each_meta_file * 3
    buffer = []

    task_queue = []
    files = [os.path.join(local_save_dir, file) for file in os.listdir(local_save_dir)]
    meta_file_idx = 0

    with ProcessPoolExecutor(max_workers=32) as executor:
        for file in files:
            task_queue.append(executor.submit(read_jsonl_file, file))

        with tqdm(total=len(task_queue)) as pbar:
            for future in futures.as_completed(task_queue):
                data = future.result()
                buffer.extend(data)
                if len(buffer) >= max_buffer_size:
                    random.shuffle(buffer)
                    while len(buffer) >= item_num_each_meta_file:
                        meta_file = os.path.join(local_save_dir,'music_rec_meta_file_part-{:05d}.jsonl'.format(meta_file_idx))
                        with jsonlines.open(meta_file, 'w') as fw:
                            fw.write_all(buffer[:item_num_each_meta_file])
                        buffer = buffer[item_num_each_meta_file:]
                        meta_file_idx += 1
                pbar.update(1)

        while len(buffer) > 0:
            random.shuffle(buffer)
            meta_file = os.path.join(local_save_dir,'music_rec_meta_file_part-{:05d}.jsonl'.format(meta_file_idx))
            with jsonlines.open(meta_file, 'w') as fw:
                fw.write_all(buffer[:item_num_each_meta_file])
            buffer = buffer[item_num_each_meta_file:]
            meta_file_idx += 1

    print("Split file finished, start to upload to HDFS")
    # hdfs_upload_root = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/input_meta'
    # hdfs_upload_root = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/finetune_v1_meta'
    hdfs_upload_root = 'hdfs://harunasg/home/byte_data_tt_m/jinyang.leo/music_rec_data/eval_v1_meta'
    task_queue = []
    with ProcessPoolExecutor(max_workers=128) as executor:
        for idx in range(0, meta_file_idx):
            local_path = os.path.join(local_save_dir,'music_rec_meta_file_part-{:05d}.jsonl'.format(idx))
            task_queue.append(executor.submit(upload_file_to_hdfs, local_path, hdfs_upload_root))
        with tqdm(total=len(task_queue)) as pbar:
            for future in futures.as_completed(task_queue):
                _ = future.result()
                pbar.update(1)
    print("Done")


def check_jsonl_file(filepath):
    total_items = 0
    has_caption_items = 0
    missing_item_ids = []

    try:
        with jsonlines.open(filepath, 'r') as reader:
            data = list(reader)

        total_items = len(data)
        for item in data:
            if item['video_caption'] != '':
                has_caption_items += 1
            else:
                missing_item_ids.append({'item_id': item['item_id']})

        task_queue = []
        caption_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for missing_item in missing_item_ids:
                task_queue.append(executor.submit(data_pipeline.request_caption_info, missing_item['item_id']))

            for future in futures.as_completed(task_queue):
                result = future.result()
                if result is not None:
                    caption_results.append(result)
        
        output_path = filepath + '.caption'
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(caption_results)

        return total_items, has_caption_items
            
    except Exception as e:
        print(f"Loading {filepath} ERROR: {e}")
        return 0, 0


def check_the_data_status():
    root_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/finetune_v1_data_plus'
    subdirs = hlist_files([root_dir])

    total_items = 0
    has_caption_items = 0

    for subdir in tqdm(subdirs):
        part_files = hlist_files([subdir])
        part_files = [f for f in part_files if f.find('_SUCCESS') < 0 and f.find("_temporary") < 0 and f.find(".caption") < 0]
        print(f"Now processing {len(part_files)} items, {part_files[:2]}")
        task_queue = []

        with ProcessPoolExecutor(max_workers=50) as executor:
            for file in part_files:
                caption_result_file = file + '.caption'
                if not os.path.exists(caption_result_file):
                    task_queue.append(executor.submit(check_jsonl_file, file))

            for future in futures.as_completed(task_queue):
                a, b = future.result()
                total_items += a
                has_caption_items += b

    print(total_items, has_caption_items)


def merge_caption_worker(filepath, qwen_caption_dict):
    missing_item_ids = []
    tt_caption_items = []

    try:
        with jsonlines.open(filepath, 'r') as reader:
            data = list(reader)

        caption_result_file = filepath + '.caption'
        caption_data_dict = {}
        with jsonlines.open(caption_result_file, 'r') as reader:
            for item in reader:
                item_id = item['item_id']
                caption_data_dict[item_id] = item['video_caption_v2']

        for item in data:
            item_id = item['item_id']

            if item_id in caption_data_dict:
                item['video_caption'] = caption_data_dict[item_id]

            if item_id in qwen_caption_dict:
                item['video_caption'] = qwen_caption_dict[item_id]

            if item['video_caption'] == '':
                missing_item_ids.append({'item_id': item['item_id']})

        with jsonlines.open(filepath, 'w') as writer:
            writer.write_all(data)
            
    except Exception as e:
        print(f"Loading {filepath} ERROR: {e}")
    
    return missing_item_ids


def merge_the_caption_results():
    root_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/finetune_v1_data_plus'
    subdirs = hlist_files([root_dir])

    missing_item_ids = []
    qwen_caption_dict = {}
    with jsonlines.open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/captions_for_music_qwen_7b_all.jsonl', 'r') as reader:
        for item in tqdm(reader):
            item_id = item['item_id']
            qwen_caption_dict[item_id] = item['output_text']

    print(f"Totally {len(qwen_caption_dict)} items to for qwen 7B")

    tt_captions = []
    task_queue = []

    with ProcessPoolExecutor(max_workers=256) as executor:

        for subdir in tqdm(subdirs):
            part_files = hlist_files([subdir])
            part_files = [f for f in part_files if f.find('_SUCCESS') < 0 and f.find("_temporary") < 0 and f.find(".caption") < 0]
            print(f"Now processing {len(part_files)} items, {part_files[:2]}")
            
            for file in part_files:
                task_queue.append(executor.submit(merge_caption_worker, file, qwen_caption_dict))

        print(f"Totally {len(task_queue)} files to deal with")

        for future in tqdm(futures.as_completed(task_queue)):
            result = future.result()
            missing_item_ids.extend(result)
            # tt_caption = future.result()
            # tt_captions.extend(tt_caption)

    # with jsonlines.open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/tt_captions_for_music.jsonl', 'w') as writer:
    #     writer.write_all(tt_captions)
    
    print(f'Totally {len(missing_item_ids)} items to recap')
    with jsonlines.open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/missing_captions_for_music.jsonl', 'w') as writer:
        writer.write_all(missing_item_ids)


def merge_all_captions():
    qwen_caption_dict = {}
    with jsonlines.open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/captions_for_music_qwen_7b_all.jsonl', 'r') as reader:
        for item in tqdm(reader):
            item_id = item['item_id']
            qwen_caption_dict[item_id] = item['output_text']

    print(f"Totally {len(qwen_caption_dict)} items to for qwen 7B")

    data = []
    miss_num = 0
    with jsonlines.open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/tt_captions_for_music.jsonl', 'r') as reader:
        for item in tqdm(reader):
            if item['video_caption_v2'] == '':
                item_id = item['item_id']
                if item_id in qwen_caption_dict:
                    item['video_caption_v2'] = qwen_caption_dict[item_id]
            
            if item['video_caption_v2'] == '':
                miss_num += 1
            data.append(item)
    
    print(f"Missing caption items: {miss_num}")
    with jsonlines.open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/tt_captions_for_music.jsonl', 'w') as writer:
        writer.write_all(data)


def merge_feature_worker(hdfs_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename_dict = {}
    # image_url_server = ImageUriToURL()
    with hopen(hdfs_path, 'r') as fr:
        for line in tqdm(fr):
            item_list = line.decode().strip()
            item = json.loads(item_list)
            music_id = item['clip_id']
            # music_uri = item['cover_uri']
            # music_url = image_url_server.request(music_uri)
            # if music_url is None:
            #     music_url = ''
            # else:
            #     music_url = music_url[0]
            # item['cover_url'] = music_url
            output_path = os.path.join(output_dir, f'{music_id}.json')
            with open(output_path, 'w') as fw:
                json.dump(item, fw)
            filename_dict[music_id] = output_path
    
    return filename_dict
            

# 将lutong给到的music id的caption 和 meta info 信息 和 原本的music vector进行合并
def merge_music_features():
    hdfs_anno_dir = 'hdfs://harunava/home/byte_recommend_anote/data/relevance_model/fine_tune_train/music_clip_ids_1009_enriched_650w_all'
    output_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions'
    task_queue = []

    filename_dict = {}
    with ProcessPoolExecutor(max_workers=256) as executor:
        for index in range(200):
            anno_file_name = 'part-{:05d}-57f1d76e-f57d-4880-81d0-91357fe7c368-c000.json'.format(index)
            anno_path = f'{hdfs_anno_dir}/{anno_file_name}'
            subdir = 'part-{:05d}'.format(index)
            anno_output_dir = os.path.join(output_dir, subdir)
            task_queue.append(executor.submit(merge_feature_worker, anno_path, anno_output_dir))

        for future in tqdm(futures.as_completed(task_queue)):
            result = future.result()
            filename_dict.update(result)

    print(f"totally {len(filename_dict)} item ids")
    with open('/mnt/bn/jiny-ttls-i18n-fr1q/ttls_hive_data/music_captions/music_clip_ids_1009_enriched_650w.pkl', 'wb') as fw:
        pickle.dump(filename_dict, fw)


def get_language_mapping():
    mapping_file = '/mnt/bn/jiny-ttls-i18n-fr1q/MMPretrain/music_ue-country_code_mapping-Query.csv'
    language_mapping = {}

    df = pd.read_csv(mapping_file)
    for _, row in df.iterrows():
        lg_code = row['language_code']
        english_full_name = row['english_full_name']
        language_mapping[lg_code] = english_full_name

    print(language_mapping)


if __name__ == "__main__":
    # get_language_mapping()
    merge_music_features()
    # uri = 'tos-alisg-v-2102/oI9UZmwWmIQADABh3CCFfA5EQqDADaCXrgfsNQ'
    # image_url_server = ImageUriToURL()
    # print(image_url_server.request([uri, 'tos-alisg-v-2102/dceffaf38ead49c096149b0b2ec84659']))
    # prepare_music_rec_data()
    # spilt_meta_file_to_hdfs()
    # check_the_data_status()
    # merge_the_caption_results()
    # merge_all_captions()
    # data_pipeline = TiktokDataFusedPipeline()
    # item_id = 7503539237917035783
    # item_id = 7495522364176747794
    # result = data_pipeline.request_caption_info(item_id)
    # print(result)
