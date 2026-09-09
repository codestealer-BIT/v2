#!/usr/bin/env python
"""
export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200
export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_filter_music_220w_continue_from_39200_unfreeze_text/checkpoint-40000

python -u music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_search.py \
    --index_path "$EMB_PATH/220w_4n5_faiss_index_flatip.index" \
    --id_map_path "$EMB_PATH/220w_4n5_id_map.csv" \
    --query_meta_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/query_meta.csv \
    --query_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/query.npy --normalize --topk 500 \
    --output_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/search_results_220w_top500_39200.jsonl 2>&1 | tee LT/logs/faiss_search_220w.log

python -u music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_search.py \
    --index_path "$EMB_PATH/50m_faiss_index_flatip.index" \
    --id_map_path "$EMB_PATH/50m_id_map.csv" \
    --query_meta_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/query_meta.csv \
    --query_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/query.npy --normalize --topk 100 \
    --output_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/search_results_t100_50m.jsonl 2>&1 | tee LT/logs/faiss_search_eval_t100_50m.log

python -u music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_search.py \
    --index_path "$EMB_PATH/online_66m_faiss_index_flatip.index" \
    --id_map_path "$EMB_PATH/online_66m_id_map.csv" \
    --query_meta_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/query_meta.csv \
    --query_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/query.npy --normalize --topk 1000 \
    --output_path /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/search_results_t1000_66m.jsonl 2>&1 | tee LT/logs/faiss_search_eval_t1000_66m.log

python -u music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_search.py \
    --index_path "$EMB_PATH/online_66m_faiss_index_flatip.index" \
    --id_map_path "$EMB_PATH/online_66m_id_map.csv" \
    --query_meta_path "$EMB_PATH/5w5_query_meta.csv" \
    --query_path "$EMB_PATH/5w5_query.npy" --normalize --topk 1000 \
    --output_path $EMB_PATH/v2v_0217_5w5_online_top1000.jsonl 2>&1 | tee LT/logs/faiss_search_66m_5w5_1000.log

python -u music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_search.py \
    --index_path "$EMB_PATH/50m_4n5_faiss_index_flatip.index" \
    --id_map_path "$EMB_PATH/50m_4n5_id_map.csv" \
    --query_meta_path "$EMB_PATH/5w5_query_meta.csv" \
    --query_path "$EMB_PATH/5w5_query.npy" --normalize --topk 300 \
    --output_path $EMB_PATH/v2v_0217_5w5_offline_top300.jsonl 2>&1 | tee LT/logs/faiss_search_40m_5w5_300.log
"""
import argparse
import numpy as np
import faiss
import csv
import json
import time
import sys
import os
from tqdm import tqdm

def l2_normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index_path', type=str, required=True)
    p.add_argument('--id_map_path', type=str, required=True)
    p.add_argument('--query_path', type=str, required=True)
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--topk', type=int, default=20)
    p.add_argument('--output_path', type=str, required=True)
    p.add_argument('--query_meta_path', type=str, default=None)
    # gpu error: CUDA error 209 no kernel image is available for execution on the device
    p.add_argument('--batch_size', type=int, default=4096)
    args = p.parse_args()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    index = faiss.read_index(args.index_path)
    print("loaded index", flush=True)
    print(f"index.ntotal:{index.ntotal}, index.d:{index.d}", flush=True)
    
    q = np.load(args.query_path)
    if args.normalize:
        q = l2_normalize(q.astype(np.float32))
    else:
        q = q.astype(np.float32)
    print(f"loaded queries shape={q.shape} normalize={bool(args.normalize)} topk={args.topk} batch_size={args.batch_size}", flush=True)

    query_meta = None
    if args.query_meta_path:
        query_meta = []
        with open(args.query_meta_path, 'r') as f:
            rd = csv.reader(f)
            next(rd)
            for row in tqdm(rd, desc="loading query_meta"):
                if len(row) == 3: # global_idx,item_id,tag
                    query_meta.append((row[1], row[2])) # item_id, tag
                elif len(row) == 2: # idx,item_id
                    query_meta.append((row[1], None)) # item_id, tag

    id_map = []
    with open(args.id_map_path, 'r') as f:
        rd = csv.DictReader(f)
        for row in tqdm(rd, desc="loading id_map"):
            id_map.append(row)

    print("starting batched search", flush=True)
    t = tqdm(total=q.shape[0], desc="faiss search", unit="q", file=sys.stdout) if tqdm is not None else None
    print(f"writing results to {args.output_path}", flush=True)
    written = 0
    with open(args.output_path, 'w') as out:
        n = q.shape[0]
        bs = max(1, int(args.batch_size))
        for start in range(0, n, bs):
            end = min(start + bs, n)
            D, I = index.search(q[start:end], args.topk)
            for local_qi in range(I.shape[0]):
                qi = start + local_qi
                for rk in range(I.shape[1]):
                    idx = int(I[local_qi, rk])
                    sim = float(D[local_qi, rk])
                    m = id_map[idx]
                    obj = {
                        'query_id': qi,
                        'rank': rk,
                        'sim': sim,
                        'global_idx': idx
                    }
                    obj.update(m)
                    if query_meta is not None and qi < len(query_meta):
                        query_data = query_meta[qi]
                        obj['query_item_id'] = [query_data[0]]
                        obj['query_tag'] = [query_data[1]]
                    out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                    written += 1
            if t is not None:
                t.update(end - start)
        out.flush()
        os.fsync(out.fileno())
    if t is not None:
        t.close()
    if os.path.exists(args.output_path):
        print(f"wrote {written} lines to {args.output_path}", flush=True)
    else:
        print(f"warning: output not found at {args.output_path}", flush=True)

if __name__ == '__main__':
    print("starting faiss exact search", flush=True)
    start_time = time.time()
    try:
        main()
    finally:
        print("finish search, time cost: ", time.time() - start_time, flush=True)
        sys.stdout.flush()
