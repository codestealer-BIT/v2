#!/usr/bin/env python
"""
export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_filter_music_220w_continue_from_39200_unfreeze_text/checkpoint-40000
python music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_build.py \
  --embeds_dir "$EMB_PATH/220w_4_5_only" \
  --index_out "$EMB_PATH/220w_4n5_faiss_index_flatip.index" \
  --id_map_out "$EMB_PATH/220w_4n5_id_map.csv" \
  --normalize 2>&1 | tee LT/logs/faiss_build_220w.log

export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200
python music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_build.py \
  --embeds_dir "$EMB_PATH/220w_4_5_only" \
  --index_out "$EMB_PATH/220w_4n5_faiss_index_flatip.index" \
  --id_map_out "$EMB_PATH/220w_4n5_id_map.csv" \
  --normalize 2>&1 | tee LT/logs/faiss_build_220w.log

export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200
python music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_build.py \
  --embeds_dir "$EMB_PATH/selected_1w_by_tag" \
  --index_out "$EMB_PATH/50m_faiss_index_flatip.index" \
  --id_map_out "$EMB_PATH/50m_id_map.csv" \
  --normalize 2>&1 | tee LT/logs/faiss_build.log

export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200
python music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_build.py \
  --embeds_dir "$EMB_PATH/50m_4_5_only" \
  --index_out "$EMB_PATH/50m_4n5_faiss_index_flatip.index" \
  --id_map_out "$EMB_PATH/50m_4n5_id_map.csv" \
  --normalize 2>&1 | tee LT/logs/faiss_build_4n5.log

export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200
python music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_build.py \
  --parquet_dir 'hdfs://harunava/home/byte_tiktok_music/proj/content_understanding/video/video_ue/0129_video_ue_v3_20260204_to_20260206' \
  --parquet_emb_col emb \
  --parquet_batch_size 100000 \
  --index_out "$EMB_PATH/online_66m_faiss_index_flatip.index" \
  --id_map_out "$EMB_PATH/online_66m_id_map.csv" \
  --count_rows \
  --normalize 2>&1 | tee LT/logs/faiss_build_online.log
"""
import os
import argparse
import glob
import numpy as np
import faiss
import csv
import time

def l2_normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeds_dir', type=str, default=None)
    p.add_argument('--embeds_glob', type=str, default='embeddings_rank_*_chunk_*.npy')
    p.add_argument('--index_out', type=str, required=True)
    p.add_argument('--id_map_out', type=str, required=True)
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--parquet_dir', type=str, default=None)
    p.add_argument('--parquet_emb_col', type=str, default='emb')
    p.add_argument('--parquet_batch_size', type=int, default=100000)
    p.add_argument('--count_rows', action='store_true')
    args = p.parse_args()
    print(f"args:{args}")

    idx_dir = os.path.dirname(args.index_out) or '.'
    map_dir = os.path.dirname(args.id_map_out) or '.'
    if idx_dir == '/' or map_dir == '/':
        raise RuntimeError('output directory cannot be root; set a valid EMB_PATH or output dir')
    os.makedirs(idx_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    if not os.access(idx_dir, os.W_OK) or not os.access(map_dir, os.W_OK):
        raise RuntimeError(f'no write permission to output dirs: {idx_dir}, {map_dir}')

    if args.parquet_dir:
        import pyarrow.dataset as ds
        ds_obj = ds.dataset(args.parquet_dir, format="parquet")
        schema = ds_obj.schema
        cols = [f.name for f in schema]
        if args.parquet_emb_col not in cols:
            raise RuntimeError('parquet emb column not found')
        meta_cols = [c for c in cols if c != args.parquet_emb_col]
        total_rows = None
        if bool(args.count_rows):
            tcr0 = time.time()
            try:
                total_rows = ds_obj.count_rows()
            except Exception:
                total_rows = None
            print(f"count_rows took {time.time() - tcr0:.2f}s", flush=True)
        with open(args.id_map_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['global_idx'] + meta_cols)
            index = None
            global_idx = 0
            try:
                scanner = ds.Scanner.from_dataset(ds_obj, columns=cols, batch_size=args.parquet_batch_size)
            except Exception:
                scanner = ds_obj.scan(columns=cols, batch_size=args.parquet_batch_size)
            t0 = time.time()
            for rb in scanner.to_batches():
                emb_arr = rb.column(args.parquet_emb_col)
                emb_py = emb_arr.to_pylist()
                mat = np.array(emb_py, dtype=np.float32)
                if args.normalize:
                    mat = l2_normalize(mat.astype(np.float32))
                if index is None:
                    d = mat.shape[1]
                    index = faiss.IndexFlatIP(d)
                index.add(mat)
                for i in range(rb.num_rows):
                    row_out = [global_idx + i]
                    for c in meta_cols:
                        val = rb.column(c)[i].as_py()
                        row_out.append("" if val is None else str(val))
                    w.writerow(row_out)
                global_idx += rb.num_rows
                elapsed = time.time() - t0
                rate = global_idx / max(elapsed, 1e-9)
                if total_rows is None:
                    print(f"processed {global_idx} elapsed {elapsed:.2f}s rate {rate:.0f}/s", flush=True)
                else:
                    print(f"processed {global_idx}/{total_rows} elapsed {elapsed:.2f}s rate {rate:.0f}/s", flush=True)
        faiss.write_index(index, args.index_out)
        print(f"finished processed {global_idx}" + (f"/{total_rows}" if total_rows is not None else "") + f" in {time.time() - t0:.2f}s", flush=True)
        return

    if not args.embeds_dir:
        raise RuntimeError('embeds_dir is required when parquet_dir is not set')
    emb_files = sorted(glob.glob(os.path.join(args.embeds_dir, args.embeds_glob)))
    if len(emb_files) == 0:
        raise RuntimeError('no embeddings chunk files found')
    print(f"Found {len(emb_files)} embeddings chunk files")

    all_embeds = []
    id_rows = []
    meta_headers = None
    processed = 0
    t0 = time.time()
    for ef in emb_files:
        em = np.load(ef)
        base = os.path.basename(ef)
        meta_base = base.replace('.npy', '.csv').replace('embeddings', 'metadata')
        mf = os.path.join(args.embeds_dir, meta_base) # find corresponding metadata file
        if not os.path.exists(mf):
            raise RuntimeError(f"metadata file {mf} not found")
        with open(mf, 'r') as f:
            rd = csv.DictReader(f)
            if meta_headers is None:
                meta_headers = [h for h in rd.fieldnames if h != 'idx']
            for row in rd:
                id_rows.append(row)
        all_embeds.append(em)
        processed += em.shape[0]
        elapsed = time.time() - t0
        rate = processed / max(elapsed, 1e-9)
        print(f"processed {processed} elapsed {elapsed:.2f}s rate {rate:.0f}/s", flush=True)

    mat = np.concatenate(all_embeds, axis=0)
    print(f"embedding shape: {mat.shape}")
    if args.normalize:
        mat = l2_normalize(mat.astype(np.float32))
    else:
        mat = mat.astype(np.float32)

    d = mat.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(mat)
    faiss.write_index(index, args.index_out)
    print(f"finished processed {processed} in {time.time() - t0:.2f}s", flush=True)

    with open(args.id_map_out, 'w', newline='') as f:
        w = csv.writer(f)
        header_out = ['global_idx'] + (meta_headers if meta_headers is not None else ['music_id','item_id','score'])
        w.writerow(header_out)
        for i, r in enumerate(id_rows):
            row_out = [i]
            for h in header_out[1:]:
                row_out.append(r.get(h, ''))
            w.writerow(row_out)

if __name__ == '__main__':
    main()
