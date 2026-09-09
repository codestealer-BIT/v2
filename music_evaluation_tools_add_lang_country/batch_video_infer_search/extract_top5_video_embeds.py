#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/lutong/checkpoints/video_siglip2_for_music_only_match_llm_v5_filter_music_220w_continue_from_39200_unfreeze_text/checkpoint-40000

python music_evaluation_tools_add_lang_country/batch_video_infer_search/extract_top5_video_embeds.py \
  --v2v_jsonl /mnt/bn/jiny-ttls-i18n-fr1q/lutong/demo_cases/search_results_220w_top500_new.jsonl \
  --embeddings_dir "$EMB_PATH/220w_4_5_only" \
  --output_dir "$EMB_PATH/220w_eval_top5_videos" \
  --topk 5
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
import pandas as pd


def load_v2v_jsonl(v2v_jsonl_path, max_video = 5):
    df = pd.read_json(v2v_jsonl_path, lines=True, dtype={'music_id': str, 'item_id': str})
    df['score'] = df['score'].astype(int)
    df['sim'] = df['sim'].astype(float)
    df['query_item_id'] = df['query_item_id'].apply(lambda x: x[0])

    df_sorted = df.sort_values(['query_item_id','sim'], ascending=[True, False])
    base = (df_sorted.groupby('query_item_id', sort=False)['item_id']
            .apply(lambda s: s.drop_duplicates().tolist())
            .reset_index(name='item_ids'))
    rows = []
    needed_item_ids = set()
    for row in base.itertuples(index=False):
        item_ids = row.item_ids[:max_video]
        rows.append((str(row.query_item_id), item_ids))
        for item_id in item_ids:
            needed_item_ids.add(str(item_id))
    return rows, needed_item_ids

def load_top5_csv(top5_csv_path, query_col, item_ids_col, topk):
    rows = []
    needed_item_ids = set()
    with open(top5_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row[query_col]
            item_ids_str = row[item_ids_col].strip()
            item_ids = item_ids_str.split() if item_ids_str else []
            if topk is not None:
                item_ids = item_ids[:topk]
            rows.append((query_id, item_ids))
            for item_id in item_ids:
                needed_item_ids.add(str(item_id))
    return rows, needed_item_ids


def find_embedding_pairs(embeddings_dir):
    meta_files = sorted(Path(embeddings_dir).glob("metadata_rank_*_chunk_*.csv"))
    pairs = []
    for meta_path in meta_files:
        emb_path = Path(str(meta_path).replace("metadata_rank_", "embeddings_rank_").replace(".csv", ".npy"))
        if emb_path.exists():
            pairs.append((meta_path, emb_path))
    return pairs


def collect_embeddings(pairs, needed_item_ids):
    itemid_to_embed = {}
    missing = set(needed_item_ids)
    for meta_path, emb_path in tqdm(pairs, desc="Scanning chunks"):
        if not missing:
            break
        matched = []
        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row["item_id"]
                if item_id in missing:
                    matched.append((item_id, int(row["idx"])))
        if matched:
            emb = np.load(emb_path, mmap_mode="r")
            for item_id, idx in matched:
                if item_id in missing:
                    itemid_to_embed[item_id] = emb[idx].astype(np.float32)
                    missing.remove(item_id)
    return itemid_to_embed, missing


def infer_embedding_dim(pairs, itemid_to_embed):
    if itemid_to_embed:
        return int(next(iter(itemid_to_embed.values())).shape[0])
    if not pairs:
        raise RuntimeError("No embedding files found.")
    emb = np.load(pairs[0][1], mmap_mode="r")
    return int(emb.shape[1])


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2v_jsonl", required=True, help="jsonl with v2v raw results")
    parser.add_argument("--embeddings_dir", required=True, help="Directory containing metadata_rank_*.csv and embeddings_rank_*.npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_col", default="query_item_id")
    parser.add_argument("--item_ids_col", default="item_ids_str")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    rows, needed_item_ids = load_v2v_jsonl(args.v2v_jsonl, args.topk)
    print(f"total {len(needed_item_ids)} needed_item_ids")
    pairs = find_embedding_pairs(args.embeddings_dir)
    if not pairs:
        raise RuntimeError(f"No metadata/embedding pairs found in {args.embeddings_dir}")

    itemid_to_embed, missing = collect_embeddings(pairs, needed_item_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = output_dir / "top5_itemid_to_embed.npz"

    out_item_ids = []
    out_embeds = []
    missing_pairs = 0
    for query_id, item_ids in rows:
        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in itemid_to_embed:
                out_item_ids.append(f"{item_id_str}_{query_id}")
                out_embeds.append(itemid_to_embed[item_id_str])
            else:
                missing_pairs += 1
    item_ids = out_item_ids
    embeds = np.stack(out_embeds, axis=0) if out_embeds else np.empty((0, 0), dtype=np.float32)
    np.savez(out_npz, item_ids=np.array(item_ids), embeddings=embeds)

    print(f"Saved item_id->embedding: {out_npz}")
    print(f"Unique item_ids needed: {len(needed_item_ids)}")
    print(f"Found item_ids: {len(itemid_to_embed)}")
    print(f"Missing item_ids: {len(missing)}")
    print(f"Missing pairs: {missing_pairs}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
