"""
export EMB_PATH=/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/checkpoints/video_siglip2_for_music_only_match_huber_loss_delta_1_filter_music_ids/checkpoint-39200

python music_evaluation_tools_add_lang_country/batch_video_infer_search/map_sub_emb_from_meta.py \
  --index_path "$EMB_PATH/50m_faiss_index_flatip.index" \
  --id_map_path "$EMB_PATH/50m_id_map.csv" \
  --subset_path "$EMB_PATH/5w5_id_map.csv" \
  --query_out "$EMB_PATH/5w5_query.npy" \
  --query_meta_out "$EMB_PATH/5w5_query_meta.csv" \
  --key item_id
"""
import argparse
import csv
import numpy as np
import faiss

def read_csv(path):
    rows = []
    with open(path, "r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index_path", type=str, required=True)
    p.add_argument("--id_map_path", type=str, required=True)
    p.add_argument("--subset_path", type=str, required=True)
    p.add_argument("--query_out", type=str, required=True)
    p.add_argument("--query_meta_out", type=str, required=True)
    p.add_argument("--key", type=str, default=None)
    args = p.parse_args()

    id_rows = read_csv(args.id_map_path)
    subset_rows = read_csv(args.subset_path)

    key = args.key
    if key is not None:
        key2idx = {}
        for i, r in enumerate(id_rows):
            if key in r:
                k = r[key]
                if k not in key2idx:
                    key2idx[k] = i

    indices = []
    out_meta = []
    for r in subset_rows:
        if "global_idx" in r and r["global_idx"] != "":
            i = int(r["global_idx"])
        else:
            k = r.get(key, "")
            i = key2idx.get(k, None)
            if i is None:
                continue
        indices.append(i)
        out_meta.append(r)

    index = faiss.read_index(args.index_path)
    d = index.d
    q = np.empty((len(indices), d), dtype=np.float32)
    for j, i in enumerate(indices):
        v = index.reconstruct(i)
        q[j] = v.astype(np.float32)

    np.save(args.query_out, q)
    with open(args.query_meta_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_meta[0].keys()))
        wr.writeheader()
        for r in out_meta:
            wr.writerow(r)

if __name__ == "__main__":
    main()
