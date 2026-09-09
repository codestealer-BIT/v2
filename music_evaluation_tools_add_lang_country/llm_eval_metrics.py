# -*- coding: utf-8 -*-
"""
llm_eval_metrics.py

Evaluate embedding-retrieval similarity results with LLM judgments.

Inputs:
1) similarity csv: each line
   item_id, "mid1 mid2 ...", "sim1 sim2 ..."
   (music_ids already sorted by model score DESC in your generation script, but we still sort robustly)

2) llm tsv: each line
   item_id \t music_id \t score \t comment

Metrics:
- Binary AUC: binarize LLM score (>=3 for 1-5 scale, >=7 for 1-10 scale)
- Top@K Recall / Precision (K=10/20/50) on binary label
- regAUC (C-index): use raw LLM score (ordinal) without binarization
- Hit@50 (distribution-based): compare LLM-score distribution between "optimal top50 by LLM score"
  and "predicted top50 by model score", ignore exact id coverage

Outputs:
- Print summary metrics
- score_sample.csv: per item, top20 music ids, llm scores, model scores
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import re


# -------------------------
# Utility: regAUC = C-index
# -------------------------
def reg_auc_cindex(llm_scores: np.ndarray, model_scores: np.ndarray, max_score: int, ignore_zero_llm: bool = True) -> float:
    """
    regAUC (C-index / concordance probability):
    Consider all pairs (i, j) with llm_score[i] != llm_score[j].
    Count as correct if ordering in model_scores matches ordering in llm_scores.
    If model_scores tie, count as 0.5.
    Return correct / total_pairs.

    This is a standard "generalized AUC" for ordinal labels.

    Args:
      llm_scores: shape [N], raw LLM score (1..max_score). 0 may mean "missing"
      model_scores: shape [N], predicted similarity score
      max_score: 5 or 10
      ignore_zero_llm: if True, drop llm_scores==0 (missing) before computing

    Returns:
      float in [0,1] or NaN if not enough comparable pairs.
    """
    if llm_scores.size <= 1:
        return np.nan

    if ignore_zero_llm:
        m = llm_scores > 0
        llm_scores = llm_scores[m]
        model_scores = model_scores[m]

    n = llm_scores.size
    if n <= 1:
        return np.nan

    # Sort by model score ascending, and group ties
    order = np.argsort(model_scores, kind="mergesort")
    llm_sorted = llm_scores[order].astype(np.int32)
    model_sorted = model_scores[order].astype(np.float64)

    # Prefix counts of llm labels among strictly smaller model scores
    cnt = np.zeros(max_score + 1, dtype=np.int64)
    total_prev = 0

    concord = 0.0
    tie_pred = 0.0
    total_pairs = 0.0

    i = 0
    while i < n:
        j = i
        # same model score group [i, j)
        while j < n and model_sorted[j] == model_sorted[i]:
            j += 1

        group_llm = llm_sorted[i:j]
        group_size = j - i

        # Pairs between previous (lower model score) and current group:
        # For each sample with llm=s:
        #   correct if previous llm < s (because model_prev < model_curr)
        #   comparable pairs are those with previous llm != s; but we count via lower/higher.
        for s in group_llm:
            lower = cnt[:s].sum()  # prev llm < s
            higher = total_prev - cnt[:s+1].sum()  # prev llm > s
            concord += lower
            total_pairs += (lower + higher)

        # Pairs inside group: model ties => 0.5 for llm-different pairs
        if group_size >= 2:
            total_g = group_size * (group_size - 1) / 2
            _, counts = np.unique(group_llm, return_counts=True)
            same_g = np.sum(counts * (counts - 1) / 2)
            diff_g = total_g - same_g
            tie_pred += diff_g * 0.5
            total_pairs += diff_g

        # update prefix counts
        for s in group_llm:
            if 0 <= s <= max_score:
                cnt[s] += 1
        total_prev += group_size
        i = j

    if total_pairs <= 0:
        return np.nan
    return float((concord + tie_pred) / total_pairs)


# -------------------------
# Tag loader
# -------------------------
def load_tag_df(tag_path: str) -> pd.DataFrame:
    df = pd.read_json(tag_path, lines=True, dtype={"item_id": "string"})
    if "item_id" not in df.columns:
        raise ValueError("tag_df must contain 'item_id'")
    if "tag" not in df.columns and "tags" not in df.columns:
        raise ValueError("tag_df must contain 'tag' or 'tags'")
    df["item_id"] = df["item_id"].apply(lambda x: int(x))
    df["tag"] = df["tag"].astype(str)
    print(f"Loaded {len(df)} item tags.")
    return df


# -------------------------
# Utility: Hit@K by LLM-score distribution
# -------------------------
def hit_rate_by_topk_distribution(
    llm_scores_all: np.ndarray,
    model_scores_all: np.ndarray,
    topk: int,
    max_score: int,
    ignore_zero_llm_for_opt: bool = True
) -> tuple[float, float]:
    """
    Distribution-based Hit@K:
    - "Optimal topK": pick topK by LLM score (ignore exact ID coverage), count distribution opt_cnt[s]
    - "Pred topK": pick topK by model score, then look at their LLM scores, count distribution pred_cnt[s]
    - hit = sum_s min(opt_cnt[s], pred_cnt[s])
    - hit_rate = hit / K

    Args:
      llm_scores_all: [N], LLM score for each candidate in this video line. 0 may mean missing.
      model_scores_all: [N], model similarity score.
      topk: K, typically 50.
      max_score: 5 or 10.
      ignore_zero_llm_for_opt: if True, "optimal topK" is selected from llm_scores>0 only.

    Returns:
      float in [0,1] or NaN.
    """
    n = llm_scores_all.size
    if n == 0:
        return np.nan, np.nan

    # Choose candidate pool for "optimal"
    if ignore_zero_llm_for_opt:
        valid = llm_scores_all > 0
        if valid.sum() == 0:
            return np.nan, np.nan
        llm_opt = llm_scores_all[valid]
        # If we filter for opt, topk is limited by labeled size
        k_opt = min(topk, llm_opt.size)
        opt_idx_local = np.argsort(-llm_opt)[:k_opt]
        opt_scores = llm_opt[opt_idx_local].astype(int)
        # For hit denominator we still use topk (as you defined hit@50),
        # but if labeled pool < topk, we fallback to that size to avoid weirdness.
        k = k_opt
    else:
        k = min(topk, n)
        opt_idx = np.argsort(-llm_scores_all)[:k]
        opt_scores = llm_scores_all[opt_idx].astype(int)

    # Pred topK always from full candidates by model score, but its LLM score may be 0 (missing)
    k_pred = min(k, n)
    pred_idx = np.argsort(-model_scores_all)[:k_pred]
    pred_scores = llm_scores_all[pred_idx].astype(int)

    opt_cnt = np.zeros(max_score + 1, dtype=np.int64)
    pred_cnt = np.zeros(max_score + 1, dtype=np.int64)

    for s in opt_scores:
        if 1 <= s <= max_score:
            opt_cnt[s] += 1
    for s in pred_scores:
        if 1 <= s <= max_score:
            pred_cnt[s] += 1

    hit = 0
    hit_high = 0
    high_total = 0
    for s in range(1, max_score + 1):
        hit += min(opt_cnt[s], pred_cnt[s])
        if s >= 4:
            hit_high += min(opt_cnt[s], pred_cnt[s])
            high_total += opt_cnt[s]
    if k_pred == 0:
        return np.nan, np.nan
    return float(hit / k_pred), float(hit_high / high_total) if high_total > 0 else np.nan


def dcg_at_k(rels: np.ndarray, k: int) -> float:
    if rels.size == 0 or k <= 0:
        return np.nan
    k = min(k, rels.size)
    gains = (2 ** rels[:k] - 1).astype(np.float64)
    discounts = np.log2(np.arange(2, k + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(llm_scores: np.ndarray, model_scores: np.ndarray, k: int) -> float:
    if llm_scores.size == 0 or model_scores.size == 0:
        return np.nan
    order_pred = np.argsort(-model_scores)
    order_ideal = np.argsort(-llm_scores)
    dcg = dcg_at_k(llm_scores[order_pred], k)
    idcg = dcg_at_k(llm_scores[order_ideal], k)
    if idcg <= 0 or np.isnan(idcg):
        return np.nan
    return float(dcg / idcg)


# -------------------------
# Main evaluation
# -------------------------
def evaluate_similarity_result(sim_csv_path, llm_tsv_path, output_path, max_score=5, tag_path=None):
    # 1) Load LLM judgments
    llm_df = pd.read_csv(
        llm_tsv_path,
        sep="\t",
        header=None,
        names=["item_id", "music_id", "score", "comment"],
        engine="python",
    )
    llm_df["item_id"] = llm_df["item_id"].astype(int)
    llm_df["music_id"] = llm_df["music_id"].astype(int)
    llm_df["score"] = pd.to_numeric(llm_df["score"], errors="coerce").fillna(0).astype(float)

    if max_score == 10:
        llm_df["relevant"] = (llm_df["score"] >= 7).astype(int)
    elif max_score == 5:
        llm_df["relevant"] = (llm_df["score"] >= 3).astype(int)
    else:
        raise ValueError("max_score must be 5 or 10")

    # map for fast lookup
    llm_dict = {(int(r.item_id), int(r.music_id)): r for r in llm_df.itertuples(index=False)}
    print(f"Loaded {len(llm_df)} LLM judged pairs from {llm_tsv_path}")

    # 2) Load similarity results
    results = []
    with open(sim_csv_path, "r") as fr:
        for line in tqdm(fr, desc="Reading similarity csv"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            item_id = int(parts[0])
            music_ids = parts[1].split()
            sim_scores = parts[2].split()
            if len(music_ids) != len(sim_scores):
                continue
            try:
                music_ids = [int(x) for x in music_ids]
                sim_scores = [float(x) for x in sim_scores]
            except Exception:
                continue
            results.append((item_id, music_ids, sim_scores))

    print(f"Loaded {len(results)} video lines from {sim_csv_path}")

    # 3) Metrics accumulators
    all_auc = []
    all_regauc = []
    all_hit50 = []
    all_hit_high_50 = []
    all_ndcg10, all_ndcg20, all_ndcg50 = [], [], []

    all_top10_recall, all_top20_recall, all_top50_recall = [], [], []
    all_top10_prec, all_top20_prec, all_top50_prec = [], [], []

    score_samples = []
    item_metrics = []

    for item_id, music_ids, sim_scores in tqdm(results, desc="Evaluating per item"):
        y_score = np.asarray(sim_scores, dtype=np.float64)

        # LLM raw scores (0 means missing)
        llm_scores = np.asarray(
            [
                float(llm_dict[(item_id, mid)].score) if (item_id, mid) in llm_dict else 0.0
                for mid in music_ids
            ],
            dtype=np.float64,
        )

        # Binary labels for classical AUC/recall/precision
        y_true = np.asarray(
            [
                int(llm_dict[(item_id, mid)].relevant) if (item_id, mid) in llm_dict else 0
                for mid in music_ids
            ],
            dtype=np.int32,
        )

        # Skip if no positive in binary labels (same as your original behavior)
        if y_true.sum() == 0:
            continue

        # --- AUC (binary) ---
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan
        all_auc.append(auc)

        # --- regAUC (C-index using raw LLM score) ---
        regauc = reg_auc_cindex(llm_scores, y_score, max_score=max_score, ignore_zero_llm=True)
        if not np.isnan(regauc):
            all_regauc.append(regauc)

        # --- Hit@50 by score distribution (ignore exact ID coverage) ---
        hit50, hit_high_50 = hit_rate_by_topk_distribution(
            llm_scores_all=llm_scores,
            model_scores_all=y_score,
            topk=50,
            max_score=max_score,
            ignore_zero_llm_for_opt=True,  # optimal top50 only from labeled pairs
        )
        if not np.isnan(hit50):
            all_hit50.append(hit50)
        if not np.isnan(hit_high_50):
            all_hit_high_50.append(hit_high_50)


        if (llm_scores >= 4).sum() > 0:
            ndcg10 = ndcg_at_k(llm_scores, y_score, 10)
            ndcg20 = ndcg_at_k(llm_scores, y_score, 20)
            ndcg50 = ndcg_at_k(llm_scores, y_score, 50)
            if not np.isnan(ndcg10):
                all_ndcg10.append(ndcg10)
            if not np.isnan(ndcg20):
                all_ndcg20.append(ndcg20)
            if not np.isnan(ndcg50):
                all_ndcg50.append(ndcg50)

        # Sort by model score DESC (robust even if input already sorted)
        sorted_idx = np.argsort(-y_score)

        def top_k_indices(k: int):
            return sorted_idx[: min(k, len(sorted_idx))]

        # Top@K recall / precision on binary labels
        for k, recall_list, prec_list in zip(
            [10, 20, 50],
            [all_top10_recall, all_top20_recall, all_top50_recall],
            [all_top10_prec, all_top20_prec, all_top50_prec],
        ):
            idx = top_k_indices(k)
            top_relevant = y_true[idx].sum()
            recall = top_relevant / y_true.sum()
            precision = top_relevant / len(idx)
            recall_list.append(float(recall))
            prec_list.append(float(precision))

        # score_sample.csv (top20 by model score)
        top20_idx = top_k_indices(20)
        top20_music_ids = [music_ids[i] for i in top20_idx]
        top20_model_scores = [y_score[i] for i in top20_idx]
        top20_llm_scores = [
            float(llm_dict[(item_id, mid)].score) if (item_id, mid) in llm_dict else 0.0
            for mid in top20_music_ids
        ]

        score_samples.append(
            {
                "item_id": item_id,
                "top20_music_ids": " ".join(map(str, top20_music_ids)),
                "top20_llm_scores": " ".join(f"{s:.3f}" for s in top20_llm_scores),
                "top20_model_scores": " ".join(f"{s:.6f}" for s in top20_model_scores),
                "auc_binary": float(auc) if not np.isnan(auc) else np.nan,
                "regauc_cindex": float(regauc) if not np.isnan(regauc) else np.nan,
                "hit50_dist": float(hit50) if not np.isnan(hit50) else np.nan,
            }
        )
        item_metrics.append(
            {
                "item_id": int(item_id),
                "auc_binary": float(auc) if not np.isnan(auc) else np.nan,
                "regauc_cindex": float(regauc) if not np.isnan(regauc) else np.nan,
                "hit50_dist": float(hit50) if not np.isnan(hit50) else np.nan,
                "top50_precision": float(all_top50_prec[-1]),
            }
        )

    # 4) Summary
    print("\n===== Evaluation Results =====")
    print(f"有效视频数(至少1个二值正例): {len(all_auc)}")
    print(f"平均 AUC(binary): {np.nanmean(all_auc):.4f}")

    if len(all_regauc) > 0:
        print(f"平均 regAUC(C-index, raw LLM score): {np.nanmean(all_regauc):.4f}")
    else:
        print("平均 regAUC(C-index): NaN (可能是 LLM 标注太稀疏或全为同分)")

    if len(all_hit50) > 0:
        print(f"平均 Hit@50(分布命中率): {np.nanmean(all_hit50):.4f}")
    else:
        print("平均 Hit@50(分布命中率): NaN (可能是 LLM 标注太稀疏)")

    if len(all_hit_high_50) > 0:
        print(f"平均 Hit_High@50(高分召回率): {np.nanmean(all_hit_high_50):.4f}")
    else:
        print("平均 Hit_High@50(高分召回率): NaN (可能是 LLM 标注太稀疏或全为低分)")

    if len(all_ndcg10) > 0:
        print("\nTop@K NDCG (至少1个>=4分):")
        print(f"  NDCG@10: {np.nanmean(all_ndcg10):.4f}")
        print(f"  NDCG@20: {np.nanmean(all_ndcg20):.4f}")
        print(f"  NDCG@50: {np.nanmean(all_ndcg50):.4f}")

    print("\nTop@K 召回率:")
    print(f"  Top10 Recall: {np.mean(all_top10_recall):.4f}")
    print(f"  Top20 Recall: {np.mean(all_top20_recall):.4f}")
    print(f"  Top50 Recall: {np.mean(all_top50_recall):.4f}")

    print("\nTop@K 准确率:")
    print(f"  Top10 Precision: {np.mean(all_top10_prec):.4f}")
    print(f"  Top20 Precision: {np.mean(all_top20_prec):.4f}")
    print(f"  Top50 Precision: {np.mean(all_top50_prec):.4f}")

    # 5) Save score_sample.csv
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "score_sample.csv")
    pd.DataFrame(score_samples).to_csv(output_file, index=False)
    print(f"\n✅ 已保存 Top20 样本 + 指标到: {output_file}")
    if tag_path:
        tag_df = load_tag_df(tag_path)
        merged = pd.DataFrame(item_metrics).merge(tag_df, on="item_id", how="inner")
        per_tag = (
            merged.groupby("tag")
            .agg(
                regauc_cindex=("regauc_cindex", "mean"),
                hit50_dist=("hit50_dist", "mean"),
                item_count=("item_id", "count"),
            )
            .reset_index()
        )
        per_tag = per_tag.sort_values("regauc_cindex", ascending=False)
        per_tag_path = os.path.join(output_path, "per_tag_metrics.csv")
        per_tag.to_csv(per_tag_path, index=False)
        print(f"✅ 已保存按标签聚合指标到: {per_tag_path}")
    else:
        item_metrics_path = os.path.join(output_path, "item_metrics.csv")
        pd.DataFrame(item_metrics).to_csv(item_metrics_path, index=False)
        print(f"✅ 已保存 item 级指标到: {item_metrics_path}")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True, help="Base model dir; eval files under data_path/evalset_1w_data/")
    p.add_argument("--max_score", type=int, default=5, choices=[5, 10], help="LLM score scale")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # LLM judged tsv
    llm_data = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_all_llm_judge_data_v5.tsv"
    print(f"llm score data: {llm_data}")
    tag_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/evalset_1k_video_data_tags.jsonl"

    data_path = os.path.join(args.data_path, "evalset_1w_data/")
    sim_csv = os.path.join(data_path, "video_to_1w_music_similarity.csv")

    evaluate_similarity_result(
        sim_csv_path=sim_csv,
        llm_tsv_path=llm_data,
        output_path=data_path,
        max_score=args.max_score,
        tag_path=tag_path,
    )
