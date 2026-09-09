import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Example: read data_path from command line")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory or file')
    args = parser.parse_args()
    return args

def evaluate_similarity_result(sim_csv_path, llm_tsv_path, output_path):
    # 1️⃣ 读取 LLM 打分数据
    llm_df = pd.read_csv(llm_tsv_path, sep='\t', names=['item_id', 'music_id', 'score'])
    llm_df['relevant'] = (llm_df['score'] >= 7).astype(int)
    llm_dict = {(int(row.item_id), int(row.music_id)): row for row in llm_df.itertuples(index=False)}

    print(f"Loaded {len(llm_df)} LLM judged pairs.")

    # 2️⃣ 读取模型相似度输出
    results = []
    with open(sim_csv_path, 'r') as fr:
        for line in tqdm(fr, desc="Processing similarity results"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            item_id = parts[0]
            music_ids = parts[1].split()
            sim_scores = [float(x) for x in parts[2].split()]
            results.append((item_id, music_ids, sim_scores))

    print(f"Loaded {len(results)} video similarity lines.")

    # 3️⃣ 对每个视频计算指标 + 生成score_sample输出
    all_auc = []
    all_top10_recall, all_top20_recall, all_top50_recall = [], [], []
    all_top10_prec, all_top20_prec, all_top50_prec = [], [], []

    score_samples = []  # 🔹 新增：存储每个 item_id 的top20信息

    for item_id, music_ids, sim_scores in results:
        # 构建 ground truth 标签
        labels = [llm_dict.get((int(item_id), int(mid)), None) for mid in music_ids]
        y_true = np.array([int(x.relevant) if x is not None else 0 for x in labels])
        y_score = np.array(sim_scores)

        # 跳过没有正样本的情况
        if y_true.sum() == 0:
            continue

        # AUC
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan

        # 排序索引（按模型分数降序）
        sorted_idx = np.argsort(-y_score)

        def top_k_indices(k):
            return sorted_idx[:min(k, len(sorted_idx))]

        # 计算 Top@K recall / precision
        for k, recall_list, prec_list in zip(
            [10, 20, 50],
            [all_top10_recall, all_top20_recall, all_top50_recall],
            [all_top10_prec, all_top20_prec, all_top50_prec]
        ):
            idx = top_k_indices(k)
            top_relevant = y_true[idx].sum()
            recall = top_relevant / y_true.sum()
            precision = top_relevant / len(idx)
            recall_list.append(recall)
            prec_list.append(precision)

        all_auc.append(auc)

        # 🔹 新增：生成 score_sample.csv 所需内容
        top20_idx = top_k_indices(20)
        top20_music_ids = [music_ids[i] for i in top20_idx]
        top20_model_scores = [y_score[i] for i in top20_idx]
        top20_llm_scores = [
            llm_dict.get((int(item_id), int(mid))).score if llm_dict.get((int(item_id), int(mid))) else 0.0
            for mid in top20_music_ids
        ]

        score_samples.append({
            "item_id": item_id,
            "top20_music_ids": " ".join(map(str, top20_music_ids)),
            "top20_llm_scores": " ".join(f"{s:.3f}" for s in top20_llm_scores),
            "top20_model_scores": " ".join(f"{s:.3f}" for s in top20_model_scores)
        })

    # 4️⃣ 汇总输出结果
    print("\n===== Evaluation Results =====")
    print(f"有效视频数: {len(all_auc)}")
    print(f"平均AUC: {np.nanmean(all_auc):.4f}\n")

    print("Top@K 召回率:")
    print(f"  Top10 Recall: {np.mean(all_top10_recall):.4f}")
    print(f"  Top20 Recall: {np.mean(all_top20_recall):.4f}")
    print(f"  Top50 Recall: {np.mean(all_top50_recall):.4f}\n")

    print("Top@K 准确率:")
    print(f"  Top10 Precision: {np.mean(all_top10_prec):.4f}")
    print(f"  Top20 Precision: {np.mean(all_top20_prec):.4f}")
    print(f"  Top50 Precision: {np.mean(all_top50_prec):.4f}")

    # 🔹 新增：输出 score_sample.csv
    output_file = os.path.join(output_path, "score_sample.csv")
    score_df = pd.DataFrame(score_samples)
    score_df.to_csv(output_file, index=False)
    print(f"\n✅ 已保存 Top20 打分样本到: {output_file}")

if __name__ == "__main__":
    args = parse_args()
    llm_data = "/mnt/bn/tt-search-ads-nas/wangxiuqi.0601/data/evalset_all_llm_judge_data.tsv"
    data_path = os.path.join(args.data_path, "evalset_1w_data/")
    evaluate_similarity_result(
        sim_csv_path=os.path.join(data_path, "video_to_1w_music_similarity.csv"),
        llm_tsv_path=llm_data,
        output_path=data_path   # 🔹 新增：输出路径传入
    )
