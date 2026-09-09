import json
from sklearn.metrics import roc_auc_score

def load_scores(jsonl_path, label, use_llm_score=False):
    """读取 jsonl 文件，返回 (scores, labels)"""
    scores = []
    labels = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if use_llm_score:
                score_key = "prev_llm_score"  #"video_music_match_score" # # "panel_score" #"capsule_score" 
                if  obj.get(score_key) is None:
                    print(obj["item_id"], obj["music_id"])
                    continue
                scores.append(int(obj[score_key])) # 5 分制
                # scores.append(round(obj[score_key]//2)) # 10 分制
            else:
                scores.append(obj["predict_score"])
            labels.append(label)
    return scores, labels


def compute_auc(pos_file, neg_file, use_llm_score=False):
    # 读取正样本
    pos_scores, pos_labels = load_scores(pos_file, 1, use_llm_score)

    # 读取负样本
    neg_scores, neg_labels = load_scores(neg_file, 0, use_llm_score)

    # 合并
    scores = pos_scores + neg_scores
    labels = pos_labels + neg_labels

    # 计算 AUC
    auc = roc_auc_score(labels, scores)
    return auc

if __name__ == "__main__":
    pos_file = "auc_results/gate_v2_ct_drop_05_20_positive_llm_v5_pred/predictions_all.jsonl"
    neg_file = "auc_results/gate_v2_ct_drop_05_20_negative_llm_v5_pred/predictions_all.jsonl"

    # pos_file = "auc_results/llm_v4/1029_positive_pred_all_llm_v4.jsonl"
    # neg_file = "auc_results/llm_v4/1029_negative_pred_all_llm_v4.jsonl"

    # pos_file = "auc_results/llm_v1/1029_positive_pred_all_llm_v1_final.jsonl"
    # neg_file = "auc_results/llm_v1/1029_negative_pred_all_llm_v1.jsonl"

    auc = compute_auc(pos_file, neg_file, use_llm_score=False)
    print(f"AUC: {auc:.6f}")