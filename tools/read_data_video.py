import json
import random
import subprocess
from pathlib import Path
from collections import Counter


# =========================
# 配置
# =========================

LOCAL_VIDEO_DIR = Path("/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/video")

HDFS_SCORE_FILTERED_DIR = (
    "hdfs://harunafr/home/byte_video_rec_music/proj/content_understanding/"
    "merged_train_data_v5/20260331_5000w_n_300w_high_5100w_remove_score_1_2_1000parts"
)

# 第一步：把上面 HDFS 目录下的 part 文件列表保存到本地 JSON
LOCAL_SCORE_FILTERED_JSON = (
    LOCAL_VIDEO_DIR
    / "fr_train_data_pgc_v5_score_20260331_5000w_n_300w_high_5100w_remove_score_1_2_1000parts.json"
)

# 已有两个路径列表 JSON
NEG_1080W_JSON = (
    LOCAL_VIDEO_DIR
    / "fr_train_data_pgc_v5_score_20260425_50m_neg_1080w.json"
)

NEG_300W_JSON = (
    LOCAL_VIDEO_DIR
    / "fr_train_data_pgc_v5_score_20260416_300w_neg_300w.json"
)

# 最终混合后的输出
OUTPUT_MIXED_JSON = (
    LOCAL_VIDEO_DIR
    / "fr_train_data_pgc_v5_score_20260507_64m_remove_score_1_2_mixed.json"
)

SEED = 42
DEDUP = True
SHUFFLE = True


# =========================
# 工具函数
# =========================

def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        print("[STDOUT]")
        print(p.stdout)
        print("[STDERR]")
        print(p.stderr)
        raise RuntimeError(f"Command failed with return code {p.returncode}: {' '.join(cmd)}")
    return p.stdout


def list_leaf_files_hdfs(root):
    """
    递归列出 HDFS root 下的数据文件。
    过滤掉 _SUCCESS、_temporary、.caption 等非数据文件。
    """
    root = root.strip()

    out = run_cmd(["hdfs", "dfs", "-ls", "-R", root])

    files = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue

        # hdfs dfs -ls 输出一般类似：
        # -rw-r--r--   3 hdfs supergroup 123 2026-xx-xx xx:xx hdfs://.../part-xxx.snappy
        parts = line.split()
        if len(parts) < 8:
            continue

        perm = parts[0]
        path = parts[-1]

        # 只要文件，不要目录
        if not perm.startswith("-"):
            continue

        name = path.rsplit("/", 1)[-1]

        if name == "_SUCCESS":
            continue
        if "_temporary" in path:
            continue
        if name.endswith(".caption"):
            continue

        # 一般只保留 part 文件更安全
        if not name.startswith("part-"):
            continue

        files.append(path)

    files = sorted(files)
    return files


def save_json_list(items, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {path}")
    print(f"[SAVE] num_items = {len(items)}")


def load_json_list(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON, got {type(data)} from {path}")

    return data


def print_parent_dir_stats(paths, title):
    counter = Counter()
    for p in paths:
        parent = p.rsplit("/", 1)[0]
        dirname = parent.rsplit("/", 1)[-1]
        counter[dirname] += 1

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("total file paths:", len(paths))
    print("unique dataset dirs:", len(counter))
    print()

    for dirname, cnt in counter.most_common():
        print(cnt, dirname)


def merge_json_lists(json_paths, output_path, dedup=True, shuffle=True, seed=42):
    all_items = []

    for p in json_paths:
        items = load_json_list(p)
        print(f"[LOAD] {p}: {len(items)}")
        all_items.extend(items)

    print("[MERGE] before dedup:", len(all_items))

    if dedup:
        # 保持首次出现顺序去重
        all_items = list(dict.fromkeys(all_items))
        print("[MERGE] after dedup:", len(all_items))

    if shuffle:
        random.seed(seed)
        random.shuffle(all_items)
        print("[MERGE] shuffled with seed:", seed)

    save_json_list(all_items, output_path)
    print_parent_dir_stats(all_items, "final mixed json parent dir stats")

    return all_items


# =========================
# 主流程
# =========================

def main():
    # 1. 扫描 HDFS filtered 目录，生成本地 JSON 路径列表
    filtered_files = list_leaf_files_hdfs(HDFS_SCORE_FILTERED_DIR)

    if len(filtered_files) == 0:
        raise RuntimeError(f"No part files found under {HDFS_SCORE_FILTERED_DIR}")

    print_parent_dir_stats(filtered_files, "filtered 20260331 path list stats")
    save_json_list(filtered_files, LOCAL_SCORE_FILTERED_JSON)

    # 2. 混合三个 JSON 路径列表
    json_paths = [
        LOCAL_SCORE_FILTERED_JSON,
        NEG_1080W_JSON,
        NEG_300W_JSON,
    ]

    mixed = merge_json_lists(
        json_paths=json_paths,
        output_path=OUTPUT_MIXED_JSON,
        dedup=DEDUP,
        shuffle=SHUFFLE,
        seed=SEED,
    )

    print()
    print("[DONE]")
    print("filtered json:", LOCAL_SCORE_FILTERED_JSON)
    print("mixed json:", OUTPUT_MIXED_JSON)
    print("mixed file paths:", len(mixed))


if __name__ == "__main__":
    main()