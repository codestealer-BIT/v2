#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="统计 recall_results 中 id 的平均出现行数"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="输入 CSV 文件名"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="每行只统计 recall_results 的前 N 个 id（默认不限制）"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    id_row_count = defaultdict(int)  # id -> 出现过的行数
    total_rows = 0

    with open(args.csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            recall_results = row["recall_results"].strip()
            if not recall_results:
                continue

            ids = recall_results.split()
            if args.top_n is not None:
                ids = ids[:args.top_n]

            # 每行 id 不重复，直接计数即可
            for _id in ids:
                id_row_count[_id] += 1

    if not id_row_count:
        print("没有统计到任何 id")
        return

    total_id_occurrences = sum(id_row_count.values())
    unique_id_count = len(id_row_count)

    avg_occurrence = total_id_occurrences / unique_id_count

    print(f"总行数: {total_rows}")
    print(f"唯一 id 数量: {unique_id_count}")
    print(f"所有 id 累计出现行数: {total_id_occurrences}")
    print(f"平均每个 id 出现行数: {avg_occurrence:.4f}")

if __name__ == "__main__":
    main()
