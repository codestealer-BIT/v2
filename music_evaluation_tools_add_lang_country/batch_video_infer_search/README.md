# V2V Faiss 建库和检索

这个目录主要用于做视频向量的离线抽取、Faiss 精确索引构建，以及基于查询向量的近邻检索。

如果只看主流程，可以理解为 3 步：

1. 用 `extract_video_embeds_from_train_data.py` 把视频数据跑成 embedding chunk。
2. 用 `faiss_exact_build.py` 把 embedding 建成 Faiss 索引，并生成 `id_map.csv`。
3. 用 `faiss_exact_search.py` 对 query embedding 做检索，输出召回结果。

## 目录内脚本说明

### 1. `extract_video_embeds_from_train_data.py`

作用：多卡推理抽取训练/候选视频 embedding，写成分块文件。

输入：

- 模型目录和 checkpoint
- 视频数据标注 `data_path`
- tokenizer / 数据加载相关配置

输出：

- `embeddings_rank_{rank}_chunk_{chunk_idx}.npy`
- `metadata_rank_{rank}_chunk_{chunk_idx}.csv`

其中每个 `metadata` 文件默认包含：

- `idx`: 当前 chunk 内的行号
- `music_id`
- `item_id`
- `score`

这个脚本本身不做 Faiss 检索，它的职责是先把候选库的视频 embedding 准备好，供后续 build 使用。

### 2. `run_extract_train_video_embeds.sh`

作用：`extract_video_embeds_from_train_data.py` 的一个实际运行脚本示例。

它主要做了几件事：

- 指定模型路径 `MODEL_DIR`
- 指定数据路径 `DATA_PATH`
- 指定 embedding 输出目录 `OUTPUT_DIR`
- 用 `torchrun --nproc_per_node=8` 启动多卡推理

通常你要抽新的候选库 embedding，可以先参考这个脚本改路径和参数。

### 3. `faiss_exact_build.py`

作用：把候选视频 embedding 建成 Faiss 精确检索索引，同时生成一份 `global_idx -> metadata` 的映射表。

当前实现使用的是：

- `faiss.IndexFlatIP`

也就是 **精确内积检索**。如果加上 `--normalize`，则相当于对向量做 L2 归一化后再算内积，此时可以近似理解为 **cosine similarity 检索**。

这个脚本支持两种建库输入方式。

#### 方式 A：从本目录生成的 embedding chunk 建库

输入：

- `--embeds_dir`
- `--embeds_glob`，默认匹配 `embeddings_rank_*_chunk_*.npy`

脚本会：

1. 找到所有 embedding chunk
2. 为每个 chunk 找到对应的 `metadata_rank_*_chunk_*.csv`
3. 把所有 embedding 拼接起来
4. 构建 `IndexFlatIP`
5. 写出 index 文件
6. 写出 `id_map.csv`

`id_map.csv` 的第一列固定是：

- `global_idx`

后面会跟上 metadata 里的字段，默认一般是：

- `music_id`
- `item_id`
- `score`

这里的 `global_idx` 就是 Faiss 返回结果里的向量下标，后续 search 结果会通过它回查业务字段。

#### 方式 B：直接从 hdfs parquet 建库 （不用下载下来，直接远程建库）

输入：

- `--parquet_dir`
- `--parquet_emb_col`
- `--parquet_batch_size`

适合 embedding 已经存成 parquet 的情况。脚本会流式读取 parquet batch，边读边 `index.add(...)`，同时把除 embedding 列以外的列全部写入 `id_map.csv`。

如果数据量很大，这种方式比一次性把所有 `.npy` 拼到内存里更稳。

#### 常用参数

- `--index_out`: 输出 Faiss index 路径
- `--id_map_out`: 输出 id 映射表路径
- `--normalize`: 建库前先做 L2 归一化
- `--count_rows`: parquet 模式下先统计总行数，方便看进度

#### 示例

```bash
export EMB_PATH=/path/to/checkpoint

python music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_build.py \
  --embeds_dir "$EMB_PATH/220w_4_5_only" \
  --index_out "$EMB_PATH/220w_4n5_faiss_index_flatip.index" \
  --id_map_out "$EMB_PATH/220w_4n5_id_map.csv" \
  --normalize
```

生成结果通常包括：

- `*.index`: Faiss 索引文件
- `*_id_map.csv`: Faiss 行号到业务字段的映射

### 4. `map_sub_emb_from_meta.py`

作用：从一个已有的大索引里，按子集 metadata 抽出对应向量，生成 query embedding。

这个脚本常用于：

- 已经有一个候选大库和对应 Faiss index
- 想从其中选一批样本作为 query 集合
- 不想重新走模型推理

它的做法是：

1. 读取大库的 `id_map.csv`
2. 读取 `subset_path`
3. 按 `global_idx` 或某个 `key`（例如 `item_id`）找到对应向量
4. 调用 `index.reconstruct(i)` 从 Faiss index 中恢复向量
5. 输出：
   - `query_out`：`query.npy`
   - `query_meta_out`：query 对应的 metadata

也就是说，它是一个“从已有 index 里反查出 query embedding”的辅助工具。

### 5. `faiss_exact_search.py`

作用：读取 Faiss index 和 query embedding，执行批量 TopK 检索，并把结果写成 `jsonl`。

输入：

- `--index_path`: build 阶段生成的 `.index`
- `--id_map_path`: build 阶段生成的 `id_map.csv`
- `--query_path`: 查询向量 `.npy`
- `--query_meta_path`: 可选，query 的 metadata

输出：

- `--output_path`: 检索结果 `jsonl`

检索时的核心逻辑：

1. 读入 Faiss index
2. 读入 query 向量
3. 如果传了 `--normalize`，对 query 做 L2 归一化
4. 按 `--batch_size` 分批执行 `index.search(q, topk)`
5. 把 Faiss 返回的 `I` / `D` 结合 `id_map.csv` 还原成可读结果
6. 写成逐行 JSON

其中：

- `I` 是命中的候选 `global_idx`
- `D` 是相似度分数

如果 build 和 search 两边都用了 `--normalize`，则 `sim` 可以按 cosine 相似度来理解。

#### 输出字段说明

结果 `jsonl` 中常见字段包括：

- `query_id`: query 在 `query.npy` 中的行号
- `rank`: 当前召回名次，从 0 开始
- `sim`: 相似度分数
- `global_idx`: 命中的候选向量下标
- `music_id` / `item_id` / `score`: 来自 `id_map.csv`
- `query_item_id` / `query_tag`: 如果提供了 `query_meta_path`，则会把 query 侧信息也带出来

#### 示例

```bash
export EMB_PATH=/path/to/checkpoint

python -u music_evaluation_tools_add_lang_country/batch_video_infer_search/faiss_exact_search.py \
  --index_path "$EMB_PATH/220w_4n5_faiss_index_flatip.index" \
  --id_map_path "$EMB_PATH/220w_4n5_id_map.csv" \
  --query_meta_path /path/to/query_meta.csv \
  --query_path /path/to/query.npy \
  --normalize \
  --topk 500 \
  --output_path /path/to/search_results.jsonl
```

#### 常用参数

- `--topk`: 每个 query 返回多少个候选
- `--batch_size`: 检索批大小，默认 `4096`
- `--normalize`: 对 query 先归一化

### 6. `extract_top5_video_embeds.py`

作用：从检索结果里取每个 query 的 TopK 视频，再回原始 embedding chunk 中把这些视频的 embedding 导出来。

它适合做：

- case study
- 可视化分析
- 后续 rerank / 聚类 / 评估

处理逻辑大致是：

1. 读取 `faiss_exact_search.py` 输出的 `jsonl`
2. 对每个 `query_item_id` 按 `sim` 排序
3. 取前 `topk` 个去重后的 `item_id`
4. 在 `metadata_rank_*_chunk_*.csv` 里扫描这些 `item_id`
5. 从对应的 `embeddings_rank_*_chunk_*.npy` 中取出向量
6. 输出 `top5_itemid_to_embed.npz`

## 使用顺序

如果你要完整跑一遍目录里的能力，通常顺序如下：

```text
extract_video_embeds_from_train_data.py
    -> 生成 embeddings_rank_*.npy + metadata_rank_*.csv

faiss_exact_build.py
    -> 生成 *.index + *_id_map.csv

map_sub_emb_from_meta.py
    -> 可选，从已有索引中抽 query.npy + query_meta.csv

faiss_exact_search.py
    -> 生成 search_results.jsonl

extract_top5_video_embeds.py
    -> 可选，导出 topK 命中视频的 embedding
```

