import numpy as np
import os
import pandas as pd
from tqdm import tqdm

chunk_size = int(1e5)
world_size = 8
numpy_dir = '/mnt/bn/yexiaoyu-test/data/faiss'
output_dir = '/mnt/bn/jiny-ttls-i18n-fr1q/yexiaoyu/plain_embeddings'

all_embeddings = []
for r in range(world_size):
    rank_embeddings = np.load(os.path.join(numpy_dir, f"embeddings_rank{r}.npy"))
    all_embeddings.append(rank_embeddings)
all_embeddings = np.concatenate(all_embeddings, axis=0)

all_item_ids = []
for r in range(world_size):
    rank_item_ids = np.load(os.path.join(numpy_dir, f"item_ids_rank{r}.npy"))
    all_item_ids.append(rank_item_ids)
all_item_ids = np.concatenate(all_item_ids, axis=0)

all_item_ids, unique_indices = np.unique(all_item_ids, return_index=True)
all_embeddings = all_embeddings[unique_indices]
print("item_ids:", all_item_ids[:5])

id_count = all_embeddings.shape[0]
print("total ids:", id_count)

df_all = pd.DataFrame({
        "item_id": all_item_ids,
        "embedding": list(all_embeddings)  # 转为列表，使每个元素作为独立数组存入DataFrame
    })

df_all.to_parquet(os.path.join(output_dir,f"embedding_without_caption.parquet"), index=False)