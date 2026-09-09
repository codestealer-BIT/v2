import os
import json
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

df = pd.read_csv("/mnt/bn/jiny-ttls-i18n-fr1q/lutong/codebases/xiuqi/VideoSiglip2-music/LT/llm_prompt_gt.csv")
target_mids = list(set(df.music_id))
print(f"Found {len(target_mids)} target music ids")

def find_music_records(root_dirs, targets, id_key='clip_id'):
    files = []
    files = [ os.path.join(root_dirs, f) for f in os.listdir(root_dirs) if f.endswith('.json')]
    print(f"Found {len(files)} music parts to search")

    results = {}
    remaining = set(targets)

    def worker(path):
        found = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line or id_key not in line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                mid = int(obj.get(id_key))
                if mid in remaining:
                    print(f"Found {mid} in {path}!")
                    found[mid] = obj
        return found

    max_workers = min(len(files) or 1, (os.cpu_count() or 2) * 4, 64)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker, p): p for p in files}
        pbar = tqdm(total=len(futs), desc='Seaching through music part files', unit='file')
        for fut in as_completed(futs):
            found = fut.result()
            if found:
                for k, v in found.items():
                    if k not in results:
                        results[k] = v
                        remaining.discard(k)
            pbar.set_postfix({'remaining': len(remaining)})
            pbar.update(1)
            if not remaining:
                break
        pbar.close()
    return results

root_dirs = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/all_song_candidates_1216_7600w/"
records = find_music_records(root_dirs, target_mids)
print(f'Found {len(records)} of {len(target_mids)}')
import pandas as pd; df = pd.DataFrame(records.values())
with open('found_mid_info.jsonl', 'w', encoding='utf-8') as fw:
    for rec in records.values():
        fw.write(json.dumps(rec, ensure_ascii=False) + '\n')