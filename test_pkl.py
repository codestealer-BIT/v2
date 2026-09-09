import pickle, json

pkl_path = "/mnt/bn/jiny-ttls-i18n-fr1q/lutong/data/music/music_train_20260301_1336w.pkl"

music_id = 7324088047211907845  # 换成你样本里的 music_id

with open(pkl_path, "rb") as f:
    d = pickle.load(f)

print("music_id in pkl:", music_id in d)
print("meta file:", d.get(music_id))

if music_id in d:
    with open(d[music_id], "r") as f:
        meta = json.load(f)
    print(meta.keys())

    for k in ["content", "content_vector", "title_vector", "cover_vector", "lyric_vector"]:
        print(k, k in meta, type(meta.get(k)))