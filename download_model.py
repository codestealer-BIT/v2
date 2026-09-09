from huggingface_hub import snapshot_download

# hdfs://harunasg/home/byte_data_tt_m/jinyang.leo
# hdfs://harunava/recommend/data/muse/jinyang.leo
# export HF_ENDPOINT=http://huggingface-proxy-sg.byted.org
# export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
# export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
# export https_proxy=http://sys-proxy-rd-relay.byted.org:8118


model_path = '/mnt/bn/jiny-ttls-i18n-fr1q/models/Qwen3-32B'   # The local directory to save downloaded checkpoint
snapshot_download("Qwen/Qwen3-32B", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')