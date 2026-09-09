## 1. 训练模型
### 准备数据
1. 用 spark 任务生成好需要的数据，写到 hdfs 中
2. 用 [这个notebook](tools/read_data_video.ipynb) 生成训练用的 video 目录 和 音乐 pkl （如果音乐有变化的话）
### 模型训练
 - 所有的模型具体结构都在 video_clip 目录下，训练入口在 scripts 目录下
    - regression 模型
        - 0129模型
            - 模型 video_clip/video_siglip2_for_music_only_matching.py
            - script：scripts/train_video_music_match_original.sh
    - 对比学习
        - 最新（简化版）： 
            - 模型 video_clip/video_siglip2_for_music_only_matching_contrastive.py
            - script：scripts/train_video_music_match_contrastive.sh
        - 0104 模型： 
            - 模型 video_clip/video_siglip2_for_music_negative_filtering_add_lang.py
            - script：scripts/train_video_siglip2_for_music_add_lang_1029_dis_gate.sh
- 训练指令示例
    ```bash
    bash scripts/train_video_music_match_original.sh 2>&1 | tee xxx/logs/51m_original_$(date +%Y%m%d_%H%M%S)_p${ARNOLD_ID}.log
    ```
    或者直接在正式任务里把script当作入口指令运行，比如 [这个任务](https://ml.tiktok-row.net/development/instance/jobs/63dcc121908a691e?tabState=task_config&trialId=20188264)
### 数据入口
数据读取和处理在这个script里：dataset/video_music_dataset.py
- 如果要修改音乐数据，只要把`all_music_caption_dict`的pkl路径改掉即可 (line142)

## 2. 评估模型 （auc、hit率等指标）
```bash
bash run_evalset_lang.sh
```
注意跑之前先确定:
1. `music_evaluation_tools_add_lang_country/encode_music_embedding.py` 
    1. `build_model` 里面是你想要评估的模型
    2. 【重要】 在 `if __name__ == '__main__':` 里面 uncomment 掉 `1. evalset video & music` 这部分代码，把2和3comment掉
2. `music_evaluation_tools_add_lang_country/calculate_recall.py`
    - 【重要】  在 `if __name__ == '__main__':` 里面 uncomment 掉 `3.auc benchmark` 这部分代码， 确保1、2、4 comment掉
跑的结果会自动存到ckpt/case_summary目录下

## 3. 批量生成music embedding
当需要部署全量音乐ue的时候，我们需要刷好全量曲库的music embedding，上传到hdfs中，然后可以给志高用来搭建召回库
1. 确认音乐数据ready，比如现在最新的是这个8000w的数据 `/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_song_candidates_8000w`
    - 如果需要更换其他数据 ，需要在 `music_evaluation_tools_add_lang_country/encode_music_embedding_batch.py` 里面修改 `input_path`
    - 记得在line195更改对应的存储路径，default是存到每个ckpt folder里面的`200w_deploy`的新folder
2. 跑这个代码生成embedding，注意这个是可以distributed地跑的，即如果你有16卡两个机子，可以用16卡一起跑会快很多，你只需要分别在每个机子都跑一遍下面的指令就可以，他们会互相交流等待 （master_addr & master_port）
    先更新 CKPT_PATH 为你想要的模型路径
    ```bash
    bash music_evaluation_tools_add_lang_country/run_encode_music_650w.sh 2>&1 | tee xxx/logs/music_encode_650w.log
    ```

## 4. 看case
1. 先跑上面第3步，确认音乐embedding已经生产好了，现在看case一般用这个200w的召回库(default已经是了): `/mnt/bn/jiny-ttls-i18n-fr1q/wangxiuqi.0601/data/all_song_candidates_200w`
    这个会生成音乐embdding到ckpt folder里面的`200w_deploy`目录下
2. 看case的指令
    ```bash
    bash music_evaluation_tools_add_lang_country/check_cases_top50.sh 2>&1 | tee xxx/logs/check_cases_top50.log
    ```
    - 修改 MODEL_DIR 和 CHECKPOINTS （你可以一次性跑多个ckpt，只要他们都已经跑完第三步了）
    - 跑之前确保：
        1. `music_evaluation_tools_add_lang_country/encode_music_embedding.py`
            1. `build_model` 里面是你想要评估的模型
            2. 【重要】 在 `if __name__ == '__main__':` 里面 uncomment 掉 `3. check cases` 这部分代码，把1和2comment掉
        2. `music_evaluation_tools_add_lang_country/calculate_recall.py`
            - 【重要】  在 `if __name__ == '__main__':` 里面 uncomment 掉 `2.check case` 这部分代码， 确保1、3、4 comment掉
