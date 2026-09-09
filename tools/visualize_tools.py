# import json
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")   # Use non-GUI backend
# import matplotlib.pyplot as plt

# # Path to your log file
# log_path = "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_large_dataset_formal_adjust_ratio/log.txt"

# # Read logs into a list of dicts
# records = []
# with open(log_path, "r") as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             try:
#                 records.append(json.loads(line))
#             except json.JSONDecodeError:
#                 print("Skipping invalid line:", line[:100])

# # Convert to DataFrame
# df = pd.DataFrame(records)


# # # Path to your log file
# # log_path = "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_short_unfreeze/log.txt"

# # # Read logs into a list of dicts
# # records = []
# # with open(log_path, "r") as f:
# #     for line in f:
# #         line = line.strip()
# #         if line:
# #             try:
# #                 records.append(json.loads(line))
# #             except json.JSONDecodeError:
# #                 print("Skipping invalid line:", line[:100])

# # # Convert to DataFrame
# # df_ours = pd.DataFrame(records)

# # Show available columns (for debugging)
# print("Available metrics:", df.columns.tolist())

# # === Plot 1: Training loss ===
# plt.figure(figsize=(8, 5))
# # plt.plot(df["epoch"], df["train_loss"], label="internvid_train_loss")
# # plt.plot(df_ours["epoch"], df_ours["train_loss"], label="ours_train_loss")
# if "train_short_text_loss" in df:
#     plt.plot(df["epoch"], df["train_short_text_loss"], label="short text loss")
# if "train_long_text_loss" in df:
#     plt.plot(df["epoch"], df["train_long_text_loss"], label="long text loss")

# # if "train_video_text_loss" in df:
# #     plt.plot(df["epoch"], df["train_video_text_loss"], label="t2v_loss")
# # if "train_fused_text_loss" in df:
# #     plt.plot(df["epoch"], df["train_fused_text_loss"], label="t2mm_loss")
# # if "train_lm_loss" in df:
# #     plt.plot(df["epoch"], df["train_lm_loss"], label="lm_loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig("/opt/tiger/MMPretrain/vis/train_loss.png")

# # === Plot 2: Learning rate schedule ===
# plt.figure(figsize=(8, 5))
# plt.plot(df["epoch"], df["train_lr"], label="learning rate")
# plt.plot(df["epoch"], df["train_min_lr"], label="min lr", linestyle="--")
# plt.xlabel("Epoch")
# plt.ylabel("Learning Rate")
# plt.title("Learning Rate Schedule")
# plt.legend()
# plt.grid(True)
# plt.savefig("/opt/tiger/MMPretrain/vis/learning_rate.png")

# # === Plot 3: Gradient norm ===
# if "train_grad_norm" in df:
#     plt.figure(figsize=(8, 5))
#     plt.plot(df["epoch"], df["train_grad_norm"], label="grad norm")
#     plt.xlabel("Epoch")
#     plt.ylabel("Gradient Norm")
#     plt.title("Gradient Norm Curve")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("/opt/tiger/MMPretrain/vis/grad_norm.png")

# # === Plot 4: Weight decay ===
# if "train_weight_decay" in df:
#     plt.figure(figsize=(8, 5))
#     plt.plot(df["epoch"], df["train_weight_decay"], label="weight decay")
#     plt.xlabel("Epoch")
#     plt.ylabel("Weight Decay")
#     plt.title("Weight Decay Over Training")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("/opt/tiger/MMPretrain/vis/weight_decay.png")

# print("✅ Saved plots: train_loss.png, learning_rate.png, grad_norm.png, weight_decay.png")

import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt

# # Path to your validation log
# val_log_path = "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_large_dataset_formal_adjust_ratio/val_tiktok_log.txt"

# # Read logs
# records = []
# with open(val_log_path, "r") as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             try:
#                 records.append(json.loads(line))
#             except json.JSONDecodeError:
#                 print("Skipping invalid line:", line[:100])

# # Convert to DataFrame
# df = pd.DataFrame(records)

# # Use line number as "epoch"
# df["epoch"] = range(1, len(df) + 1)

# # Plot Recall@1
# plt.figure(figsize=(8,5))
# if 't2v_retrieval_recall@1' in df:
#     plt.plot(df['epoch'], df['t2v_retrieval_recall@1'], label='Text->Video Recall@1')
# if 'v2t_retrieval_recall@1' in df:
#     plt.plot(df['epoch'], df['v2t_retrieval_recall@1'], label='Video->Text Recall@1')

# plt.xlabel("Epoch")
# plt.ylabel("Recall@1")
# plt.title("30k TikTok Dataset Recall@1 Curve")
# plt.legend()
# plt.grid(True)
# plt.savefig("/opt/tiger/MMPretrain/vis/tiktok_recall_at_1.png")

# print("✅ Saved plot: tiktok_recall_at_1.png")



# Path to your validation log
val_log_path = "/mnt/bn/yexiaoyu-test/checkpoints/stage_1_final_no_frame_mask/val_log.txt"

# Read logs
records = []
with open(val_log_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping invalid line:", line[:100])

# Convert to DataFrame
df = pd.DataFrame(records)

# Use line number as "epoch"
df["epoch"] = range(1, len(df) + 1)

# Plot Recall@1
plt.figure(figsize=(8,5))
if 't2v_retrieval_recall@1' in df:
    plt.plot(df['epoch'], df['t2v_retrieval_recall@1'], label='Text->Video Recall@1')
if 'v2t_retrieval_recall@1' in df:
    plt.plot(df['epoch'], df['v2t_retrieval_recall@1'], label='Video->Text Recall@1')

plt.xlabel("Epoch")
plt.ylabel("Recall@1")
plt.title("MSR_VTT Dataset Recall@1 Curve")
plt.legend()
plt.grid(True)
plt.savefig("/opt/tiger/MMPretrain/vis/recall_at_1.png")

print("✅ Saved plot: recall_at_1.png")


