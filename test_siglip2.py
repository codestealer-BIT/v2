import pandas as pd
import numpy as np

df_test = pd.read_parquet("/mnt/bn/yexiaoyu-test/data/faiss/fusion_embedding_chunk_0.parquet")
print(df_test["item_id"].dtype)  # 应输出int64
print(df_test["item_id"].iloc[0])
print(df_test["embedding"].iloc[0])  # 应输出int64
print(df_test["embedding"].iloc[0].shape)  # 应输出(768,)

# from transformers import AutoProcessor, AutoConfig
# import torch

# import re
# import random

# def clean_caption(caption: str) -> str:
    
#     # strip leading/trailing whitespace first
#     text = caption.strip()

#     # 1. remove leading 'A ' or 'An ' (case-insensitive, only at start)
#     text = re.sub(r'^(a|an)\s+', '', text, flags=re.IGNORECASE)

#     # 2. remove trailing '.' (only if it's the very last char)
#     text = re.sub(r'\.$', '', text)

#     # 3. with 50% chance, drop conjunctions and commas
#     if random.random() < 0.5:
#         # remove conjunction words surrounded by spaces
#         text = re.sub(r'\b(?:and|or|but|while|where|)\b', '', text, flags=re.IGNORECASE)
        
#         # remove commas
#         text = text.replace(',', '')

#         # clean up multiple spaces after removal
#         text = re.sub(r'\s+', ' ', text).strip()

#     return text

# for c in [
#     "A dog running in the park.",
#     "An apple, red and shiny.",
#     "A boy runnning while a man singing."
# ]:
#     print(clean_caption(c))

# import json
# from transformers import AutoTokenizer,AutoConfig
# from tqdm import tqdm
# import torch.nn as nn

# tokenizer =  AutoTokenizer.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
# config = AutoConfig.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
# token_embedding = nn.Embedding(config.text_config.vocab_size, 768)


# max_id = -1
# bad_lines = []
# with open("/mnt/bn/jiny-ttls-i18n-fr1q/vd-foundation___InternVid-10M-FLT/intervid_anno/InternVid-10M-FLT-INFO_with_path_long_caption.jsonl", "r", encoding="utf-8") as f:
#     for i, line in tqdm(enumerate(f)):
#         d = json.loads(line)
#         try:
#             text = d["long_caption"]
#             ids = tokenizer(text, return_tensors="pt")["input_ids"]
#             local_max = ids.max().item()
#             if local_max > max_id:
#                 max_id = local_max
#             output = token_embedding(ids)
            
#         except Exception as e:
#             print("found bad line: ",d['YoutubeID'], "\t", e)
#             bad_lines.append((i, d['YoutubeID']))
        

# print("===========================================")
# print("Global max token id:", max_id)
# print("Vocab size:", tokenizer.vocab_size)
# print("Lines with OOV tokens (if any):", bad_lines)




# proc = AutoProcessor.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/blip2")
# proc.num_query_tokens = 10
# print(proc.tokenizer.pad_token_id)
# import torch

# @torch.no_grad()
# def dwa_weights(losses):
#     hist = torch.tensor([2,0.5,1.0], dtype = torch.bfloat16)
#     hist = hist.to(losses[0].device)
#     w = 1.0 / (hist + 1e-8)
#     for i in range(3):
#         hist[i] = losses[i].detach()
#     return w


# print(dwa_weights(torch.tensor([1.0,1.0,1.0])))
# image_tokens = proc.image_token.content * proc.num_query_tokens

# image_text_encoding = proc.tokenizer(image_tokens, add_special_tokens = False, padding = False, truncation = False, return_tensors = "pt")
# print(torch.cat((image_text_encoding.input_ids, image_text_encoding.input_ids[:,:5]),dim = -1))

# output = proc(images = torch.ones(3,224,224), text=["i was just fooling around"],add_special_tokens = True,return_tensors = "pt")
# print(output)

# print(type(proc))

# tok = proc.tokenizer

# print("image_token: ", proc.image_token)




# import random

# def random_keep_or_drop(text):
#     # split by '.' and strip spaces
#     sentences = [s.strip() for s in text.split('.') if s.strip()]
#     n = len(sentences)
#     if n == 0:
#         return text
    
    
#     min_drop = 0
#     num_drop = random.randint(min_drop, n-1)
#     drop_indices = set(random.sample(range(n), num_drop))
#     kept = [s for i, s in enumerate(sentences) if i not in drop_indices]
    
#     return '. '.join(kept) + ('.' if kept else '')


# my_string = "hey1. hey 2, hey whatever. this is what it sounds like. hey what are you doing"
# print(random_keep_or_drop(my_string))

# import re

# def clean_caption(caption: str) -> str:
#     """
#     Removes prefixes like '[xxx:] The|This video|image <first word>' 
#     from a caption.
#     """
#     # Regex explanation:
#     # ^\s*                  → start of string, allow leading spaces
#     # (?:[^:]+:\s*)?        → optional 'xxx:' prefix (e.g., "Intro:")
#     # (?:the|this)\s+       → 'the ' or 'this ' (case-insensitive)
#     # (?:video|image)\s+    → 'video ' or 'image '
#     # \w+\s+                → one word + trailing spaces (e.g., "features ", "shows ")
#     pattern = r'^\s*(?:[^:]+:\s*)?(?:the|this)\s+(?:video|image)\s+\w+\s+'
#     return re.sub(pattern, '', caption, flags=re.IGNORECASE)

# captions = [
#     "Intro: The video features a cat playing piano",
#     "The video shows a dog running",
#     "Clip1: This video describes a sunset at the beach",
#     "The image displays a mountain landscape",
#     "This image illustrates a diagram",
# ]

# for c in captions:
#     print(clean_caption(c))


# from PIL import Image
# import requests
# from transformers import AutoProcessor, Siglip2VisionModel
# from transformers import AutoTokenizer, AutoModel,AutoConfig

# # model = Siglip2VisionModel.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
# # processor = AutoProcessor.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")

# # from video_clip.siglip2_temporal import Siglip2VisionTower


# config = AutoConfig.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base").text_config
# print(config.max_position_embeddings)
#64

# model_custom = Siglip2VisionTower(config = config)
# model_custom.load_state_dict(model.state_dict(),strict=False)
# model_custom.to("cuda")


# url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
# image = [Image.open(url) for i in range(8)]

# inputs = processor(images=image, return_tensors="pt")


# inputs.to("cuda")

# pixel_values = inputs.pixel_values.unsqueeze(0)
# pixel_attention_mask = inputs.pixel_attention_mask.unsqueeze(0)
# spatial_shapes= inputs.spatial_shapes.unsqueeze(0)

# outputs = model_custom(pixel_values, pixel_attention_mask, spatial_shapes)
# print(outputs.shape)


# import torch
# from transformers import AutoTokenizer, Siglip2TextModel, AutoConfig

# # model = Siglip2TextModel.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
# tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")

# sample_ids = tokenizer("<bos> a cat on the mat", return_tensors="pt").input_ids[:,:-1]
# print(sample_ids.shape)

# sample_new = tokenizer("<bos> a dog dancing wildly on the mat", return_tensors="pt").input_ids[:,:-1]
# res = torch.cat([sample_ids, sample_new],dim = 1)
# print(res.shape)

# print(sample_ids.attention_mask[0,:-1])
# print(tokenizer.eos_token_id)
# if(sample_ids[0, 0] == tokenizer.bos_token_id):
#     print("added bos")
# config = AutoConfig.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base").text_config

# from video_clip.siglip2 import Siglip2TextTower
# model_custom = Siglip2TextTower(config)
# model_custom.load_state_dict(model.state_dict(),strict=False)
# model_custom.to("cuda")
# # important: make sure to set padding="max_length" as that's how the model was trained
# inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", max_length = 64, return_tensors="pt", return_attention_mask = True)
# print(inputs)
# inputs.to(model_custom.device)
# outputs = model_custom(**inputs)
# print(outputs.shape)

# from PIL import Image
# import requests
# import os
# from transformers import AutoProcessor, AutoModel
# from dataset import Siglip2ImageProcessorFast
# import torch

# model = AutoModel.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")
# processor = AutoProcessor.from_pretrained("/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base")

# processor_2 = Siglip2ImageProcessorFast()

# url = "/mnt/bn/yexiaoyu-test/data/000000039769.jpg"
# image = [Image.open(url)]
# # important: we pass `padding=max_length` since the model was trained with this
# inputs = processor(text=["two cats sleeping on a magenta bed"], images=image, padding="max_length", return_tensors="pt")

# inputs_2 = processor_2(images=image, return_tensors="pt")

# print(inputs.pixel_values)

# print(inputs_2.pixel_values)

# if(torch.equal(inputs.pixel_values,inputs_2.pixel_values)):
#     print("The same")


# with torch.no_grad():
#     outputs = model(**inputs)

# logits_per_image = outputs.logits_per_image
# probs = torch.sigmoid(logits_per_image) # these are the probabilities
# for i in range(probs.shape[0]):
#     print(f"{probs[i][i]:.6%} that image 0 is '{texts[i]}'")