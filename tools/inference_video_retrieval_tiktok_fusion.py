import argparse



import numpy as np
import torch
import tqdm
import os
import json

import sys
sys.path.append('/opt/tiger/MMPretrain')

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from video_clip import VideoSigLIP2TemporalFusion
from dataset import TikTokRetrievalDataset

from transformers import AutoTokenizer


def generate_caption(model, image_embeds, tokenizer, max_len=256, device="cuda"):
    
    # Start with [CLS] as BOS
    input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Build attention mask: all ones since no padding
            
            # Forward through your decode()
            logits = model.decode(
                input_ids=input_ids,
                attention_mask=None,
                encoder_embeddings=image_embeds,
                encoder_attention_mask=None,  # if you don't need masking for image feats
            )
            
            # Pick last token
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # Stop if [SEP] predicted
            if next_token_id.item() == tokenizer.sep_token_id:
                break
    
    # Decode into text
    caption = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return caption



def validate_msrvtt(model, dataloader, tokenizer,
                    recall_k_list=[1, 5, 10],
                    eval_batch_size=32, output_dir=None):

    video_features = []
    text_features = []
    captions = []
    model.eval()
    device = model.vision_tower.device

    # compute text and vision features
    count = 0
    for idx,data in tqdm.tqdm(enumerate(dataloader)):
        item_ids = data['item_ids']

        pixel_values = data['pixel_values'].to(device)
        pixel_attention_mask = data['pixel_attention_mask'].to(device)
        spatial_shapes = data['spatial_shapes'].to(device)
        

        title_input_ids = data['title_input_ids'].to(device)
        title_segment_ids = data['title_segment_ids'].to(device)
        title_attention_mask = data['title_attention_mask'].to(device)#在text encoder的时候不要传
        
        with torch.no_grad():
            
            # title
            title_embeds, title_pooled = model.encode_text(input_ids = title_input_ids, segment_ids = title_segment_ids)
            #video
            video_embeds, video_pooled = model.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
            #fusion
            fused_embeds, fused_pooled = model.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                                vision_attn_mask=None, title_attn_mask = title_attention_mask)
    
        # decode

        decoded_caption = generate_caption(model, fused_embeds, tokenizer=tokenizer)
        caption = data['captions']
        print("gt caption:" + caption[0])
        print("decoded caption:" + decoded_caption)

        count += 1
        if(count > 10):
            break

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate MSR-VTT', add_help=False)
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--bert_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased", type=str, help='the config path of the bert tokenizer to be used')
    parser.add_argument('--video_root',default="/mnt/bn/yexiaoyu-test/data/tiktok_data_fusion_30k", type=str, help='video paths')
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--pretrained_path',default=None, type=str, help='the pretrained siglip2 models to be used')
    parser.add_argument('--output_path',default=None, type=str, help='validation results export path')
    parser.add_argument('--batch_size',default=32, type=int, help='evaluation batch size')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    test_dataset = TikTokRetrievalDataset(video_root=args.video_root, 
                max_text_len=256,
                max_frame_len=args.max_frames,
                tokenizer_dir=args.model_path,
                bert_tokenizer_dir=args.bert_path,
                )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_collate,
    ) 

    model = VideoSigLIP2TemporalFusion(model_path=args.model_path, tokenizer_path=args.bert_path)
    
    model.load_state_dict(torch.load(args.pretrained_path + '/pytorch_model.bin',map_location='cpu'))

    model.to("cuda")

    metrics = validate_msrvtt(model = model, dataloader = test_loader, eval_batch_size=args.batch_size, output_dir = args.output_path, tokenizer = tokenizer)