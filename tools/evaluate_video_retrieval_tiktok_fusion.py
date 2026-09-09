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
from video_clip import VideoSigLIP2ShortLongBLIP
from dataset import TikTokRetrievalDataset


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def validate_msrvtt(model, dataloader,
                    recall_k_list=[1, 5, 10],
                    eval_batch_size=32, output_dir=None, use_segment=True):

    video_features = []
    text_features = []
    captions = []
    device = model.vision_tower.device

    # compute text and vision features
    print('Computing text and vision features', flush=True)
    for idx,data in tqdm.tqdm(enumerate(dataloader)):
        pixel_values = data['pixel_values'].to(device)
        pixel_attention_mask = data['pixel_attention_mask'].to(device)
        spatial_shapes = data['spatial_shapes'].to(device)
        
        caption_input_ids = data['fusion_input_ids'].to(device)#caption的segmentid是zero可以直接传 

        title_input_ids = data['title_input_ids'].to(device)
        
        batch_size, concat, seq_len = title_input_ids.shape
        if use_segment:
            segment_ids = torch.zeros_like(title_input_ids)# 0: title
            segment_ids = segment_ids.view(batch_size, concat* seq_len).contiguous()
            segment_ids[:, 64:128] = 1    # 1: sticker
            segment_ids[:, 128:192] = 2   # 2: ocr
            segment_ids[:, 192:256] = 3   # 3: asr
        else:
            segment_ids = None

        title_input_ids = title_input_ids.view(batch_size * concat, seq_len).contiguous()
        
        
        with torch.no_grad():
            # caption
            caption_embeds, caption_pooled = model.encode_text(input_ids=caption_input_ids)
            # title
            title_embeds, title_pooled = model.encode_text(input_ids = title_input_ids)

            title_embeds = title_embeds.view(batch_size, concat, seq_len, -1).contiguous()

            title_embeds = title_embeds.view(batch_size, concat * seq_len, -1).contiguous()
            #video
            video_embeds, video_pooled = model.encode_video(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask, spatial_shapes=spatial_shapes)
            #fusion
            fused_embeds, fused_pooled = model.fuse_video_title(vision_embed = video_embeds,title_embed = title_embeds,  
                                                                vision_attn_mask=None, title_attn_mask = None, segment_ids = segment_ids)
        

        video_features.append(fused_pooled.cpu())
        text_features.append(caption_pooled.cpu())
    
    text_features = torch.cat(text_features)
    video_features = torch.cat(video_features)

    print('Computing metrics', flush=True)

    texts_emb = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    images_emb = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

    # get the score for each text and image pair
    scores = torch.matmul(texts_emb, images_emb.t().to(texts_emb.device))
    logit_scale, logit_bias = model.logit_scale.to(texts_emb.device), model.logit_bias.to(texts_emb.device)
    scores = scores * logit_scale.exp() + logit_bias
    probs = torch.sigmoid(scores)

    for i in range(20):
        print(f"{probs[i][i]:.4%} that image 0 is target")
    #print(prob[:8][:8])

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True

    scores_T = scores.T
    positive_pairs_T = positive_pairs.T

    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f't2v_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores, positive_pairs, eval_batch_size, scores.device,
                             k=recall_k) > 0).float().mean().item()
        metrics[f'v2t_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores_T, positive_pairs_T, eval_batch_size, scores.device,
                             k=recall_k) > 0).float().mean().item()

    print(metrics)

    if output_dir is not None:
        with open(output_dir + "/val_log.txt", "a") as f:
            f.write(json.dumps(metrics) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate MSR-VTT', add_help=False)
    parser.add_argument('--model_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/siglip2-base", type=str, help='the config path of the siglip2 models to be used')
    parser.add_argument('--bert_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/bert_uncased", type=str, help='the config path of the bert tokenizer to be used')
    parser.add_argument('--blip_path',default="/mnt/bn/yexiaoyu-test/models/huggingface/blip2", type=str, help='the config path of the blip tokenizer to be used')
    parser.add_argument('--video_root',default="/mnt/bn/jiny-ttls-i18n-fr1q/tiktok_30k_recaption_long_fusion_dedup.jsonl", type=str, help='video paths')
    parser.add_argument('--max_frames', default=8, type=int, help='number of max video frames')
    parser.add_argument('--pretrained_path',default=None, type=str, help='the pretrained siglip2 models to be used')
    parser.add_argument('--output_path',default=None, type=str, help='validation results export path')
    parser.add_argument('--use_segment',action='store_true',default=False,help='whether to use segment')
    parser.add_argument('--batch_size',default=32, type=int, help='evaluation batch size')
    args = parser.parse_args()

    

    test_dataset = TikTokRetrievalDataset(video_root=args.video_root, 
                max_short_len=64,
                max_text_len=284,
                max_frame_len=args.max_frames,
                tokenizer_dir=args.model_path,
                bert_tokenizer_dir=args.bert_path,
                )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=default_collate,
    ) 

    model = VideoSigLIP2ShortLongBLIP(model_path=args.model_path, blip_path=args.blip_path, interpolate=6)
    
    model.load_state_dict(torch.load(args.pretrained_path + '/pytorch_model.bin',map_location='cpu'))

    model.to("cuda")
    model.eval()

    metrics = validate_msrvtt(model = model, dataloader = test_loader, eval_batch_size=args.batch_size, output_dir = args.output_path, use_segment = args.use_segment)