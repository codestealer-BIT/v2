import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import accelerate
from .utils import MetricLogger, SmoothedValue
from collections import deque

# dwa weights

total_task = 4
loss_keys = ['video_music_siglip_loss', 'video_music_matching_loss', 'lang_side_task_loss', 'lang_country_side_task_loss']
base_task_weights = torch.tensor([1.0, 1.0, 0.2, 0.2])
hist = [deque(maxlen=2) for _ in range(total_task)]
T = 2.0

@torch.no_grad()
def dwa_weights(losses):
    for i, li in enumerate(losses):
        hist[i].append(li.detach())
    if any(len(h) < 2 for h in hist):
        w = torch.ones(total_task, device=losses[0].device)
        return w
    ratios = torch.tensor([h[-1] / (h[-2] + 1e-8) for h in hist], device=losses[0].device)
    torch.clamp_(ratios, 0.2, 5.0)
    exps = torch.exp(ratios / T)
    w = total_task * exps / (exps.sum() + 1e-8)

    if torch.isnan(w).any() or torch.isinf(w).any():
        return torch.ones(total_task, device=losses[0].device)
    
    return w


def update_model_ema(model, model_ema, accelerator, decay):
    """Apply exponential moving average update.

    The  weights are updated in-place as follow:
    w_ema = w_ema * decay + (1 - decay) * w
    Args:
        model: active model that is being optimized
        model_ema: running average model
        decay: exponential decay parameter
    """
    with torch.no_grad():
        msd = accelerator.get_state_dict(model)
        for k, ema_v in model_ema.state_dict().items():
            if k in msd:
                model_v = msd[k].detach().to(ema_v.device, dtype=ema_v.dtype)
                ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)

#whatever this is
def train_one_epoch_for_video_siglip_for_music_rec(
    runner,
    model_ema: torch.nn.Module,
    accelerator: accelerate.Accelerator,
    model_dtype: str,
    data_loader: Iterable, 
    optimizer: torch.optim.Optimizer,
    lr_schedule_values,
    epoch: int, 
    clip_grad: float = 1.0,
    start_steps=None,
    args=None,
    print_freq=20,
    iters_per_epoch=2000,
    ema_update_step=1,
    ema_decay=0.9999,
    use_dwa = True
):
    runner.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('pred_score', SmoothedValue(window_size=10, fmt='{value:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('m_bias', SmoothedValue(window_size=10, fmt='{value:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('regression_scale', SmoothedValue(window_size=10, fmt='{value:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('regression_bias', SmoothedValue(window_size=10, fmt='{value:.4f} ({global_avg:.4f})'))

    # metric_logger.add_meter('neg_filter_ratio', SmoothedValue(window_size=10, fmt='{value:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('user_gate_mean', SmoothedValue(window_size=10, fmt='{value:.6f} ({global_avg:.6f})'))
    # metric_logger.add_meter('same_meta_total', SmoothedValue(window_size=10, fmt='{value:.4f} ({global_avg:.4f})'))
    # metric_logger.add_meter('same_meta_filtered_count', SmoothedValue(window_size=10, fmt='{value:.6f} ({global_avg:.6f})'))
    # metric_logger.add_meter('same_meta_filtered_ratio', SmoothedValue(window_size=10, fmt='{value:.6f} ({global_avg:.6f})'))
    header = 'Epoch: [{}]'.format(epoch)
    # train_loss = 0.0
    train_loss_dict = {}

    print("Start training epoch {}, {} iters per inner epoch. Training dtype {}".format(epoch, iters_per_epoch, model_dtype))
    base_runner = accelerator.unwrap_model(runner)
    if hasattr(base_runner, "user_context_adapter") and hasattr(base_runner.user_context_adapter, "alpha"):
        print(f"Epoch {epoch} alpha: {base_runner.user_context_adapter.alpha.item():.6f}")
    skipped_batches = 0
    for step in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
        if step >= iters_per_epoch:
            break

        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[start_steps] * param_group.get("lr_scale", 1.0)
                
                

        for _ in range(args.gradient_accumulation_steps):

            with accelerator.accumulate(runner):
                # To fetch the data sample and Move the input to device
                samples = next(data_loader)

                # Per-rank batch validation, then sync decision across all ranks
                local_ok = torch.tensor(1, device=accelerator.device)
                try:
                    local_batch_size = torch.tensor(
                        [samples['pixel_values'].shape[0]],
                        device=accelerator.device
                    )
                except Exception as e:
                    local_ok = torch.tensor(0, device=accelerator.device)
                    local_batch_size = torch.tensor([0], device=accelerator.device)
                    if accelerator.process_index == 0:
                        print(f"[Rank {accelerator.process_index}] Error in batch validation: {e}")

                if torch.distributed.is_initialized():
                    # If any rank has invalid batch, all ranks skip this step
                    ok_sum = local_ok.clone()
                    torch.distributed.all_reduce(ok_sum, op=torch.distributed.ReduceOp.SUM) # 这里求和
                    if ok_sum.item() != accelerator.num_processes:
                        if accelerator.process_index == 0:
                            print(f"⚠️ Warning: Invalid batch detected at step {step}, skip all ranks")
                        skipped_batches += 1
                        continue

                    batch_sizes = [
                        torch.zeros_like(local_batch_size)
                        for _ in range(accelerator.num_processes)
                    ]
                    torch.distributed.all_gather(batch_sizes, local_batch_size)
                    mismatch = torch.tensor(0, device=accelerator.device)
                    if not all(bs.item() == batch_sizes[0].item() for bs in batch_sizes):
                        mismatch = torch.tensor(1, device=accelerator.device)
                    torch.distributed.all_reduce(mismatch, op=torch.distributed.ReduceOp.MAX)
                    # Keep ranks in lock-step to avoid NCCL hang
                    if mismatch.item() > 0:
                        if accelerator.process_index == 0:
                            print(f"⚠️ Warning: Inconsistent batch sizes across ranks at step {step}: {[bs.item() for bs in batch_sizes]}")
                        skipped_batches += 1
                        continue

                    unexpected = torch.tensor(0, device=accelerator.device)
                    if local_batch_size.item() != args.batch_size:
                        unexpected = torch.tensor(1, device=accelerator.device)
                    torch.distributed.all_reduce(unexpected, op=torch.distributed.ReduceOp.MAX)
                    # Skip if any rank has unexpected batch size
                    if unexpected.item() > 0:
                        if accelerator.process_index == 0:
                            print(f"⚠️ Warning: Unexpected batch size {local_batch_size.item()} != {args.batch_size} at step {step}")
                        skipped_batches += 1
                        continue

                for key in samples:
                    if isinstance(samples[key], torch.Tensor):
                        samples[key] = samples[key].to(accelerator.device)

                ret_dict = runner(
                    samples, model_ema=model_ema, 
                    rank=accelerator.process_index, world_size=accelerator.num_processes
                )
                pred_score = 0.0
                if 'pred_score' in ret_dict:
                    pred_score = ret_dict['pred_score']
                    if isinstance(pred_score, torch.Tensor):
                        pred_score = pred_score.detach().mean().item()
                m_bias = 0.0
                if 'm_bias' in ret_dict:
                    m_bias = ret_dict['m_bias']
                    if isinstance(m_bias, torch.Tensor):
                        m_bias = m_bias.detach().mean().item()
                regression_scale = None
                regression_bias = None
                if 'logit_scale' in ret_dict:
                    regression_scale = ret_dict['logit_scale']
                    if isinstance(regression_scale, torch.Tensor):
                        regression_scale = regression_scale.detach().mean().item()
                if 'logit_bias' in ret_dict:
                    regression_bias = ret_dict['logit_bias']
                    if isinstance(regression_bias, torch.Tensor):
                        regression_bias = regression_bias.detach().mean().item()
                # neg_filter_ratio = 0.0
                # if 'filtered_ratio' in ret_dict:
                #     neg_filter_ratio = ret_dict['filtered_ratio']
                #     if isinstance(neg_filter_ratio, torch.Tensor):
                #         neg_filter_ratio = neg_filter_ratio.detach().mean().item()
                # user_gate_mean = None
                # if 'user_gate_mean' in ret_dict:
                #     user_gate_mean = ret_dict['user_gate_mean']
                #     if isinstance(user_gate_mean, torch.Tensor):
                #         user_gate_mean = user_gate_mean.detach().mean().item()
                # same_meta_total = 0.0
                # if 'same_meta_total' in ret_dict:
                #     same_meta_total = ret_dict['same_meta_total']
                #     if isinstance(same_meta_total, torch.Tensor):
                #         same_meta_total = same_meta_total.detach().mean().item()
                # same_meta_filtered_count = 0.0
                # if 'same_meta_filtered_count' in ret_dict:
                #     same_meta_filtered_count = ret_dict['same_meta_filtered_count']
                #     if isinstance(same_meta_filtered_count, torch.Tensor):
                #         same_meta_filtered_count = same_meta_filtered_count.detach().mean().item()
                # same_meta_filtered_ratio = 0.0
                # if 'same_meta_filtered_ratio' in ret_dict:
                #     same_meta_filtered_ratio = ret_dict['same_meta_filtered_ratio']
                #     if isinstance(same_meta_filtered_ratio, torch.Tensor):
                #         same_meta_filtered_ratio = same_meta_filtered_ratio.detach().mean().item()
                if use_dwa:
                    device = ret_dict['loss'].device
                    loss_avg = []
                    for key in loss_keys:
                        val = ret_dict.get(key, torch.tensor(0.0, device=device))
                        loss_avg.append(accelerator.gather(val.unsqueeze(0)).mean())
                    
                    weights = dwa_weights(loss_avg).to(device)
                    weights = weights * base_task_weights.to(device)
                    weights = (weights / (weights.sum() + 1e-8)) * total_task
                    loss = torch.tensor(0.0, device=device)
                    for i in range(total_task):
                        loss += weights[i] * ret_dict.get(loss_keys[i], torch.tensor(0.0, device=device))
                    
                    ret_dict['weights'] = weights
                    ret_dict['loss'] = loss

                else:
                    loss = ret_dict['loss']
                

                for key in ret_dict.keys():
                    if 'loss' in key:
                        train_loss_dict.setdefault(key, 0.0)
                        cur_loss_value = ret_dict[key]
                        avg_loss = accelerator.gather(cur_loss_value.repeat(args.batch_size)).mean()
                        train_loss_dict[key] += avg_loss.item() / args.gradient_accumulation_steps
                
                if 'weights' in ret_dict:
                    for i in range(total_task):
                        train_loss_dict[f'weight_{loss_keys[i]}'] =  weights[i].item()

                # avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                # train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # loss = ret_dict['loss']

                # Check if the loss is nan
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value), force=True)
                    if('weights' in ret_dict.keys()):
                        print("weights: ", ret_dict['weights'])
                    
                    print("hist", hist)
                    sys.exit(1)

                accelerator.backward(loss)

                # clip the gradient
                if accelerator.sync_gradients:
                    params_to_clip = runner.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, clip_grad)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                # Update every 100 steps
                if model_ema is not None and start_steps % ema_update_step == 0:
                    cur_ema_decay = ema_decay
                    update_model_ema(runner, model_ema, accelerator, decay=cur_ema_decay)

                start_steps += 1

                train_loss_dict['pred_score'] = pred_score
                train_loss_dict['m_bias'] = m_bias
                # train_loss_dict['neg_filter_ratio'] = neg_filter_ratio
                # if user_gate_mean is not None:
                #     train_loss_dict['user_gate_mean'] = user_gate_mean
                # train_loss_dict['same_meta_total'] = same_meta_total
                # train_loss_dict['same_meta_filtered_count'] = same_meta_filtered_count
                # train_loss_dict['same_meta_filtered_ratio'] = same_meta_filtered_ratio
                if regression_scale is not None:
                    train_loss_dict['regression_scale'] = regression_scale
                if regression_bias is not None:
                    train_loss_dict['regression_bias'] = regression_bias
                
                # Report to tensorboard
                # accelerator.log({"train_loss": train_loss}, step=start_steps)
                # metric_logger.update(loss=train_loss)
                accelerator.log(train_loss_dict, step=start_steps)
                metric_logger.update(**train_loss_dict)
 
                # metric_logger.update(neg_filter_ratio=neg_filter_ratio)
                # if user_gate_mean is not None:
                #     metric_logger.update(user_gate_mean=user_gate_mean)
                # metric_logger.update(same_meta_total=same_meta_total)
                # metric_logger.update(same_meta_filtered_count=same_meta_filtered_count)
                # metric_logger.update(same_meta_filtered_ratio=same_meta_filtered_ratio)
                train_loss_dict = {}
                # train_loss = 0.0

                min_lr = 10.
                max_lr = 0.
                for group in optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                weight_decay_value = None
                for group in optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)
                metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}