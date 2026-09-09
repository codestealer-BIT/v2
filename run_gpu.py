#!/usr/bin/env python3
"""
saturate_gpus.py

单机多卡（每卡一个进程）GPU 压力/占用脚本。
设计目标：长期循环大量矩阵运算 + 占用显存，逼近较高的显卡利用率。

注意：
- 需要安装 PyTorch（CUDA 版本）。
- 默认每卡会创建一个占位显存张量 + 持续的 matmul / conv 操作。
- 可通过命令行参数调整矩阵大小 / batch / 持续时间 / 是否用 fp16 等。
"""

import argparse
import os
import time
import math
import signal
from multiprocessing import Event
import torch
import torch.multiprocessing as mp
import torch.nn as nn

def worker_fn(rank: int, args, stop_event: Event):
    # 把每个子进程绑定到一个 CUDA device
    device_id = rank
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    # 设置 cudnn benchmark 等
    torch.backends.cudnn.benchmark = True

    # 占住一部分显存作为“常驻 buffer”
    reserved_bytes = int(args.reserve_mem_gb * (1024**3))
    reserved_elems = reserved_bytes // 4  # float32
    if reserved_elems > 0:
        try:
            # 创建一个大的张量并固定在 GPU（float32）
            hold = torch.empty(reserved_elems, dtype=torch.float32, device=device)
            # 写入防止稀疏分配
            hold.uniform_(0, 1)
        except Exception as e:
            print(f"[GPU {device_id}] reserve mem failed: {e}")
            hold = None
    else:
        hold = None

    # 准备循环运算张量
    M = args.mat_dim
    N = args.mat_dim
    K = args.mat_dim
    batch = args.batch

    # Decide dtype
    dtype = torch.float16 if args.fp16 else torch.float32

    # Create random weight tensors (kept on GPU)
    try:
        A = torch.randn((batch, M, K), device=device, dtype=dtype, requires_grad=False)
        B = torch.randn((batch, K, N), device=device, dtype=dtype, requires_grad=False)
    except RuntimeError as e:
        # Fallback to smaller size if allocation fails
        print(f"[GPU {device_id}] initial alloc failed ({e}), reducing sizes")
        # reduce size by half until success
        reduce_factor = 2
        while True:
            try:
                M = max(64, M // reduce_factor)
                N = max(64, N // reduce_factor)
                K = max(64, K // reduce_factor)
                A = torch.randn((batch, M, K), device=device, dtype=dtype, requires_grad=False)
                B = torch.randn((batch, K, N), device=device, dtype=dtype, requires_grad=False)
                print(f"[GPU {device_id}] resized to M,N,K = {M},{N},{K}")
                break
            except Exception as e2:
                reduce_factor *= 2
                if M <= 64 and N <= 64 and K <= 64:
                    raise RuntimeError(f"[GPU {device_id}] cannot allocate minimal tensors: {e2}")

    # optional small conv network to mix compute pattern
    convnet = None
    if args.use_conv:
        convnet = nn.Sequential(
            nn.Conv2d(args.conv_in_channels, args.conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.conv_out_channels, args.conv_out_channels, kernel_size=3, padding=1),
        ).to(device).eval()
        # convert to float16 if requested (if supported)
        if args.fp16:
            convnet.half()

        # conv input
        conv_input = torch.randn((batch, args.conv_in_channels, args.conv_hw, args.conv_hw),
                                 device=device, dtype=dtype)

    # Stats
    iterations = 0
    sync_every = max(1, args.sync_every)
    start_ts = time.time()
    last_report = start_ts

    try:
        while not stop_event.is_set():
            # Heavy matmul: batch matrix multiply
            # shape: (batch, M, N) = (batch, M, K) @ (batch, K, N)
            # if batch==1, matmul fallback to normal matmul
            C = torch.matmul(A, B)

            # Optional conv compute
            if convnet is not None:
                out = convnet(conv_input)

            # A small reduction to keep some memory traffic
            s = C.sum()
            # force some device work / synchronization but don't call synchronize every iter
            if iterations % sync_every == 0:
                # Note: .item() will synchronize the device
                _ = s.item()

            iterations += 1

            # periodic progress print
            now = time.time()
            if now - last_report >= args.report_interval:
                elapsed = now - start_ts
                it_per_sec = iterations / elapsed if elapsed > 0 else 0.0
                try:
                    mem_alloc = torch.cuda.memory_allocated(device)
                    mem_reserved = torch.cuda.memory_reserved(device)
                except Exception:
                    mem_alloc = mem_reserved = 0
                print(f"[GPU {device_id}] time={elapsed:.1f}s it={iterations} it/s={it_per_sec:.2f} "
                      f"mem_alloc={mem_alloc//1024**2}MB reserved={mem_reserved//1024**2}MB")
                last_report = now

            # Check duration timeout
            if args.duration > 0 and (time.time() - start_ts) >= args.duration:
                break

    except KeyboardInterrupt:
        print(f"[GPU {device_id}] KeyboardInterrupt received, stopping.")
    except Exception as e:
        print(f"[GPU {device_id}] exception: {e}")
    finally:
        # cleanup
        if hold is not None:
            del hold
        del A, B, C
        if convnet is not None:
            del convnet
        torch.cuda.empty_cache()
        print(f"[GPU {device_id}] finished after {iterations} iterations, elapsed {time.time() - start_ts:.1f}s")


def spawn_workers(num_gpus: int, args):
    mp.set_start_method('spawn', force=True)
    stop_event = mp.Event()

    # handle ctrl-c in master to notify children
    def _signal_handler(sig, frame):
        print("Master got signal to stop, notifying workers...")
        stop_event.set()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    procs = []
    for rank in range(num_gpus):
        p = mp.Process(target=worker_fn, args=(rank, args, stop_event), daemon=False)
        p.start()
        procs.append(p)

    try:
        # wait for children
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("Master KeyboardInterrupt, setting stop event")
        stop_event.set()
        for p in procs:
            p.join(timeout=5)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU saturate script")
    p.add_argument('--gpus', type=int, default=8, help='number of GPUs / processes to launch')
    p.add_argument('--duration', type=int, default=300, help='total run time in seconds (0 means infinite)')
    p.add_argument('--mat-dim', type=int, default=4096, help='matrix dimension (MxK x KxN) default 4096')
    p.add_argument('--batch', type=int, default=1, help='batch dimension for batched matmul')
    p.add_argument('--fp16', action='store_true', help='use float16 for compute (if supported)')
    p.add_argument('--use-conv', action='store_true', help='add conv workload to mix compute patterns')
    p.add_argument('--conv-in-channels', dest='conv_in_channels', type=int, default=32)
    p.add_argument('--conv-out-channels', dest='conv_out_channels', type=int, default=64)
    p.add_argument('--conv-hw', dest='conv_hw', type=int, default=128)
    p.add_argument('--reserve-mem-gb', type=float, default=2.0,
                   help='per-GPU reserved memory (GB) to allocate at start to increase mem pressure')
    p.add_argument('--sync-every', type=int, default=10, help='call .item() (sync) every N iterations')
    p.add_argument('--report-interval', type=float, default=10.0, help='seconds between status prints')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # basic checks
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install CUDA-enabled PyTorch.")
    total_gpus = torch.cuda.device_count()
    if args.gpus > total_gpus:
        print(f"Warning: requested {args.gpus} GPUs but only {total_gpus} visible. Using {total_gpus}.")
        args.gpus = total_gpus

    print(f"Launching {args.gpus} worker processes (visible GPUs: {total_gpus})")
    print(f"mat_dim={args.mat_dim} batch={args.batch} fp16={args.fp16} reserve_mem_gb={args.reserve_mem_gb}")
    spawn_workers(args.gpus, args)
    print("All workers finished.")