import os
import time
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    n = 16384
    dtype = torch.float16

    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    # 预热
    for _ in range(10):
        a = torch.relu(a @ b)
    torch.cuda.synchronize()

    start = time.time()
    i = 0
    while True:
        a = torch.relu(a @ b)
        i += 1

        if i % 50 == 0:
            torch.cuda.synchronize()
            if rank == 0:
                print(f"iter={i}, elapsed={time.time() - start:.2f}s", flush=True)

if __name__ == "__main__":
    main()