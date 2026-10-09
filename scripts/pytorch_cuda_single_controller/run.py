"""Example of single-controller PyTorch.

This file is intended to be run inside uv. 

Installation instructions:

Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Create the env from pyproject.toml / uv.lock

```
cd sandbox/scripts/pytorch_cuda_single_controller
uv sync
```

Execution instructions:

```
uv run python run.py
```

Logs from a run.

```
yho_google_com@aaskhat-l4-8:~/sandbox/scripts/pytorch_cuda_single_controller$ uv run python run.py
/home/yho_google_com/sandbox/scripts/pytorch_cuda_single_controller/.venv/lib/python3.12/site-packages/torch/_subclasses/functional_tensor.py:368: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /__w/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
torch.cuda.is_available()=True
torch.cuda.device_count()=8
torch.cuda.get_device_name(torch.cuda.current_device())='NVIDIA L4'
W_hsdp=[[tensor([[-0.003, -0.005,  ...,  0.011,  0.023],
        [ 0.016,  0.007,  ...,  0.008,  0.003],
        ...,
        [ 0.025,  0.006,  ...,  0.028,  0.028],
        [ 0.019, -0.005,  ...,  0.022,  0.027]], device='cuda:0'), tensor([[ 0.028, -0.004,  ...,  0.009,  0.017],
        [ 0.025,  0.033,  ...,  0.006, -0.010],
        ...,
        [-0.002,  0.003,  ..., -0.001,  0.047],
        [-0.014,  0.005,  ..., -0.009,  0.004]], device='cuda:1'), tensor([[-0.010,  0.010,  ..., -0.022,  0.017],
        [-0.002, -0.006,  ...,  0.008,  0.008],
        ...,
        [-0.031,  0.005,  ...,  0.007,  0.030],
        [ 0.006, -0.003,  ..., -0.020,  0.065]], device='cuda:2'), tensor([[ 0.016, -0.046,  ..., -0.002,  0.019],
        [ 0.021, -0.032,  ...,  0.014,  0.071],
        ...,
        [ 0.001,  0.000,  ..., -0.012, -0.027],
        [-0.014,  0.008,  ...,  0.009,  0.007]], device='cuda:3')], [tensor([[-0.003, -0.005,  ...,  0.011,  0.023],
        [ 0.016,  0.007,  ...,  0.008,  0.003],
        ...,
        [ 0.025,  0.006,  ...,  0.028,  0.028],
        [ 0.019, -0.005,  ...,  0.022,  0.027]], device='cuda:4'), tensor([[ 0.028, -0.004,  ...,  0.009,  0.017],
        [ 0.025,  0.033,  ...,  0.006, -0.010],
        ...,
        [-0.002,  0.003,  ..., -0.001,  0.047],
        [-0.014,  0.005,  ..., -0.009,  0.004]], device='cuda:5'), tensor([[-0.010,  0.010,  ..., -0.022,  0.017],
        [-0.002, -0.006,  ...,  0.008,  0.008],
        ...,
        [-0.031,  0.005,  ...,  0.007,  0.030],
        [ 0.006, -0.003,  ..., -0.020,  0.065]], device='cuda:6'), tensor([[ 0.016, -0.046,  ..., -0.002,  0.019],
        [ 0.021, -0.032,  ...,  0.014,  0.071],
        ...,
        [ 0.001,  0.000,  ..., -0.012, -0.027],
        [-0.014,  0.008,  ...,  0.009,  0.007]], device='cuda:7')]]
Starting new step... 1
Starting new step... 2
Starting new step... 3
Starting new step... 4
Starting new step... 5
Starting new step... 6
Starting new step... 7
Starting new step... 8
Starting new step... 9
Starting new step... 10
Step 10 losses: ['1.9924', '2.0003', '1.9936', '1.9818', '1.9977', '1.9919', '1.9906', '1.9851']
Starting new step... 11
Starting new step... 12
Starting new step... 13
Starting new step... 14
Starting new step... 15
Starting new step... 16
Starting new step... 17
Starting new step... 18
Starting new step... 19
Starting new step... 20
Step 20 losses: ['1.9866', '1.9944', '1.9878', '1.9760', '1.9919', '1.9861', '1.9848', '1.9793']
Starting new step... 21
Starting new step... 22
Starting new step... 23
Starting new step... 24
Starting new step... 25
Starting new step... 26
Starting new step... 27
Starting new step... 28
Starting new step... 29
Starting new step... 30
Step 30 losses: ['1.9808', '1.9886', '1.9820', '1.9702', '1.9860', '1.9803', '1.9790', '1.9736']
Starting new step... 31
Starting new step... 32
Starting new step... 33
Starting new step... 34
Starting new step... 35
Starting new step... 36
[W1009 21:05:01.314380562 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 3 while trying to allocate 67108864 bytes (free: 20316160, total: 23745396736).
[W1009 21:05:01.314535354 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 3 while trying to allocate 67108864 bytes (free: 20316160, total: 23745396736).
step=36: Caught failure: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 3 has a total capacity of 22.11 GiB of which19.38 MiB is free. Including non-PyTorch memory, this process has 22.09 GiB memory in use. Of the allocated memory 21.74 GiB is allocated by PyTorch, and 13.75 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
Device cuda:0: HEALTHY
Device cuda:1: HEALTHY
Device cuda:2: HEALTHY
[W1009 21:05:01.317321647 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 3 while trying to allocate 67108864 bytes (free: 20316160, total: 23745396736).
[W1009 21:05:01.317458358 CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 3 while trying to allocate 67108864 bytes (free: 20316160, total: 23745396736).
Device cuda:3: FLAKY
Device cuda:4: HEALTHY
Device cuda:5: HEALTHY
Device cuda:6: HEALTHY
Device cuda:7: HEALTHY
Flaky device detected: cuda:3
Flaky device cuda:3 is part of mesh dim 0
Updated mesh after removing flaky slice: [['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']]
Starting new step... 36
Starting new step... 37
Starting new step... 38
Starting new step... 39
Starting new step... 40
Step 40 losses: ['1.9757', '1.9834', '1.9769', '1.9652']
Starting new step... 41
Starting new step... 42
Starting new step... 43
Starting new step... 44
Starting new step... 45
Starting new step... 46
Starting new step... 47
Starting new step... 48
Starting new step... 49
Starting new step... 50
Step 50 losses: ['1.9716', '1.9793', '1.9728', '1.9611']
Starting new step... 51
Starting new step... 52
Starting new step... 53
Starting new step... 54
Starting new step... 55
Starting new step... 56
Starting new step... 57
Starting new step... 58
Starting new step... 59
Starting new step... 60
Step 60 losses: ['1.9676', '1.9753', '1.9688', '1.9571']
Starting new step... 61
Starting new step... 62
Starting new step... 63
Starting new step... 64
Starting new step... 65
Starting new step... 66
Starting new step... 67
Starting new step... 68
Starting new step... 69
Starting new step... 70
Step 70 losses: ['1.9635', '1.9712', '1.9647', '1.9531']
Starting new step... 71
Starting new step... 72
Starting new step... 73
Starting new step... 74
Starting new step... 75
Starting new step... 76
Starting new step... 77
Starting new step... 78
Starting new step... 79
Starting new step... 80
Step 80 losses: ['1.9595', '1.9671', '1.9607', '1.9490']
Starting new step... 81
Starting new step... 82
Starting new step... 83
Starting new step... 84
Starting new step... 85
Starting new step... 86
Starting new step... 87
Starting new step... 88
Starting new step... 89
Starting new step... 90
Step 90 losses: ['1.9554', '1.9631', '1.9566', '1.9450']
Starting new step... 91
Starting new step... 92
Starting new step... 93
Starting new step... 94
Starting new step... 95
Starting new step... 96
Starting new step... 97
Starting new step... 98
Starting new step... 99
Starting new step... 100
Step 100 losses: ['1.9514', '1.9590', '1.9526', '1.9410']
```
"""


import itertools
import sys
from pprint import pprint

import torch
from torch import nn

KEEPALIVE=[]

def main():
    torch.set_printoptions(edgeitems=2, threshold=10)
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.set_printoptions(linewidth=80)

    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.device_count()=}")
    print(f"{torch.cuda.get_device_name(torch.cuda.current_device())=}")

    assert torch.cuda.device_count() > 1
    num_devices = torch.cuda.device_count()

    # We arrange the devices in a rank 2 mesh. dim=0 is dp, dim=1 is fsdp. 
    # We treat each dp slices as the blast radius for elastic training. 
    mesh_size_dp = 2
    mesh_size_fsdp = num_devices // mesh_size_dp
    mesh_size_total = mesh_size_dp * mesh_size_fsdp
    
    # Create a logical mesh of devices. This style of creating an
    # empty pytree of Nones with list comprehension, then 
    # a nested loop to fill out the data, will be repeated throughout this example. 
    mesh = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
    for dp in range(mesh_size_dp):
        for fsdp in range(mesh_size_fsdp):
            mesh[dp][fsdp] = f"cuda:{dp * mesh_size_fsdp + fsdp}"

    # Notice that there is no rendezvous here. There is no dist.init_process_group.
    # This is single-controller! This process can directly access all GPUs. 

    # Create a logical weight W of size 2048, 8192. This projects 2048 features into 8192,
    # like a FFN1 in a transformer FFN. 
    # Shard the weight per hsdp: replicated on dp, sharded on fsdp. 
    W_true = torch.randn(2048, 8192) / (2048 ** 0.5)
    W_local = torch.randn(2048, 8192) / (2048 ** 0.5)
    # Create pytree of shape dp, fsdp
    W_hsdp = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
    for dp in range(mesh_size_dp):
        for fsdp in range(mesh_size_fsdp):
            slice_size = 2048 // mesh_size_fsdp
            slice_start = fsdp * slice_size
            slice_end = slice_start + slice_size
            W_hsdp[dp][fsdp] = W_local[slice_start:slice_end, :].to(mesh[dp][fsdp])

    # Create a dataset and dataloaders returning (x, ytrue) tuples. 
    raw_dataset = []
    for _ in range(2048):
        x = torch.randn(2048)
        ytrue = x @ W_true
        raw_dataset.append((x, ytrue))
    dataloader_2048 = itertools.cycle(torch.utils.data.DataLoader(raw_dataset, batch_size=2048))
    dataloader_1024 = itertools.cycle(torch.utils.data.DataLoader(raw_dataset, batch_size=1024))

    # Set a learning rate.
    lr = 0.3

    print(f"{W_hsdp=}")

    # Train this model for 100 steps. A failure will occur on step 35.
    step = 1
    while step <= 100:
        try:
            print(f"Starting new step... {step}")
            
            # Get the next batch of data, distribute to devices. 
            match mesh_size_dp:
                case 1:
                    batch_x, batch_y = next(dataloader_1024)
                case 2:
                    batch_x, batch_y = next(dataloader_2048)
                case _:
                    raise ValueError(f"Unsupported mesh size: {mesh_size_dp}")

            x_hsdp = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
            ytrue_hsdp = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]

            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    slice_size = batch_x.size(0) // mesh_size_total
                    slice_start = (dp * mesh_size_fsdp + fsdp) * slice_size
                    slice_end = slice_start + slice_size
                    x_hsdp[dp][fsdp] = batch_x[slice_start:slice_end, :].to(mesh[dp][fsdp])
                    ytrue_hsdp[dp][fsdp] = batch_y[slice_start:slice_end, :].to(mesh[dp][fsdp])


            # FSDP on weights. 
            # Set up [mesh_size_dp, mesh_size_fsdp] pytree for result of AG. 
            gathered_W = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    gathered_W[dp][fsdp] = torch.empty(2048, 8192, device=mesh[dp][fsdp], requires_grad=True)

            # AG along the fsdp axis, repeated dp times. That's HSDP. 
            for dp in range(mesh_size_dp):
                torch.cuda.nccl.all_gather(W_hsdp[dp], gathered_W[dp])

            # Linear: x_hsdp @ gathered_W
            z_hsdp  = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    dev = mesh[dp][fsdp]
                    with torch.cuda.stream(torch.cuda.current_stream(device=dev)):
                        # There are no re-entrancy issues here. The return value of the matmul
                        # is a Python object that happens to be a future. 
                        z_hsdp[dp][fsdp] = torch.matmul(x_hsdp[dp][fsdp], gathered_W[dp][fsdp])

            # Calculate losses locally. 
            losses = []
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    dev = mesh[dp][fsdp]
                    with torch.cuda.stream(torch.cuda.current_stream(device=dev)):
                        losses.append(nn.functional.mse_loss(z_hsdp[dp][fsdp], ytrue_hsdp[dp][fsdp]))

            # Backprop the losses locally. 
            torch.autograd.backward(losses)
            # for dp in range(mesh_size_dp):
            #     for fsdp in range(mesh_size_fsdp):
            #         dev = mesh[dp][fsdp]
            #         with torch.cuda.stream(torch.cuda.current_stream(device=dev)):
            #             losses[dp * mesh_size_fsdp + fsdp].backward()

            # Put the grads in their own container.
            grad_gathered = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    grad_gathered[dp][fsdp] = gathered_W[dp][fsdp].grad

            # FSDP backwards: Reduce_scatter the grads. 
            grad_rs = [[None for _ in range(mesh_size_fsdp)] for _ in range(mesh_size_dp)]
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    grad_rs[dp][fsdp] = torch.empty_like(W_hsdp[dp][fsdp])

            for dp in range(mesh_size_dp):
                torch.cuda.nccl.reduce_scatter(grad_gathered[dp], grad_rs[dp], op=4) # ncclAvg

            # DP backwards: AR the gradients across the data parallel axis.
            # Transpose grad_rs to shape [fsdp, dp] so each slice is across DP replicas.
            if mesh_size_dp > 1:
                grad_rs_t = [[grad_rs[dp][fsdp] for dp in range(mesh_size_dp)] for fsdp in range(mesh_size_fsdp)]
                grad_ar = [[None for _ in range(mesh_size_dp)] for _ in range(mesh_size_fsdp)]
                for fsdp in range(mesh_size_fsdp):
                    for dp in range(mesh_size_dp):
                        grad_ar[fsdp][dp] = torch.empty_like(grad_rs_t[fsdp][dp])

                for fsdp in range(mesh_size_fsdp):
                    torch.cuda.nccl.all_reduce(grad_rs_t[fsdp], grad_ar[fsdp], op=4) # ncclAvg

                # Assign the synchronized gradients from grad_ar [fsdp][dp] to parameters [dp][fsdp]
                for dp in range(mesh_size_dp):
                    for fsdp in range(mesh_size_fsdp):
                        W_hsdp[dp][fsdp].grad = grad_ar[fsdp][dp]
            else:
                for dp in range(mesh_size_dp):
                    for fsdp in range(mesh_size_fsdp):
                        W_hsdp[dp][fsdp].grad = grad_rs[dp][fsdp]

            # Step the optimizer (SGD) for each local weight.
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    with torch.no_grad():
                        W_hsdp[dp][fsdp] -= lr * W_hsdp[dp][fsdp].grad
                        W_hsdp[dp][fsdp].grad = None
            
            # Eval the model
            eval_losses = []
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    with torch.no_grad():
                        loss_val = losses[dp * mesh_size_fsdp + fsdp].item()
                        eval_losses.append(f"{loss_val:.4f}")
            if step % 10 == 0:
                print(f"Step {step} losses: {eval_losses}")

            # Simulate a failure on device 3 by using up all its remaining memory.
            if step == 35:
                with torch.cuda.device(3):
                    # Empty PyTorch caching allocator cache first so we know exact driver free bytes
                    torch.cuda.empty_cache()
                    free_bytes, _ = torch.cuda.mem_get_info(3)
                    # Leave 20 MB for driver/runtime overhead so this allocation succeeds,
                    # but leaves cuda:3 with almost 0 bytes free for next step.
                    KEEPALIVE.append(torch.empty(free_bytes - 20 * 1024**2, dtype=torch.uint8, device="cuda:3"))
            # for dev_id in range(torch.cuda.device_count()):
            #     print(f"Device cuda:{dev_id}: {torch.cuda.mem_get_info(dev_id)=}")

            # Step succeeded, advance to next step
            step += 1
        except RuntimeError as e:
            print(f"{step=}: Caught failure: {e}")
            # Find the flaky device
            flaky_device = None
            flakey_mesh_dim = None
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    dev_id = int(mesh[dp][fsdp].split(":")[-1])
                    try:
                        with torch.cuda.device(dev_id):
                            torch.cuda.synchronize(dev_id)
                            # Probe with 64 MiB (the typical buffer size for step forward/gather)
                            _ = torch.ones(64 * 1024 * 1024, dtype=torch.int8, device=f"cuda:{dev_id}")
                            print(f"Device cuda:{dev_id}: HEALTHY")
                    except RuntimeError as dev_err:
                        print(f"Device cuda:{dev_id}: FLAKY")
                        flaky_device = dev_id
                        flakey_mesh_dim = dp

            # This is basically pathwaysutils.elastic.manager & elastic_handler
            print(f"Flaky device detected: cuda:{flaky_device}")
            print(f"Flaky device cuda:{flaky_device} is part of mesh dim {flakey_mesh_dim}")

            # Create a new mesh and weight pytree without the flaky slice. 
            if flakey_mesh_dim is not None:
                mesh = mesh[0:flakey_mesh_dim] + mesh[flakey_mesh_dim+1:]
                W_hsdp = W_hsdp[0:flakey_mesh_dim] + W_hsdp[flakey_mesh_dim+1:]
                mesh_size_dp = len(mesh)
                mesh_size_total = mesh_size_dp * mesh_size_fsdp
                lr = lr * (0.5 ** 0.5) # sqrt lr rate suggests we use a lower lr.
                # Note: for simplicity, we don't worry about deterministic dataloading on re-running the failed step.
                print(f"Updated mesh after removing flaky slice: {mesh}")

if __name__ == "__main__":
    sys.exit(main())