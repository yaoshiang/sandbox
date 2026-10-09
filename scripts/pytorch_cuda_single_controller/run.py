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
    raw_dataset = [(torch.randn(2048), torch.randn(8192)) for _ in range(2048)]
    dataloader_2048 = itertools.cycle(torch.utils.data.DataLoader(raw_dataset, batch_size=2048))
    dataloader_1024 = itertools.cycle(torch.utils.data.DataLoader(raw_dataset, batch_size=1024))

    # Set a learning rate.
    lr = 2.0

    print(f"{W_hsdp=}")

    # Train this model for 20 steps. A failure will occur on step 11 out of 20. 
    step = 1
    while step <= 20:
        try:
            print(f"Starting new step... {step}")
            
            # Get the next batch of data, distribute to devices. 
            match mesh_size_dp:
                case 1:
                    batch_x, batch_y = next(dataloader_2048)
                case 2:
                    batch_x, batch_y = next(dataloader_1024)
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
            for dp in range(mesh_size_dp):
                for fsdp in range(mesh_size_fsdp):
                    dev = mesh[dp][fsdp]
                    with torch.cuda.stream(torch.cuda.current_stream(device=dev)):
                        losses[dp * mesh_size_fsdp + fsdp].backward()

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
                torch.cuda.nccl.reduce_scatter(grad_gathered[dp], grad_rs[dp])

            # DP backwards: AR the gradients across the data parallel axis.
            # Transpose grad_rs to shape [fsdp, dp] so each slice is across DP replicas.
            if mesh_size_dp > 1:
                grad_ar_out = [[None for _ in range(mesh_size_dp)] for _ in range(mesh_size_fsdp)]
                for fsdp in range(mesh_size_fsdp):
                    for dp in range(mesh_size_dp):
                        grad_ar_out[fsdp][dp] = torch.empty_like(grad_rs[dp][fsdp])

                for fsdp in range(mesh_size_fsdp):
                    dp_shards = [grad_rs[dp][fsdp] for dp in range(mesh_size_dp)]
                    torch.cuda.nccl.all_reduce(dp_shards, grad_ar_out[fsdp])

                for dp in range(mesh_size_dp):
                    for fsdp in range(mesh_size_fsdp):
                        grad_rs[dp][fsdp] = grad_ar_out[fsdp][dp] / mesh_size_dp

            # Assign the synchronized gradients to local parameters
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
                        eval_losses.append(f"{loss_val:.2f}")
            print(f"Step {step} losses: {eval_losses}")

            # Simulate a failure on device 3 by using up all its remaining memory.
            if step == 11:
                with torch.cuda.device(3):
                    # Empty PyTorch caching allocator cache first so we know exact driver free bytes
                    torch.cuda.empty_cache()
                    free_bytes, _ = torch.cuda.mem_get_info(3)
                    # Leave 20 MB for driver/runtime overhead so this allocation succeeds,
                    # but leaves cuda:3 with almost 0 bytes free for step 12.
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
                lr = lr * (0.5 ** 2) # sqrt lr rate suggests we use a lower lr.
                # Note: for simplicity, we don't worry about deterministic dataloading on re-running the failed step.
                print(f"Updated mesh after removing flaky slice: {mesh}")

if __name__ == "__main__":
    sys.exit(main())