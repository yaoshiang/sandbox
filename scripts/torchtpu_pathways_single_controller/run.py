"""Simple example of hsdp usage with a single controller in TorchTPU with Pathways.

This example does not use remote Python, so the CPU client manages data loading and checkpointing. 

Do not use torchrun. This is single-controller. 
"""
import sys
import tempfile
from typing import Final

import torch
import torch.distributed as dist
from torch import nn
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torch.utils.data import DataLoader, TensorDataset

def create_fake_data_and_checkpoint(tempdir, dp:int, fsdp:int) -> None:
    """Create a fake checkpoint (sharded as dp=2, fsdp=4) and data.
    
    Stores the model shards as `model_{dp}_{fsdp}.pt` in the temporary directory.
    Stores the data as `data.pt` in the temporary directory.

    Args:
        tempdir: Temporary directory to store the checkpoint and data.
        dp: Number of data parallel shards.
        fsdp: Number of FSDP shards.
    """
    assert torch.cuda.current_device() == "cpu"

    # Create fake model weights
    torch.manual_seed(0)
    model_cpu = nn.Linear(2048, 8192, bias=False)

    for dp_ in range(dp):
        for fsdp_ in range(fsdp):
            model_shard_cpu = nn.Linear(2048 // fsdp, 8192, bias=False)
            slice_size = 2048 // fsdp
            slice_start = fsdp_ * slice_size
            slice_end = slice_start + slice_size
            model_shard_cpu.state_dict()["weight"][:] = model_cpu.state_dict()["weight"][slice_start:slice_end, ...]
            
            name = f"{tempdir.name}/model_dp{dp_}_fsdp{fsdp_}"
            torch.save(model_shard_cpu.state_dict(), f"{name}.pt")

    model_true = nn.Linear(2048, 8192, bias=False)
    x = torch.randn(32, 2048)
    y = model_true(x)
    torch.save((x, y), f"{tempdir.name}/data.pt")

    return

class _HSDPLinearFunction(torch.autograd.Function):
    """A custom autograd function for HSDP with collective activation checkpointing.

    Forward:
      1. All-Gather (AG) weight_shard along FSDP axis to assemble full weight (2048, 8192).
      2. Compute y = x @ gathered_weight.
      3. Save only x and weight_shard for backward, discarding gathered_weight (frees full weight memory).

    Backward (Activation Checkpointing for Collectives):
      1. Re-All-Gather weight_shard along FSDP axis to reconstruct full weight.
      2. Compute grad_x = grad_output @ gathered_weight.T (dL/dx).
      3. Discard gathered_weight again.
      4. Compute grad_gathered_weight = x.T @ grad_output (dL/dW_full).
      5. Reduce-Scatter (RS) grad_gathered_weight along FSDP axis to obtain local grad_weight_shard.
      6. All-Reduce (AR) grad_weight_shard across DP replicas with averaging.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight_shard: torch.Tensor, mesh: dist.DeviceMesh) -> torch.Tensor:
        ctx.mesh = mesh
        # Save x and weight_shard; discard gathered_weight so memory is not held during forward activations
        ctx.save_for_backward(x, weight_shard)

        # Forward AG along FSDP axis
        gathered_weight = torch.empty(2048, 8192, device=weight_shard.device, dtype=weight_shard.dtype)
        dist.all_gather_into_tensor(gathered_weight, weight_shard, group=mesh["fsdp"].get_group())

        # Forward GEMM: (B, 2048) @ (2048, 8192) -> (B, 8192)
        y = x @ gathered_weight

        # gathered_weight goes out of scope here and is freed from VRAM
        del gathered_weight
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight_shard = ctx.saved_tensors
        mesh = ctx.mesh

        # 1. Collective rematerialization (Re-AG) on backward
        gathered_weight = torch.empty(2048, 8192, device=weight_shard.device, dtype=weight_shard.dtype)
        dist.all_gather_into_tensor(gathered_weight, weight_shard, group=mesh["fsdp"].get_group())

        # 2. Compute input gradient: (B, 8192) @ (8192, 2048) -> (B, 2048)
        grad_x = grad_output @ gathered_weight.t()
        # Discard gathered_weight again
        del gathered_weight

        # 3. Compute full weight gradient: (2048, B) @ (B, 8192) -> (2048, 8192)
        grad_gathered_weight = x.t() @ grad_output

        # 4. FSDP Backward: Reduce-Scatter full gradient along FSDP axis
        fsdp_size = mesh.size("fsdp")
        shard_in_features = 2048 // fsdp_size
        out_features = 8192

        grad_weight_shard = torch.empty(shard_in_features, out_features, device=grad_output.device, dtype=grad_output.dtype)
        dist.reduce_scatter_tensor(grad_weight_shard, grad_gathered_weight, group=mesh["fsdp"].get_group())

        # 5. DP Backward: All-Reduce the scattered gradient across DP replicas
        if mesh.size("dp") > 1:
            dist.all_reduce(grad_weight_shard, group=mesh["dp"].get_group(), op=dist.ReduceOp.AVG)

        # These map to the arguments to forward. 
        return grad_x, grad_weight_shard, None


class HSDPLinear(nn.Module):
    def __init__(self, mesh: dist.DeviceMesh):
        """HSDP aware linear layer. 

        Assumes global size of 2048, 8192 like a FFN1 layer.

        Uses collective activation checkpointing (Re-AG on backward). 
        """
        super().__init__()
        self.mesh = mesh
        self.weight = nn.Parameter(torch.randn(2048 // mesh.size("fsdp"), 8192))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """HSDP aware linear layer with collective activation checkpointing.
        
        On forward, AG on the FSDP axis to compute y, then frees the gathered weight.
        On backward, Re-AGs the weight to compute dL/dx, RS on FSDP for dL/dW, and AR on DP.
        """
        return _HSDPLinearFunction.apply(x, self.weight, self.mesh)

def train_step(model: HSDPLinear, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Forward pass
    y_pred = model(x)
    # Compute loss (mean squared error for example)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    # Backward pass
    loss.backward()
    return loss

def tree_zip(tree):
    """Zips a pytree of iterables, yielding a matching pytree of elements at each step.

    Flattens the pytree of DataLoaders into leaf iterators, uses `zip(*leaves)` to step
    through all shards in lockstep until the dataloaders run out, and unflattens the items
    back into the original nested mesh structure.
    """
    leaves, spec = tree_flatten(tree)
    for items in zip(*leaves):
        yield tree_unflatten(items, spec)


def train(tempdir, dp_size: int, fsdp_size: int):
    """A train function. 
    
    This is not compiled and is not run on each rank. 
    
    It is run only the on the Pathways client (single-controller). 
    """
    # Initialize the process group. 
    # Unlike the "tpu" backend, which sets up multi-controller, this
    # backend allows single-controller operation.
    dist.init_process_group(backend="tpu_pathways")

    # Create the mesh.
    # In reality on TPUs, topology-aware placement scrambles physical rank assignments
    # along 2D/3D torus ICI interconnects to maximize bandwidth, resulting in a non-sequential layout:
    # e.g., rank_mesh = tensor([[0, 2, 5, 7], [1, 3, 4, 6]]) instead of trivial [[0, 1, 2, 3], [4, 5, 6, 7]].
    rank_mesh: "torch.Tensor dtype=int64 shape=2x4" = torch.tpu.distributed.topology_aware_mesh(dp_size, fsdp_size)
    # preserve_rank_order=True preserves the exact topology-aware scrambled rank mapping.
    mesh = dist.DeviceMesh("tpu", rank_mesh, ["dp", "fsdp"], preserve_rank_order=True)

    # Create the model shards mapped across the 2D mesh grid and load their corresponding checkpoints.
    # Why `rank_mesh.tolist()`: PyTorch pytrees treat `torch.Tensor` as a single indivisible
    # leaf node rather than a container. Calling `.tolist()` converts the (2, 4) tensor into
    # a nested Python list `[[r0, r1, r2, r3], [r4, r5, r6, r7]]`, exposing the tree structure.
    #
    # How `tree_map` works: `tree_map(fn, tree)` recursively traverses the container (lists/tuples/dicts),
    # applies `fn` to every scalar leaf (the rank ID), and reconstructs a new container
    # with the exact same nested (2, 4) topology without requiring manual nested loops.
    def load_model_shard(rank_id: int) -> HSDPLinear:
        shard = HSDPLinear(mesh)
        dp_idx, fsdp_idx = mesh.get_coordinate_for_rank(rank_id)
        state_dict = torch.load(f"{tempdir.name}/model_dp{dp_idx}_fsdp{fsdp_idx}.pt")
        shard.load_state_dict(state_dict)
        # In single-controller TorchTPU/Pathways, each shard is moved to its target TPU device
        return shard.to(f"tpu:{rank_id}")

    model_shards = tree_map(load_model_shard, rank_mesh.tolist())

    # Create a tree of dataloaders. 
    x_full, y_full = torch.load(f"{tempdir.name}/data.pt")
    assert x_full.size(0) == y_full.size(0)

    per_device_len = x_full.size(0) // (dp_size * fsdp_size)

    def slice_data_shard(rank_id: int, tensor: torch.Tensor) -> torch.Tensor:
        dp_idx, fsdp_idx = mesh.get_coordinate_for_rank(rank_id)
        flat_idx = dp_idx * fsdp_size + fsdp_idx
        slice_start = flat_idx * per_device_len
        return tensor[slice_start : slice_start + per_device_len]

    x_shards = tree_map(lambda r: slice_data_shard(r, x_full), rank_mesh.tolist())
    y_shards = tree_map(lambda r: slice_data_shard(r, y_full), rank_mesh.tolist())

    # Determine batch size per device (e.g. 2 samples per step per device)
    step_batch_size = 2
    # Create nested (2, 4) TensorDatasets and DataLoaders on CPU host using tree_map
    dataset_shards = tree_map(lambda xs, ys: TensorDataset(xs, ys), x_shards, y_shards)
    dataloader_shards = tree_map(lambda ds: DataLoader(ds, batch_size=step_batch_size, shuffle=False), dataset_shards)

    # By setting the "tpu_pathways_mesh" options dict key to the mesh, 
    # we ask torch to compile the function for the TPU Pathways backend.
    # 
    # This includes a function transformation so that each argument is transformed into a nested sequence 
    # that matches the shape of the mesh. 
    # 
    # This nested sequence must cover all ranks in the mesh. Internally xla::ifrt::Array with
    # SingleDeviceSharding will be unified into a xla::ifrt::Array with ConcreteEvenSharding. 

    compiled_train_step = torch.compile(backend="tpu", fn=train_step, options={"tpu_pathways_mesh": mesh})

    def save_model_shard(rank_id: int, shard: HSDPLinear) -> None:
        dp_idx, fsdp_idx = mesh.get_coordinate_for_rank(rank_id)
        torch.save(shard.to("cpu").state_dict(), f"{tempdir.name}/model_shard_dp{dp_idx}_fsdp{fsdp_idx}.pt")

    for epoch in range(1, 101):
        # Iterate over all mini-batches until the DataLoaders run out using tree_zip
        for batch_shards in tree_zip(dataloader_shards):
            # Extract CPU batches, then transfer each shard to its corresponding TPU device (tpu:{rank_id})
            batch_x_cpu = tree_map(lambda b: b[0], batch_shards)
            batch_y_cpu = tree_map(lambda b: b[1], batch_shards)
            batch_x = tree_map(lambda r, bx: bx.to(f"tpu:{r}"), rank_mesh.tolist(), batch_x_cpu)
            batch_y = tree_map(lambda r, by: by.to(f"tpu:{r}"), rank_mesh.tolist(), batch_y_cpu)

            loss = compiled_train_step(model_shards, batch_x, batch_y)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}, Loss: {loss}")
            # Checkpoint each shard using tree_map with logical coordinates
            tree_map(save_model_shard, rank_mesh.tolist(), model_shards) 

def main() -> int:
    # This is runtime setup stuff. 
    dist.init_process_group(backend="tpu_pathways")

    # Deterministically create the fake model weights and data on disk. 
    tempdir = tempfile.TemporaryDirectory()
    create_fake_data_and_checkpoint(tempdir=tempdir, dp=2, fsdp=4)

    # Train model across single-controller TPU Pathways client
    train(tempdir=tempdir, dp_size=2, fsdp_size=4)

    return 0


if __name__ == "__main__":
    sys.exit(main())