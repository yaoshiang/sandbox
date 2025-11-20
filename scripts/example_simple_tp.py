"""Simple example: Trace DTensor collectives in a 2-layer MLP with Tensor Parallel.

This is a minimal example showing how to trace torch.compile to see how DTensor
inserts collectives for RowwiseParallel.

Usage:
    torchrun --nproc_per_node=2 scripts/example_simple_tp.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from utils import CompilationTracer, pformat_trace


class SimpleMLP(nn.Module):
    """Minimal 2-layer MLP for TP demonstration."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def main():
    # Setup distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    # Create device mesh
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Create input and model and shard model
    x = torch.distributed.tensor.ones(
        4, 8, device_mesh=device_mesh, dtype=torch.bfloat16
    )
    model = SimpleMLP(dim_in=8, dim_hidden=32, dim_out=8).to(
        device, dtype=torch.bfloat16
    )
    parallelize_module(
        model,
        device_mesh,
        {
            "fc1": ColwiseParallel(),  # No all_reduce needed
            "fc2": RowwiseParallel(),  # Requires all_reduce!
        },
    )

    # Trace compilation to capture FX graphs
    tracer = CompilationTracer()
    compiled_model = torch.compile(model, backend=tracer, fullgraph=True)
    try:
        # Trigger compilation to populate the trace.
        _ = compiled_model(x)
    except torch._dynamo.exc.BackendCompilerFailed:
        # The tracer always raises to prevent execution of a non-lowered graph.
        pass

    # Format and print the trace
    print(pformat_trace(tracer))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
