"""Utilities for tracing torch.compile graphs to inspect distributed operations.

This module provides tools to trace how PyTorch compiles distributed operations,
particularly useful for understanding DTensor's collective insertion and decomposition.
"""

from typing import Optional
import torch._inductor.decomposition
import torch._inductor.compile_fx
from torch._dynamo.backends.common import aot_autograd
import torch.distributed as dist


class CompilationTracer:
    """Traces torch.compile execution to capture FX graphs at different stages.

    Attributes:
        high_level_graph: FX graph before decomposition (shows prim_* operations)
        lowered_graph: FX graph after decomposition (shows c10d_functional collectives)
        rank: Distributed rank where trace was captured
    """

    def __init__(self):
        self.dynamo_graph: Optional[torch.fx.GraphModule] = None
        self.aot_autograd_graph: Optional[torch.fx.GraphModule] = None
        self.rank: int = dist.get_rank() if dist.is_initialized() else 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        """Custom torch.compile backend that stores dyanamo and aotautograd graphs.

        Compatible with torch.compile(backend=tracer_instance).
        Traces graphs during compilation but still produces optimized code.

        Raises:
            NotImplementedError: Always raised to prevent actual execution.
        """
        # Stash dynamo graph.
        self.dynamo_graph = gm

        # Create a fw_compiler that stashes the aot_autograd lowered graph.
        def trace_and_compile(gm_inner: torch.fx.GraphModule, example_inputs_inner):
            # Trace the lowered graph (after decomposition)
            self.aot_autograd_graph = gm_inner
            return torch._inductor.compile_fx.compile_fx(gm_inner, example_inputs_inner)

        # Use Inductor's decomposition table to lower DTensor primitives
        # into concrete collective operations (e.g., prim_redistribute -> all_reduce)
        backend = aot_autograd(
            fw_compiler=trace_and_compile,
            decompositions=torch._inductor.decomposition.select_decomp_table(),
        )

        # Actually invoke the backend to trigger the fw_compiler
        backend(gm, example_inputs)

        raise NotImplementedError("Graphs stashed. Execution prevented")


def pformat_trace(trace: CompilationTracer) -> str:
    """Pretty-format a compilation trace as a string.

    Args:
        trace: The compilation trace to format
        verbose: If True, include generated Python code for each graph

    Returns:
        Formatted string suitable for printing or logging
    """
    lines = []
    rank = trace.rank

    # dynamo graph
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"[Rank {rank}] Dynamo Graph.")
    lines.append("=" * 80)
    lines.append(str(trace.dynamo_graph.graph))
    lines.append("=" * 80)

    lines.append("")
    lines.append(f"[Rank {rank}] Dynamo Graph (Generated Code)")
    lines.append("-" * 80)
    lines.append(trace.dynamo_graph.code)
    lines.append("-" * 80)

    # aot_autograd graph
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"[Rank {rank}] aot_autograd graph")
    lines.append("=" * 80)
    lines.append(str(trace.aot_autograd_graph.graph))
    lines.append("=" * 80)

    lines.append("")
    lines.append(f"[Rank {rank}] aot_autograd graph (Generated Code)")
    lines.append("-" * 80)
    lines.append(trace.aot_autograd_graph.code)
    lines.append("-" * 80)

    return "\n".join(lines)
