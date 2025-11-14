"""Utilities for tracing torch.compile graphs to inspect distributed operations.

This module provides tools to trace how PyTorch compiles distributed operations,
particularly useful for understanding DTensor's collective insertion and decomposition.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch._inductor.decomposition
import torch._inductor.compile_fx
from torch._dynamo.backends.common import aot_autograd
import torch.distributed as dist


@dataclass
class CompilationTrace:
    """Captured FX graphs from different compilation stages.

    Attributes:
    """


class CompilationTracer:
    """Traces torch.compile execution to capture FX graphs at different stages.

    Attributes:
        high_level_graph: FX graph before decomposition (shows prim_* operations)
        lowered_graph: FX graph after decomposition (shows c10d_functional collectives)
        rank: Distributed rank where trace was captured
    """

    def __init__(self):
        self.high_level_graph: Optional[torch.fx.GraphModule] = None
        self.lowered_graph: Optional[torch.fx.GraphModule] = None
        self.rank: int = dist.get_rank() if dist.is_initialized() else 0

    def create_backend(
        self,
    ) -> Callable[[torch.fx.GraphModule, list[Any]], Callable]:
        """Create a custom torch.compile backend that traces compilation stages.

        Returns a backend function compatible with torch.compile(backend=...).
        The backend traces graphs during compilation but still produces optimized code.
        """

        def tracing_backend(gm: torch.fx.GraphModule, example_inputs):
            # Trace high-level graph (before decomposition)
            self.high_level_graph = gm

            # Create a fw_compiler that traces the lowered graph then compiles it
            def trace_and_compile(gm_inner: torch.fx.GraphModule, example_inputs_inner):
                # Trace the lowered graph (after decomposition)
                self.lowered_graph = gm_inner
                # Then actually compile it so the model runs with optimizations
                return torch._inductor.compile_fx.compile_fx(
                    gm_inner, example_inputs_inner
                )

            # Use Inductor's decomposition table to lower DTensor primitives
            # into concrete collective operations (e.g., prim_redistribute -> all_reduce)
            aot_backend = aot_autograd(
                fw_compiler=trace_and_compile,
                decompositions=torch._inductor.decomposition.select_decomp_table(),
            )

            return aot_backend(gm, example_inputs)

        return tracing_backend


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

    # High-level graph (Dynamo output / pre-decomposition)
    if trace.high_level_graph:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"[Rank {rank}] Dynamo Graph (Pre-Decomposition)")
        lines.append("=" * 80)
        lines.append(str(trace.high_level_graph.graph))
        lines.append("=" * 80)

        lines.append("")
        lines.append(f"[Rank {rank}] Dynamo Graph (Generated Code)")
        lines.append("-" * 80)
        lines.append(trace.high_level_graph.code)
        lines.append("-" * 80)

    # Lowered graph (AOTAutograd output / post-decomposition)
    if trace.lowered_graph:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"[Rank {rank}] ATen Graph (Post-Decomposition)")
        lines.append("=" * 80)
        lines.append(str(trace.lowered_graph.graph))
        lines.append("=" * 80)

        lines.append("")
        lines.append(f"[Rank {rank}] ATen Graph (Generated Code)")
        lines.append("-" * 80)
        lines.append(trace.lowered_graph.code)
        lines.append("-" * 80)
    return "\n".join(lines)
