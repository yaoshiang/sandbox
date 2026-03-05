"""Simple example: Trace DTensor collectives in a 2-layer MLP with Tensor Parallel.

This is a minimal example showing how to trace torch.compile to see how DTensor
inserts collectives for RowwiseParallel.

Usage:
    torchrun --nproc_per_node=2 scripts/example_simple_tp.py

output:
```
(tt) yho_google_com@yho-l4:~/Documents/GitHub/sandbox$ torchrun --nproc_per_node=2 scripts/example_simple_tp.py
W0122 23:07:10.938000 3932495 site-packages/torch/distributed/run.py:803] 
W0122 23:07:10.938000 3932495 site-packages/torch/distributed/run.py:803] *****************************************
W0122 23:07:10.938000 3932495 site-packages/torch/distributed/run.py:803] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0122 23:07:10.938000 3932495 site-packages/torch/distributed/run.py:803] *****************************************

================================================================================
[Rank 1] Dynamo Graph.
================================================================================
graph():
    %l_x_ : torch.distributed.tensor.DTensor [num_users=1] = placeholder[target=L_x_]
    %l_self_modules_fc1_parameters_weight_ : torch.distributed.tensor.DTensor [num_users=1] = placeholder[target=L_self_modules_fc1_parameters_weight_]
    %l_self_modules_fc2_parameters_weight_ : torch.distributed.tensor.DTensor [num_users=1] = placeholder[target=L_self_modules_fc2_parameters_weight_]
    %linear : [num_users=1] = call_function[target=torch._C._nn.linear](args = (%l_x_, %l_self_modules_fc1_parameters_weight_, None), kwargs = {})
    %outputs : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_redistribute](args = (%linear,), kwargs = {})
    %hook_result : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_to_local](args = (%outputs,), kwargs = {})
    %x : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%hook_result,), kwargs = {inplace: False})
    %input_tensor : [num_users=1] = call_function[target=torch._dynamo.variables.torch.prim from_local](args = (%x,), kwargs = {})
    %linear_1 : [num_users=1] = call_function[target=torch._C._nn.linear](args = (%input_tensor, %l_self_modules_fc2_parameters_weight_, None), kwargs = {})
    %outputs_1 : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_redistribute](args = (%linear_1,), kwargs = {})
    %hook_result_1 : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_to_local](args = (%outputs_1,), kwargs = {})
    return (hook_result_1,)
================================================================================

[Rank 1] Dynamo Graph (Generated Code)
--------------------------------------------------------------------------------



def forward(self, L_x_ : torch.distributed.tensor.DTensor, L_self_modules_fc1_parameters_weight_ : torch.distributed.tensor.DTensor, L_self_modules_fc2_parameters_weight_ : torch.distributed.tensor.DTensor):
    l_x_ = L_x_
    l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
    l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
    linear = torch._C._nn.linear(l_x_, l_self_modules_fc1_parameters_weight_, None);  l_x_ = l_self_modules_fc1_parameters_weight_ = None
    outputs = torch__dynamo_variables_tensor_prim_redistribute(linear);  linear = None
    hook_result = torch__dynamo_variables_tensor_prim_to_local(outputs);  outputs = None
    x = torch.nn.functional.relu(hook_result, inplace = False);  hook_result = None
    input_tensor = torch__dynamo_variables_torch_prim_from_local(x);  x = None
    linear_1 = torch._C._nn.linear(input_tensor, l_self_modules_fc2_parameters_weight_, None);  input_tensor = l_self_modules_fc2_parameters_weight_ = None
    outputs_1 = torch__dynamo_variables_tensor_prim_redistribute_1(linear_1);  linear_1 = None
    hook_result_1 = torch__dynamo_variables_tensor_prim_to_local_1(outputs_1);  outputs_1 = None
    return (hook_result_1,)
    
--------------------------------------------------------------------------------

================================================================================
[Rank 1] aot_autograd graph
================================================================================
graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=1] = placeholder[target=primals_2]
    %primals_3 : [num_users=1] = placeholder[target=primals_3]
    %permute_1 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_2, [1, 0]), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%primals_1, %permute_1), kwargs = {})
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%mm_1, [4, 16]), kwargs = {})
    %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view,), kwargs = {})
    %view_1 : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%relu, [4, 16]), kwargs = {})
    %permute_3 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%primals_3, [1, 0]), kwargs = {})
    %mm_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_1, %permute_3), kwargs = {})
    %all_reduce : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%mm_3, sum, 0), kwargs = {})
    %wait_tensor : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce,), kwargs = {})
    %view_2 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%wait_tensor, [4, 8]), kwargs = {})
    return (view_2, primals_1, relu, view_1, permute_3)
================================================================================

[Rank 1] aot_autograd graph (Generated Code)
--------------------------------------------------------------------------------



def forward(self, primals_1, primals_2, primals_3):
    permute_1 = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    mm_1 = torch.ops.aten.mm.default(primals_1, permute_1);  permute_1 = None
    view = torch.ops.aten.view.default(mm_1, [4, 16]);  mm_1 = None
    relu = torch.ops.aten.relu.default(view);  view = None
    view_1 = torch.ops.aten.view.default(relu, [4, 16])
    permute_3 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    mm_3 = torch.ops.aten.mm.default(view_1, permute_3)
    all_reduce = torch.ops._c10d_functional.all_reduce.default(mm_3, 'sum', '0');  mm_3 = None
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
    view_2 = torch.ops.aten.view.default(wait_tensor, [4, 8]);  wait_tensor = None
    return (view_2, primals_1, relu, view_1, permute_3)
    
--------------------------------------------------------------------------------
================================================================================
[Rank 0] Dynamo Graph.
================================================================================
graph():
    %l_x_ : torch.distributed.tensor.DTensor [num_users=1] = placeholder[target=L_x_]
    %l_self_modules_fc1_parameters_weight_ : torch.distributed.tensor.DTensor [num_users=1] = placeholder[target=L_self_modules_fc1_parameters_weight_]
    %l_self_modules_fc2_parameters_weight_ : torch.distributed.tensor.DTensor [num_users=1] = placeholder[target=L_self_modules_fc2_parameters_weight_]
    %linear : [num_users=1] = call_function[target=torch._C._nn.linear](args = (%l_x_, %l_self_modules_fc1_parameters_weight_, None), kwargs = {})
    %outputs : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_redistribute](args = (%linear,), kwargs = {})
    %hook_result : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_to_local](args = (%outputs,), kwargs = {})
    %x : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%hook_result,), kwargs = {inplace: False})
    %input_tensor : [num_users=1] = call_function[target=torch._dynamo.variables.torch.prim from_local](args = (%x,), kwargs = {})
    %linear_1 : [num_users=1] = call_function[target=torch._C._nn.linear](args = (%input_tensor, %l_self_modules_fc2_parameters_weight_, None), kwargs = {})
    %outputs_1 : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_redistribute](args = (%linear_1,), kwargs = {})
    %hook_result_1 : [num_users=1] = call_function[target=torch._dynamo.variables.tensor.prim_to_local](args = (%outputs_1,), kwargs = {})
    return (hook_result_1,)
================================================================================

[Rank 0] Dynamo Graph (Generated Code)
--------------------------------------------------------------------------------



def forward(self, L_x_ : torch.distributed.tensor.DTensor, L_self_modules_fc1_parameters_weight_ : torch.distributed.tensor.DTensor, L_self_modules_fc2_parameters_weight_ : torch.distributed.tensor.DTensor):
    l_x_ = L_x_
    l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
    l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
    linear = torch._C._nn.linear(l_x_, l_self_modules_fc1_parameters_weight_, None);  l_x_ = l_self_modules_fc1_parameters_weight_ = None
    outputs = torch__dynamo_variables_tensor_prim_redistribute(linear);  linear = None
    hook_result = torch__dynamo_variables_tensor_prim_to_local(outputs);  outputs = None
    x = torch.nn.functional.relu(hook_result, inplace = False);  hook_result = None
    input_tensor = torch__dynamo_variables_torch_prim_from_local(x);  x = None
    linear_1 = torch._C._nn.linear(input_tensor, l_self_modules_fc2_parameters_weight_, None);  input_tensor = l_self_modules_fc2_parameters_weight_ = None
    outputs_1 = torch__dynamo_variables_tensor_prim_redistribute_1(linear_1);  linear_1 = None
    hook_result_1 = torch__dynamo_variables_tensor_prim_to_local_1(outputs_1);  outputs_1 = None
    return (hook_result_1,)
    
--------------------------------------------------------------------------------

================================================================================
[Rank 0] aot_autograd graph
================================================================================
graph():
    %primals_1 : [num_users=2] = placeholder[target=primals_1]
    %primals_2 : [num_users=1] = placeholder[target=primals_2]
    %primals_3 : [num_users=1] = placeholder[target=primals_3]
    %permute_1 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%primals_2, [1, 0]), kwargs = {})
    %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%primals_1, %permute_1), kwargs = {})
    %view : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%mm_1, [4, 16]), kwargs = {})
    %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view,), kwargs = {})
    %view_1 : [num_users=2] = call_function[target=torch.ops.aten.view.default](args = (%relu, [4, 16]), kwargs = {})
    %permute_3 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%primals_3, [1, 0]), kwargs = {})
    %mm_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_1, %permute_3), kwargs = {})
    %all_reduce : [num_users=1] = call_function[target=torch.ops._c10d_functional.all_reduce.default](args = (%mm_3, sum, 0), kwargs = {})
    %wait_tensor : [num_users=1] = call_function[target=torch.ops._c10d_functional.wait_tensor.default](args = (%all_reduce,), kwargs = {})
    %view_2 : [num_users=1] = call_function[target=torch.ops.aten.view.default](args = (%wait_tensor, [4, 8]), kwargs = {})
    return (view_2, primals_1, relu, view_1, permute_3)
================================================================================

[Rank 0] aot_autograd graph (Generated Code)
--------------------------------------------------------------------------------



def forward(self, primals_1, primals_2, primals_3):
    permute_1 = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    mm_1 = torch.ops.aten.mm.default(primals_1, permute_1);  permute_1 = None
    view = torch.ops.aten.view.default(mm_1, [4, 16]);  mm_1 = None
    relu = torch.ops.aten.relu.default(view);  view = None
    view_1 = torch.ops.aten.view.default(relu, [4, 16])
    permute_3 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    mm_3 = torch.ops.aten.mm.default(view_1, permute_3)
    all_reduce = torch.ops._c10d_functional.all_reduce.default(mm_3, 'sum', '0');  mm_3 = None
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
    view_2 = torch.ops.aten.view.default(wait_tensor, [4, 8]);  wait_tensor = None
    return (view_2, primals_1, relu, view_1, permute_3)
    
--------------------------------------------------------------------------------
```
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
